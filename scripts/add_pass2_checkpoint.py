"""Add checkpoint/resume to Pass 2 (save_topk_images).

Strategy:
- Checkpoint saves the set of keys already written to disk.
- On resume, those keys are subtracted from `remaining`.
- Uses the same atomic write pattern as Pass 1.
- Saves checkpoint every SAVE_EVERY batches (from CFG).
"""
import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "notebooks"
    / "Segment_3_canonical.ipynb"
)

# ── New checkpoint cell source (replaces old checkpoint cell) ──
NEW_CKPT_CELL = r'''"""Checkpoint save/load with atomic writes and corruption recovery."""

CKPT_PATH = CFG["CKPT_DIR"] / "topk_scan_ckpt.pkl"
PASS2_CKPT_PATH = CFG["CKPT_DIR"] / "pass2_save_ckpt.pkl"


def save_ckpt(
    path: Path,
    step: int,
    heaps: dict,
    ch0: int,
    ch1: int,
) -> None:
    """Save checkpoint atomically (write-then-rename).

    Writes to a temporary file first, then performs an atomic rename
    so a crash mid-write never corrupts the checkpoint.

    Args:
        path: Checkpoint file path.
        step: Current batch step index.
        heaps: Dictionary mapping channel_id -> min-heap list.
        ch0: Start of channel range.
        ch1: End of channel range (inclusive).
    """
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pkl.dump(
            {"step": step, "heaps": heaps, "ch0": ch0, "ch1": ch1},
            f,
            protocol=pkl.HIGHEST_PROTOCOL,
        )
    os.replace(tmp_path, path)  # Atomic on all platforms


def load_ckpt(path: Path) -> dict | None:
    """Load checkpoint, returning None if missing or corrupt.

    Args:
        path: Checkpoint file path.

    Returns:
        Checkpoint dict, or None if missing/corrupt.
    """
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pkl.load(f)
        return data
    except (pkl.UnpicklingError, EOFError, ValueError, KeyError) as exc:
        print(f"Warning: Corrupt checkpoint ({exc}), starting fresh.")
        return None


def save_pass2_ckpt(path: Path, saved_keys: set) -> None:
    """Save Pass 2 checkpoint (set of already-saved image keys).

    Args:
        path: Checkpoint file path.
        saved_keys: Set of WDS keys that have been saved to disk.
    """
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pkl.dump({"saved_keys": saved_keys}, f, protocol=pkl.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def load_pass2_ckpt(path: Path) -> set:
    """Load Pass 2 checkpoint, returning empty set if missing/corrupt.

    Args:
        path: Checkpoint file path.

    Returns:
        Set of previously saved image keys.
    """
    if not path.exists():
        return set()
    try:
        with open(path, "rb") as f:
            data = pkl.load(f)
        keys = data.get("saved_keys", set())
        if isinstance(keys, set):
            return keys
        return set()
    except (pkl.UnpicklingError, EOFError, ValueError, KeyError) as exc:
        print(f"Warning: Corrupt Pass 2 checkpoint ({exc}), starting fresh.")
        return set()


print(f"Pass 1 checkpoint: {CKPT_PATH} (exists={CKPT_PATH.exists()})")
print(f"Pass 2 checkpoint: {PASS2_CKPT_PATH} (exists={PASS2_CKPT_PATH.exists()})")
'''

# ── New Pass 2 code cell ──
NEW_PASS2_CELL = r'''"""Pass 2: Re-stream dataset and save top-k images to disk (with resume)."""


def map_top_keys(
    topk_by_channel: dict,
) -> tuple[set, dict]:
    """Build lookup structures from the top-k results.

    Args:
        topk_by_channel: Dict mapping channel_id -> sorted list of
            (score, key) tuples.

    Returns:
        Tuple of:
        - wanted_keys: Set of all unique image keys to find.
        - key_to_targets: Dict mapping key -> list of
          (channel, score, rank) tuples.
    """
    wanted_keys = set()
    key_to_targets = {}  # key -> list[(channel, score, rank)]

    for c, items in topk_by_channel.items():
        for rank, (score, key) in enumerate(items, start=1):
            wanted_keys.add(key)
            key_to_targets.setdefault(key, []).append(
                (c, float(score), rank)
            )

    return wanted_keys, key_to_targets


def save_topk_images(
    model: torch.nn.Module,
    device: torch.device,
    layer_name: str,
    topk_by_channel: dict,
    out_root: Path,
    crop_frac: float,
    batch_size: int,
    save_every: int,
) -> None:
    """Re-stream dataset and save top-k images to disk.

    Supports checkpoint/resume: if interrupted, re-running will skip
    images that were already saved in a previous run.

    Args:
        model: Pretrained InceptionV1 model in eval mode.
        device: Torch device.
        layer_name: Target layer name.
        topk_by_channel: Results from scan_topk_keys().
        out_root: Root directory for saved images.
        crop_frac: Crop fraction for spatial cropping.
        batch_size: DataLoader batch size.
        save_every: Checkpoint save frequency (in batches).
    """
    # Build lookup maps
    wanted_keys, key_to_targets = map_top_keys(topk_by_channel)
    total_keys = len(wanted_keys)

    # Load Pass 2 checkpoint (resume support)
    previously_saved = load_pass2_ckpt(PASS2_CKPT_PATH)
    if previously_saved:
        print(f"Resuming Pass 2: {len(previously_saved)} keys already saved.")

    # Subtract already-saved keys
    remaining = wanted_keys - previously_saved
    saved_keys = set(previously_saved)  # mutable copy for tracking
    print(f"Looking for {len(remaining)} keys ({total_keys} total, "
          f"{len(saved_keys)} already saved).")

    if not remaining:
        print("All keys already saved! Nothing to do.")
        return

    # Hook setup
    activation = {}

    def hook_fn(module, inp, out):
        """Forward hook to capture layer activations."""
        activation["feat"] = out

    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)

    layer_dir = out_root / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.eval()

        # Fresh DataLoader (WDS iterators are single-use)
        dl_train = make_dataloader(batch_size=batch_size)

        for step, (imgs, keys) in enumerate(dl_train):
            imgs = imgs.to(device, non_blocking=True)

            use_cuda = imgs.is_cuda
            with torch.amp.autocast(
                device_type="cuda", enabled=use_cuda,
            ):
                _ = model(imgs)

            feat = activation["feat"]  # [B, C, h, w]

            # Find which indices in this batch we need
            hit_indices = [
                i for i, k in enumerate(keys) if k in remaining
            ]

            if not hit_indices:
                del feat
                if (step + 1) % 200 == 0:
                    print(
                        f"  batch {step + 1:>6d} | "
                        f"remaining={len(remaining)}/{total_keys}"
                    )
                continue

            for i in hit_indices:
                key = keys[i]
                key_safe = _safe_filename(key)

                # Full-size de-normalized image (computed once per sample)
                pil_full = tensor_to_pil(imgs[i])

                # Save for each channel that selected this key
                for (c, score, rank) in key_to_targets[key]:
                    ch_dir = layer_dir / f"ch_{c:04d}"
                    ch_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        pil_crop = crop_from_tensor_and_feat(
                            imgs[i], feat[i], c, frac=crop_frac,
                        )
                        full_path = (
                            ch_dir
                            / f"rank{rank:02d}_FULL_"
                              f"score{score:.4f}_{key_safe}.jpg"
                        )
                        crop_path = (
                            ch_dir
                            / f"rank{rank:02d}_CROP_"
                              f"score{score:.4f}_{key_safe}.jpg"
                        )
                        pil_full.save(full_path)
                        pil_crop.save(crop_path)
                    except (OSError, IOError) as exc:
                        print(
                            f"  Warning: Failed to save ch{c}/rank{rank} "
                            f"key={key_safe}: {exc}"
                        )
                        continue

                remaining.discard(key)
                saved_keys.add(key)

            # Free activation memory
            del feat

            # Progress report
            if (step + 1) % 50 == 0:
                num_saved = total_keys - len(remaining)
                print(
                    f"  batch {step + 1:>6d} | "
                    f"saved={num_saved}/{total_keys} "
                    f"remaining={len(remaining)}"
                )

            # Periodic checkpoint save
            if (step + 1) % save_every == 0:
                save_pass2_ckpt(PASS2_CKPT_PATH, saved_keys)
                print(f"  Checkpoint saved at batch {step + 1} "
                      f"({len(saved_keys)} keys saved)")

            # Early exit when all keys found
            if not remaining:
                print(f"  All keys found! Done at batch {step + 1}.")
                break

    finally:
        handle.remove()

        # Always save final checkpoint
        save_pass2_ckpt(PASS2_CKPT_PATH, saved_keys)
        print(f"Hook removed. Final checkpoint saved "
              f"({len(saved_keys)} keys).")
        print(f"Output -> {layer_dir}")

    if remaining:
        print(
            f"Warning: {len(remaining)} keys not found in dataset "
            f"(possible streaming/shard mismatch)."
        )
'''

# ── New Pass 2 markdown ──
NEW_PASS2_MD = [
    "## 8. Pass 2 -- Save Top-K Images (with Resume)\n",
    "\n",
    "Re-stream the dataset, match keys found in Pass 1, and save:\n",
    "\n",
    "- **FULL**: Full 224x224 de-normalized image.\n",
    "- **CROP**: Spatially cropped patch around max-activation region.\n",
    "\n",
    "### Checkpoint/Resume\n",
    "\n",
    "Pass 2 saves its own checkpoint tracking which keys have been\n",
    "written to disk. If interrupted, re-running will skip already-saved\n",
    "images automatically.\n",
    "\n",
    "### Edge Cases Handled\n",
    "\n",
    "| Edge Case | Mitigation |\n",
    "|-----------|------------|\n",
    "| Interrupted mid-save | Checkpoint tracks saved keys, resume skips them |\n",
    "| Per-image save failure | `try/except` per image, log & continue |\n",
    "| WDS DataLoader exhausted | Fresh `make_dataloader()` call |\n",
    "| Hook not removed on error | `try/finally` block |\n",
    "| All keys found early | Early exit + final checkpoint save |\n",
]

# ── New run cell ──
NEW_RUN_CELL = r'''"""Execute the full two-pass pipeline."""

print("=" * 60)
print("PASS 1: Scanning top-k activations")
print("=" * 60)

topk_by_channel = scan_topk_keys(
    model=model,
    device=device,
    layer_name=CFG["LAYER_NAME"],
    channel_start=CFG["CHANNEL_START"],
    channel_end_inclusive=CFG["CHANNEL_END"],
    topk=CFG["TOPK"],
    heap_batch=CFG["HEAP_BATCH"],
    reduction=CFG["REDUCTION"],
    batch_size=CFG["BATCH_SIZE"],
    save_every=CFG["SAVE_EVERY"],
)

# Free GPU memory between passes
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
    free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"GPU free after Pass 1 cleanup: {free_mem:.1f} GB")

print()
print("=" * 60)
print("PASS 2: Saving images to disk")
print("=" * 60)

save_topk_images(
    model=model,
    device=device,
    layer_name=CFG["LAYER_NAME"],
    topk_by_channel=topk_by_channel,
    out_root=CFG["OUT_ROOT"],
    crop_frac=CFG["CROP_FRAC"],
    batch_size=CFG["BATCH_SIZE"],
    save_every=CFG["SAVE_EVERY"],
)

print()
print("=" * 60)
print("Pipeline complete!")
print(f"  Results saved to: {CFG['OUT_ROOT']}")
print("=" * 60)
'''


def _to_source_lines(code_str: str) -> list[str]:
    """Convert a code string to JSON source array format."""
    lines = code_str.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        elif line:
            result.append(line)
    return result


def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updates = []

    for idx, cell in enumerate(nb["cells"]):
        joined = "".join(cell["source"])

        # Replace checkpoint cell
        if cell["cell_type"] == "code" and "CKPT_PATH = CFG" in joined:
            cell["source"] = _to_source_lines(NEW_CKPT_CELL)
            cell["outputs"] = []
            cell["execution_count"] = None
            updates.append(f"Replaced checkpoint cell (index {idx})")

        # Replace Pass 2 markdown
        if (cell["cell_type"] == "markdown"
                and "Pass 2" in joined
                and "Save Top-K" in joined):
            cell["source"] = NEW_PASS2_MD
            updates.append(f"Replaced Pass 2 markdown (index {idx})")

        # Replace Pass 2 code
        if (cell["cell_type"] == "code"
                and "def save_topk_images" in joined):
            cell["source"] = _to_source_lines(NEW_PASS2_CELL)
            cell["outputs"] = []
            cell["execution_count"] = None
            updates.append(f"Replaced Pass 2 code cell (index {idx})")

        # Replace Run cell
        if (cell["cell_type"] == "code"
                and "PASS 1: Scanning" in joined
                and "save_topk_images" in joined):
            cell["source"] = _to_source_lines(NEW_RUN_CELL)
            cell["outputs"] = []
            cell["execution_count"] = None
            updates.append(f"Replaced Run cell (index {idx})")

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    for u in updates:
        print(u)
    print(f"Total: {len(updates)} cells updated.")


if __name__ == "__main__":
    main()
