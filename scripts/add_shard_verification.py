"""Insert shard integrity verification cells into Segment_3_canonical.ipynb.

Replaces the old Section 4b (naive assert-based check) with a comprehensive
4-layer verification system: presence, size match, tar integrity, JPEG decode.
Also adds a separate DataLoader smoke-test cell (Section 4c).
"""

import json
from pathlib import Path


NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "Segment_3_canonical.ipynb"
)


# ── New markdown cell: Section 4b ──
MD_4B = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4b. Shard Integrity Verification\n",
        "\n",
        "Before running the pipeline, we perform a **4-layer integrity check**\n",
        "on every downloaded shard. This is essential because the ~144 GB\n",
        "download can be interrupted mid-file, leaving truncated or corrupted\n",
        "`.tar` archives that would cause silent errors during Pass 1/2.\n",
        "\n",
        "### Verification Layers\n",
        "\n",
        "| Layer | Check | What it catches |\n",
        "|-------|-------|------------------|\n",
        "| **1. Presence** | All 1024 shard numbers exist on disk | Missing shards (not yet downloaded) |\n",
        "| **2. Size match** | Local file size == HuggingFace LFS expected size | Truncated / partial downloads |\n",
        "| **3. Tar integrity** | `tarfile.open()` + iterate all members | Structurally corrupt archives |\n",
        "| **4. Decode spot-check** | Decode first JPEG from every 50th shard | Corrupt image data inside valid tars |\n",
        "\n",
        "The output is a `HEALTHY_SHARDS` list containing **only verified**\n",
        "shard paths. The `make_dataloader()` factory uses this list, so the\n",
        "pipeline never touches a bad file.\n",
        "\n",
        "> **Note**: This cell is **re-runnable**. As more shards finish\n",
        "> downloading, re-run to pick them up and extend the healthy set.\n",
    ],
}


# ── New code cell: 4-layer verification ──
CODE_4B = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        '"""4-layer shard integrity verification.\n',
        "\n",
        "Checks presence, file size, tar structure, and sample JPEG decode\n",
        "for every local shard. Produces a HEALTHY_SHARDS list for the\n",
        "downstream pipeline.\n",
        '"""\n',
        "import io\n",
        "import json as _json\n",
        "import tarfile\n",
        "from urllib.request import urlopen, Request\n",
        "\n",
        "# ── Configuration ──\n",
        "EXPECTED_SHARD_COUNT = 1024\n",
        "SPOT_CHECK_EVERY = 50  # Decode-check every Nth shard\n",
        "HF_API_URL = (\n",
        '    "https://huggingface.co/api/datasets/"\n',
        '    "timm/imagenet-1k-wds/tree/main"\n',
        ")\n",
        "\n",
        "\n",
        "# ── Layer 1: Presence Check ──\n",
        "# Build the set of expected shard filenames (0000 to 1023)\n",
        "expected_names = {\n",
        '    f"imagenet1k-train-{i:04d}.tar"\n',
        "    for i in range(EXPECTED_SHARD_COUNT)\n",
        "}\n",
        "\n",
        "# Map each present shard filename to its full path\n",
        "local_shard_map = {\n",
        "    Path(p).name: p for p in _LOCAL_SHARDS\n",
        "}\n",
        "present_names = set(local_shard_map.keys())\n",
        "\n",
        "# Identify missing and unexpected files\n",
        "missing_names = sorted(expected_names - present_names)\n",
        "unexpected_names = sorted(present_names - expected_names)\n",
        "\n",
        'print("=" * 60)\n',
        'print("LAYER 1: Presence Check")\n',
        'print("=" * 60)\n',
        'print(f"  Expected:   {EXPECTED_SHARD_COUNT}")\n',
        'print(f"  Present:    {len(present_names)}")\n',
        'print(f"  Missing:    {len(missing_names)}")\n',
        "if missing_names:\n",
        "    # Show first 10 missing shards for brevity\n",
        "    preview = missing_names[:10]\n",
        "    suffix = (\n",
        '        f" ... and {len(missing_names) - 10} more"\n',
        "        if len(missing_names) > 10\n",
        '        else ""\n',
        "    )\n",
        '    print(f"  Examples:   {preview}{suffix}")\n',
        "if unexpected_names:\n",
        '    print(f"  Unexpected: {unexpected_names}")\n',
        "\n",
        "\n",
        "# ── Layer 2: Size Match against HuggingFace LFS Metadata ──\n",
        "print()\n",
        'print("=" * 60)\n',
        'print("LAYER 2: Size Match (HuggingFace LFS)")\n',
        'print("=" * 60)\n',
        "\n",
        "# Fetch expected sizes from HuggingFace API\n",
        'print("  Fetching expected sizes from HuggingFace API...")\n',
        "expected_sizes = {}  # filename -> expected bytes\n",
        "try:\n",
        "    # The tree API is paginated; fetch all pages\n",
        "    cursor = None\n",
        "    while True:\n",
        "        url = HF_API_URL\n",
        "        if cursor:\n",
        '            url += f"?cursor={cursor}"\n',
        "        req = Request(url)\n",
        '        req.add_header("Authorization", f"Bearer {HF_TOKEN}")\n',
        "        with urlopen(req, timeout=30) as resp:\n",
        "            page = _json.loads(resp.read())\n",
        "        for entry in page:\n",
        '            name = entry.get("path", "")\n',
        "            if (\n",
        '                name.startswith("imagenet1k-train-")\n',
        '                and name.endswith(".tar")\n',
        "            ):\n",
        "                # Use LFS size (authoritative) if available,\n",
        "                # otherwise fall back to top-level size\n",
        '                lfs = entry.get("lfs", {})\n',
        '                size = lfs.get("size", entry.get("size", 0))\n',
        "                expected_sizes[name] = int(size)\n",
        "\n",
        "        # Check for next page or break\n",
        "        if len(page) < 50:\n",
        "            break\n",
        "        # HuggingFace uses the last item's path as cursor\n",
        '        cursor = page[-1].get("path", "")\n',
        "\n",
        '    print(f"  Fetched sizes for {len(expected_sizes)} shards.")\n',
        "except Exception as exc:\n",
        '    print(f"  ⚠ Could not fetch HF metadata: {exc}")\n',
        '    print("  Falling back to size-range heuristic only.")\n',
        "\n",
        "# Compare local sizes against expected\n",
        "size_matched = []     # Paths that match expected size\n",
        "size_mismatched = []  # (filename, local_size, expected_size)\n",
        "size_unknown = []     # Present but no expected size available\n",
        "\n",
        "for name in sorted(present_names & expected_names):\n",
        "    local_path = local_shard_map[name]\n",
        "    local_size = Path(local_path).stat().st_size\n",
        "\n",
        "    if name in expected_sizes:\n",
        "        if local_size == expected_sizes[name]:\n",
        "            size_matched.append(local_path)\n",
        "        else:\n",
        "            size_mismatched.append(\n",
        "                (name, local_size, expected_sizes[name])\n",
        "            )\n",
        "    else:\n",
        "        # No metadata available — accept if size is reasonable\n",
        "        # (shards are typically 130-175 MB)\n",
        "        if 100_000_000 < local_size < 200_000_000:\n",
        "            size_unknown.append(local_path)\n",
        "        else:\n",
        "            size_mismatched.append(\n",
        '                (name, local_size, "unknown (out of range)")\n',
        "            )\n",
        "\n",
        'print(f"  Size-matched:    {len(size_matched)}")\n',
        'print(f"  Size-mismatched: {len(size_mismatched)}")\n',
        "if size_unknown:\n",
        '    print(f"  No metadata:     {len(size_unknown)} (accepted by heuristic)")\n',
        "if size_mismatched:\n",
        '    print("  ⚠ MISMATCHED FILES (likely truncated):")\n',
        "    for name, got, want in size_mismatched:\n",
        '        print(f"    {name}: {got:,} bytes (expected {want:,})")\n',
        "\n",
        "\n",
        "# ── Layer 3: Tar Structural Integrity ──\n",
        "print()\n",
        'print("=" * 60)\n',
        'print("LAYER 3: Tar Structural Integrity")\n',
        'print("=" * 60)\n',
        "\n",
        "# Only check shards that passed the size check\n",
        "candidates = size_matched + size_unknown\n",
        "tar_ok = []\n",
        "tar_corrupted = []  # (path, error_message)\n",
        "\n",
        "for idx, shard_path in enumerate(candidates):\n",
        "    try:\n",
        '        with tarfile.open(shard_path, "r") as tf:\n',
        "            # Iterate all members to verify the archive index\n",
        "            _ = tf.getmembers()\n",
        "        tar_ok.append(shard_path)\n",
        "    except (tarfile.ReadError, EOFError, OSError) as exc:\n",
        "        tar_corrupted.append((shard_path, str(exc)))\n",
        "\n",
        "    # Progress reporting every 100 shards\n",
        "    if (idx + 1) % 100 == 0 or (idx + 1) == len(candidates):\n",
        "        print(\n",
        '            f"  Checked {idx + 1}/{len(candidates)} | "\n',
        '            f"OK={len(tar_ok)} Corrupt={len(tar_corrupted)}"\n',
        "        )\n",
        "\n",
        "if tar_corrupted:\n",
        '    print("  ⚠ CORRUPTED TAR FILES:")\n',
        "    for path, err in tar_corrupted:\n",
        '        print(f"    {Path(path).name}: {err}")\n',
        "\n",
        "\n",
        "# ── Layer 4: Sample JPEG Decode Spot-Check ──\n",
        "print()\n",
        'print("=" * 60)\n',
        'print("LAYER 4: JPEG Decode Spot-Check")\n',
        'print("=" * 60)\n',
        "\n",
        "# Test every SPOT_CHECK_EVERY-th healthy shard\n",
        "spot_check_shards = tar_ok[::SPOT_CHECK_EVERY]\n",
        "decode_ok = 0\n",
        "decode_fail = []  # (path, error_message)\n",
        "\n",
        "for shard_path in spot_check_shards:\n",
        "    try:\n",
        '        with tarfile.open(shard_path, "r") as tf:\n',
        "            # Find the first JPEG member\n",
        "            for member in tf.getmembers():\n",
        "                if member.name.lower().endswith(\n",
        '                    (".jpg", ".jpeg")\n',
        "                ):\n",
        "                    f = tf.extractfile(member)\n",
        "                    if f is not None:\n",
        "                        img_data = f.read()\n",
        "                        img = Image.open(io.BytesIO(img_data))\n",
        "                        img.verify()  # Validates JPEG structure\n",
        "                        decode_ok += 1\n",
        "                        break\n",
        "    except Exception as exc:\n",
        "        decode_fail.append((shard_path, str(exc)))\n",
        "\n",
        "print(\n",
        '    f"  Spot-checked: {len(spot_check_shards)} shards | "\n',
        '    f"OK={decode_ok} Failed={len(decode_fail)}"\n',
        ")\n",
        "if decode_fail:\n",
        '    print("  ⚠ DECODE FAILURES:")\n',
        "    for path, err in decode_fail:\n",
        '        print(f"    {Path(path).name}: {err}")\n',
        "\n",
        "\n",
        "# ── Build HEALTHY_SHARDS list ──\n",
        "# Only shards that passed all applicable layers enter the pipeline\n",
        "corrupted_set = {p for p, _ in tar_corrupted}\n",
        "decode_fail_set = {p for p, _ in decode_fail}\n",
        "HEALTHY_SHARDS = sorted(\n",
        "    p for p in tar_ok\n",
        "    if p not in corrupted_set and p not in decode_fail_set\n",
        ")\n",
        "\n",
        "\n",
        "# ── Summary Report ──\n",
        "print()\n",
        'print("=" * 60)\n',
        'print("SHARD INTEGRITY REPORT")\n',
        'print("=" * 60)\n',
        'print(f"  Total expected:     {EXPECTED_SHARD_COUNT}")\n',
        'print(f"  Present on disk:    {len(present_names)}")\n',
        'print(f"  Missing:            {len(missing_names)}")\n',
        'print(f"  Size-matched:       {len(size_matched)}")\n',
        'print(f"  Size-mismatched:    {len(size_mismatched)}")\n',
        'print(f"  Tar-corrupted:      {len(tar_corrupted)}")\n',
        "print(\n",
        '    f"  Decode spot-check:  "\n',
        '    f"{decode_ok}/{len(spot_check_shards)} OK"\n',
        ")\n",
        'print(f"  HEALTHY_SHARDS:     {len(HEALTHY_SHARDS)}")\n',
        "print()\n",
        "\n",
        "# Final status\n",
        "issues = len(size_mismatched) + len(tar_corrupted) + len(decode_fail)\n",
        "if issues == 0 and len(HEALTHY_SHARDS) == EXPECTED_SHARD_COUNT:\n",
        '    print("Status: ✅ All 1024 shards present and healthy")\n',
        "elif issues == 0:\n",
        "    print(\n",
        '        f"Status: ✅ All {len(HEALTHY_SHARDS)} present shards "\n',
        '        f"are healthy"\n',
        "    )\n",
        "    print(\n",
        '        f"        ⚠ {len(missing_names)} shards still "\n',
        '        f"downloading / missing"\n',
        "    )\n",
        "else:\n",
        "    print(\n",
        '        f"Status: ❌ {issues} issue(s) found — "\n',
        '        f"only {len(HEALTHY_SHARDS)} shards usable"\n',
        "    )\n",
        "    print(\n",
        '        "        Re-download mismatched/corrupted shards "\n',
        '        "before running the pipeline."\n',
        "    )\n",
        "\n",
        'print("=" * 60)\n',
    ],
}


# ── New markdown cell: Section 4c ──
MD_4C = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4c. DataLoader Smoke Test\n",
        "\n",
        "Quick sanity check that `make_dataloader()` works with the\n",
        "verified `HEALTHY_SHARDS` list — pull a single batch and\n",
        "confirm shapes and keys look correct.\n",
    ],
}


# ── New code cell: DataLoader smoke test ──
CODE_4C = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        '"""Smoke-test the DataLoader using only healthy shards."""\n',
        "\n",
        'print("Testing DataLoader (1 batch from HEALTHY_SHARDS)...")\n',
        "test_dl = make_dataloader(batch_size=4)\n",
        "try:\n",
        "    test_batch = next(iter(test_dl))\n",
        "    imgs, keys = test_batch\n",
        '    print(f"  Got batch: {imgs.shape[0]} images, shape={imgs.shape} (OK)")\n',
        '    print(f"  Sample keys: {keys[:2]}")\n',
        "except Exception as e:\n",
        '    print(f"  FAILED: {type(e).__name__}: {e}")\n',
        "    raise\n",
        "finally:\n",
        "    del test_dl\n",
        "\n",
        'print("\\nDataLoader smoke test: PASSED")\n',
    ],
}


def find_old_4b_cells(cells: list) -> tuple[int, int]:
    """Find the start and end indices of the old Section 4b cells.

    Looks for the markdown cell containing '## 4b. Dataset Verification'
    and the immediately following code cell.

    Returns:
        Tuple of (start_index, end_index) — inclusive range to replace.

    Raises:
        ValueError: If the old cells cannot be found.
    """
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            if "## 4b. Dataset Verification" in src:
                # The next cell should be the code cell
                if (
                    i + 1 < len(cells)
                    and cells[i + 1]["cell_type"] == "code"
                ):
                    return i, i + 1
                raise ValueError(
                    "Found 4b markdown but next cell is not code"
                )
    raise ValueError("Could not find Section 4b cells")


def main() -> None:
    """Insert shard verification cells into the notebook."""
    print(f"Reading {NB_PATH}")
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    start, end = find_old_4b_cells(cells)
    print(f"Found old Section 4b at cell indices [{start}, {end}]")

    # Replace old cells with new ones
    new_cells = [MD_4B, CODE_4B, MD_4C, CODE_4C]
    cells[start : end + 1] = new_cells
    print(f"Replaced 2 cells with {len(new_cells)} new cells")

    # ── Also update make_dataloader() to use HEALTHY_SHARDS ──
    # Find the cell containing make_dataloader and update _LOCAL_SHARDS
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def make_dataloader(" in src and "_LOCAL_SHARDS" in src:
            new_source = []
            for line in cell["source"]:
                if "_LOCAL_SHARDS," in line and "wds.WebDataset(" not in line:
                    # Replace _LOCAL_SHARDS with HEALTHY_SHARDS
                    new_source.append(
                        line.replace("_LOCAL_SHARDS,", "HEALTHY_SHARDS,")
                    )
                elif (
                    "f\"DataLoader ready: {len(_LOCAL_SHARDS)}" in line
                ):
                    new_source.append(
                        line.replace(
                            "_LOCAL_SHARDS",
                            "HEALTHY_SHARDS",
                        )
                    )
                else:
                    new_source.append(line)
            cell["source"] = new_source
            print("Updated make_dataloader() to use HEALTHY_SHARDS")
            break

    # Write back
    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote updated notebook to {NB_PATH}")


if __name__ == "__main__":
    main()
