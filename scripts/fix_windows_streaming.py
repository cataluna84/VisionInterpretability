"""Fix the gopen monkey-patch to use the correct import path.

In webdataset 1.0.2:
- `wds.gopen` is a function (re-exported at package level)
- `sys.modules['webdataset.gopen']` is the actual module
- `import webdataset.gopen as wds_gopen` can fail because Python's
  attribute resolution prefers the package-level function

Fix: use `sys.modules` to access the gopen module directly.
"""
import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "notebooks"
    / "Segment_3_canonical.ipynb"
)

# The corrected dataset cell source
NEW_DATASET_CELL = [
    '"""ImageNet-1k streaming dataset via WebDataset."""\n',
    "import sys\n",
    "import urllib.request\n",
    "\n",
    "import webdataset as wds\n",
    "from torch.utils.data import DataLoader\n",
    "from torchvision import transforms\n",
    "\n",
    "# Preprocessing (ImageNet normalization)\n",
    "preprocess = transforms.Compose([\n",
    "    transforms.Resize(256),\n",
    "    transforms.CenterCrop(224),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize(\n",
    "        mean=[0.485, 0.456, 0.406],\n",
    "        std=[0.229, 0.224, 0.225],\n",
    "    ),\n",
    "])\n",
    "\n",
    "\n",
    "def _is_not_none(x: tuple | None) -> bool:\n",
    '    """Filter predicate: returns True if sample is not None."""\n',
    "    return x is not None\n",
    "\n",
    "\n",
    "def to_tensor(sample: dict) -> tuple | None:\n",
    '    """Convert a WDS sample to (tensor, key), or None if no image.\n',
    "\n",
    "    Tries multiple image extensions for compatibility with different\n",
    "    WebDataset shards.\n",
    "\n",
    "    Args:\n",
    "        sample: WebDataset sample dictionary.\n",
    "\n",
    "    Returns:\n",
    "        Tuple of (preprocessed_tensor, sample_key), or None.\n",
    '    """\n',
    "    img = None\n",
    '    for ext in ("jpg", "jpeg", "JPEG", "png"):\n',
    "        img = sample.get(ext)\n",
    "        if img is not None:\n",
    "            break\n",
    "    if img is None:\n",
    "        return None\n",
    '    key = sample.get("__key__", "")\n',
    "    return preprocess(img), key\n",
    "\n",
    "\n",
    "# ── Monkey-patch gopen for authenticated HuggingFace URLs ──\n",
    "# In webdataset 1.0, the gopen module is accessed via sys.modules\n",
    "# because `wds.gopen` resolves to the function, not the module.\n",
    '_gopen_module = sys.modules["webdataset.gopen"]\n',
    "_original_gopen = _gopen_module.gopen\n",
    "\n",
    "\n",
    'def _gopen_with_hf_auth(url, mode="rb", bufsize=8192, **kw):\n',
    '    """Open URLs with HuggingFace Bearer auth (no curl needed).\n',
    "\n",
    "    For huggingface.co URLs, uses urllib with an Authorization header.\n",
    "    For all other URLs, falls back to the original gopen.\n",
    '    """\n',
    '    if isinstance(url, str) and "huggingface.co" in url:\n',
    "        req = urllib.request.Request(\n",
    "            url,\n",
    '            headers={"Authorization": f"Bearer {HF_TOKEN}"},\n',
    "        )\n",
    "        return urllib.request.urlopen(req, timeout=120)\n",
    "    return _original_gopen(url, mode, bufsize, **kw)\n",
    "\n",
    "\n",
    "_gopen_module.gopen = _gopen_with_hf_auth\n",
    'print("Patched webdataset.gopen with HF auth opener.")\n',
    "\n",
    "\n",
    "# Build shard URLs (direct HTTPS, no pipe:curl)\n",
    '_BASE = "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/"\n',
    "_URLS_TRAIN = [\n",
    '    f"{_BASE}imagenet1k-train-{i:04d}.tar"\n',
    "    for i in range(1024)\n",
    "]\n",
    "\n",
    "\n",
    'def make_dataloader(batch_size: int = CFG["BATCH_SIZE"]) -> DataLoader:\n',
    '    """Create a fresh DataLoader (WDS iterators are single-use).\n',
    "\n",
    "    Args:\n",
    "        batch_size: Number of images per batch.\n",
    "\n",
    "    Returns:\n",
    "        A new DataLoader ready for iteration.\n",
    '    """\n',
    "    dataset = (\n",
    "        wds.WebDataset(\n",
    "            _URLS_TRAIN,\n",
    "            handler=wds.handlers.warn_and_continue,\n",
    "            empty_check=False,\n",
    "        )\n",
    '        .decode("pil", handler=wds.handlers.warn_and_continue)\n',
    "        .map(to_tensor, handler=wds.handlers.warn_and_continue)\n",
    "        .select(_is_not_none)\n",
    "    )\n",
    "    return DataLoader(\n",
    "        dataset,\n",
    "        batch_size=batch_size,\n",
    "        num_workers=0,\n",
    '        pin_memory=(device.type == "cuda"),\n',
    "    )\n",
    "\n",
    "\n",
    "print(f\"Dataset ready: {len(_URLS_TRAIN)} shards, batch_size={CFG['BATCH_SIZE']}\")\n",
]


def main():
    """Replace the dataset cell with the corrected gopen patch."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the dataset cell
    target_idx = None
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        joined = "".join(cell["source"])
        if "def make_dataloader" in joined:
            target_idx = idx
            break

    if target_idx is None:
        print("ERROR: Could not find the dataset cell.")
        return

    nb["cells"][target_idx]["source"] = NEW_DATASET_CELL
    nb["cells"][target_idx]["outputs"] = []
    nb["cells"][target_idx]["execution_count"] = None

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Fixed dataset cell (index {target_idx}).")
    print("Key change: sys.modules['webdataset.gopen'] instead of wds_gopen")


if __name__ == "__main__":
    main()
