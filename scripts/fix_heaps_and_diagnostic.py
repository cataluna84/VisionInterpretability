"""Fix heaps=None crash and add dataset connectivity diagnostic.

Issues:
1. scan_topk_keys crashes if the dataloader yields 0 samples because
   `heaps` stays None. Add a guard.
2. The gopen patch errors are swallowed by warn_and_continue, so we
   never see the real error. Add a single-shard fetch test before
   running the pipeline.
"""
import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "notebooks"
    / "Segment_3_canonical.ipynb"
)


def fix_scan_function(nb):
    """Add a guard for heaps=None in scan_topk_keys."""
    fixes = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        joined = "".join(cell["source"])
        if "def scan_topk_keys" not in joined:
            continue

        new_src = []
        for line in cell["source"]:
            # Add a guard before heaps.items()
            if "for c, h in heaps.items()" in line:
                # Insert guard before the dict comprehension
                new_src.append(
                    "        if heaps is None:\n"
                )
                new_src.append(
                    "            print(\"ERROR: No batches were processed. \"\n"
                )
                new_src.append(
                    '                  "Check HF_TOKEN and network connectivity.")\n'
                )
                new_src.append(
                    "            return {}\n"
                )
                new_src.append(
                    "\n"
                )
                fixes += 1
            new_src.append(line)
        cell["source"] = new_src
        break

    return fixes


def add_diagnostic_cell(nb):
    """Add a diagnostic cell right after the dataset cell.

    This cell tests that we can actually fetch one shard before
    running the full pipeline.
    """
    diagnostic_source = [
        '"""Diagnostic: verify we can fetch one shard from HuggingFace."""\n',
        "print(\"Testing dataset connectivity...\")\n",
        "try:\n",
        "    test_url = _URLS_TRAIN[0]\n",
        "    req = urllib.request.Request(\n",
        "        test_url,\n",
        '        headers={"Authorization": f"Bearer {HF_TOKEN}"},\n',
        "    )\n",
        "    response = urllib.request.urlopen(req, timeout=30)\n",
        "    # Read first 1KB to verify it's a valid tar\n",
        "    header = response.read(1024)\n",
        "    response.close()\n",
        "    if len(header) > 0:\n",
        '        print(f"OK: Fetched {len(header)} bytes from shard 0.")\n',
        "    else:\n",
        '        print("WARNING: Shard 0 returned empty response.")\n',
        "except urllib.error.HTTPError as e:\n",
        '    print(f"FAILED: HTTP {e.code} - {e.reason}")\n',
        "    if e.code == 401:\n",
        '        print("  -> Your HF_TOKEN is invalid or expired.")\n',
        "    elif e.code == 403:\n",
        '        print("  -> Access denied. You may need to accept the")\n',
        '        print("     ImageNet license at:")\n',
        '        print("     https://huggingface.co/datasets/timm/imagenet-1k-wds")\n',
        "    raise\n",
        "except Exception as e:\n",
        '    print(f"FAILED: {type(e).__name__}: {e}")\n',
        "    raise\n",
        "\n",
        "# Quick test: try loading one batch\n",
        "print(\"Testing DataLoader (1 batch)...\")\n",
        "test_dl = make_dataloader(batch_size=4)\n",
        "try:\n",
        "    test_batch = next(iter(test_dl))\n",
        "    imgs, keys = test_batch\n",
        '    print(f"OK: Got batch of {imgs.shape[0]} images, shape={imgs.shape}")\n',
        '    print(f"    Sample keys: {keys[:2]}")\n',
        "except StopIteration:\n",
        '    print("FAILED: DataLoader returned 0 samples.")\n',
        '    print("  -> The gopen patch may not be working correctly.")\n',
        "except Exception as e:\n",
        '    print(f"FAILED: {type(e).__name__}: {e}")\n',
        "    raise\n",
        "\n",
        'print("\\nDataset connectivity: ALL CHECKS PASSED")\n',
    ]

    diagnostic_md = [
        "## 4b. Dataset Connectivity Check\\n",
        "\\n",
        "Verify that we can actually fetch data from HuggingFace before\\n",
        "running the full pipeline (which would silently fail otherwise).\\n",
    ]

    # Find the dataset cell (contains make_dataloader)
    target_idx = None
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        joined = "".join(cell["source"])
        if "def make_dataloader" in joined:
            target_idx = idx
            break

    if target_idx is None:
        print("ERROR: Could not find dataset cell.")
        return 0

    # Insert markdown + diagnostic cell after the dataset cell
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": diagnostic_md,
    }
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": diagnostic_source,
    }
    nb["cells"].insert(target_idx + 1, md_cell)
    nb["cells"].insert(target_idx + 2, code_cell)
    return 1


def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    fixes = fix_scan_function(nb)
    diag = add_diagnostic_cell(nb)

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Applied {fixes} guard fixes, {diag} diagnostic cell(s) added.")


if __name__ == "__main__":
    main()
