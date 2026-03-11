"""Fix runtime errors in Segment_3_canonical.ipynb.

Fixes:
1. `total_mem` -> `total_memory` (correct PyTorch attribute name)
2. Replace lambda with named function (Windows can't pickle lambdas)
3. `num_workers=2` -> `num_workers=0` (pipe:curl + Windows multiprocessing incompatible)
"""
import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "Segment_3_canonical.ipynb"


def main():
    """Apply all fixes to the notebook."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    fixes = 0

    # ── Pass 1: line-level replacements across all code cells ──
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_src = []
        for line in cell["source"]:
            # Fix 1: total_mem -> total_memory
            if ".total_mem " in line and "get_device_properties" in line:
                line = line.replace(".total_mem ", ".total_memory ")
                fixes += 1

            # Fix 2: replace lambda with named function reference
            if "lambda x: x is not None" in line:
                line = line.replace(
                    "lambda x: x is not None", "_is_not_none"
                )
                fixes += 1

            # Fix 3: num_workers=2 -> num_workers=0
            if "num_workers=2" in line:
                line = line.replace("num_workers=2", "num_workers=0")
                fixes += 1

            new_src.append(line)
        cell["source"] = new_src

    # ── Pass 2: inject _is_not_none function before to_tensor ──
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        joined = "".join(cell["source"])
        if "def to_tensor" in joined:
            new_src = []
            for line in cell["source"]:
                if line.startswith("def to_tensor"):
                    # Insert named filter function before to_tensor
                    new_src.extend([
                        "def _is_not_none(x: tuple | None) -> bool:\n",
                        '    """Filter predicate: returns True if sample is not None."""\n',
                        "    return x is not None\n",
                        "\n",
                        "\n",
                    ])
                    fixes += 1
                new_src.append(line)
            cell["source"] = new_src
            break

    # ── Pass 3: clear outputs and execution counts ──
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # ── Write back ──
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Applied {fixes} fixes to {NOTEBOOK_PATH.name}")


if __name__ == "__main__":
    main()
