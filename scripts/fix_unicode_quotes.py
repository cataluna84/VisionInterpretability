"""Fix the Unicode escape in the print line of the dataset cell."""
import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "notebooks"
    / "Segment_3_canonical.ipynb"
)


def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    fixes = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_src = []
        for line in cell["source"]:
            if "\\u0027" in line:
                line = line.replace("\\u0027", "'")
                fixes += 1
            new_src.append(line)
        cell["source"] = new_src

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Fixed {fixes} Unicode escapes.")


if __name__ == "__main__":
    main()
