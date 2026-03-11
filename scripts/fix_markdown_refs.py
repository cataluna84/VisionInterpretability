"""Update the Markdown cell that still mentions pipe:curl."""
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
        if cell["cell_type"] != "markdown":
            continue
        new_src = []
        for line in cell["source"]:
            if "pipe:curl" in line and "retry" in line:
                line = "Streams directly from HuggingFace Hub using a Python-native URL opener\\n"
                fixes += 1
            elif "pipe:curl" in line and "Uses" in line:
                line = "with Bearer token auth -- **no curl or local download required**.\\n"
                fixes += 1
            new_src.append(line)
        cell["source"] = new_src

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Fixed {fixes} markdown references.")


if __name__ == "__main__":
    main()
