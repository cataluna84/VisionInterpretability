"""Fix the misplaced heaps guard — move it before the dict comprehension."""
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

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        joined = "".join(cell["source"])
        if "def scan_topk_keys" not in joined:
            continue

        # Remove the incorrectly placed guard lines
        new_src = []
        skip_guard = False
        for line in cell["source"]:
            # Skip the 4 bad guard lines that were inserted
            if "        if heaps is None:\n" == line and skip_guard is False:
                skip_guard = True
                continue
            if skip_guard and ('            print("ERROR: No batches' in line
                              or '"Check HF_TOKEN and network' in line
                              or "            return {}\n" == line):
                continue
            if skip_guard and line == "\n":
                skip_guard = False
                continue
            skip_guard = False
            new_src.append(line)
        cell["source"] = new_src

        # Now insert the guard BEFORE `topk_by_channel = {`
        final_src = []
        for line in cell["source"]:
            if "topk_by_channel = {" in line:
                final_src.append(
                    "        if heaps is None:\n"
                )
                final_src.append(
                    '            print("ERROR: No batches processed. '\
                    'Check HF_TOKEN and network.")\n'
                )
                final_src.append(
                    "            return {}\n"
                )
                final_src.append("\n")
            final_src.append(line)
        cell["source"] = final_src
        cell["outputs"] = []
        cell["execution_count"] = None
        break

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print("Fixed: guard now placed before dict comprehension.")


if __name__ == "__main__":
    main()
