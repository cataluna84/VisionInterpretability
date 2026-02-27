"""Add Master Experiment Log with folder-based run_id to the imagenet_validation notebook."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_3_dataset_images_imagenet_validation.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ─── 1. CONFIG cell: append run_id generation ───────────────────────────
for cell in nb["cells"]:
    if cell.get("id") == "config":
        # Check if run_id logic already exists
        src_text = "".join(cell["source"])
        if "run_id" in src_text:
            print("  [SKIP] CONFIG cell already has run_id logic")
            break

        # Remove trailing quote-only line if present, we'll re-add it
        # The last line is: '}'  (the CONFIG dict closing brace)
        # We need to add a newline after it, then the run_id block
        last_line = cell["source"][-1]
        if last_line.strip() == "}":
            cell["source"][-1] = "}\n"

        cell["source"].extend([
            "\n",
            "# --- Auto-increment Run ID (folder-based) ---\n",
            "_results_base = project_root / \"notebooks\" / \"results\" / \"segment_3_dataset_images\"\n",
            "_results_base.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "_existing_ids = [\n",
            "    int(f.name) for f in _results_base.iterdir()\n",
            "    if f.is_dir() and f.name.isdigit() and len(f.name) == 3\n",
            "]\n",
            "run_id = f\"{max(_existing_ids) + 1:03d}\" if _existing_ids else \"001\"\n",
            "\n",
            "print(f\"\\nRun ID: {run_id}\")\n",
            "print(f\"Results will save to: results/segment_3_dataset_images/{run_id}/\")",
        ])
        print("  [OK] CONFIG cell: added run_id generation")
        break

# ─── 2. Update Section 10 markdown cell ─────────────────────────────────
for i, cell in enumerate(nb["cells"]):
    if cell.get("id") == "experiment-log-markdown":
        cell["source"] = [
            "## 10. Master Experiment Log\n",
            "\n",
            "After each run, manually add a row below with the printed `Run ID` and config values.\n",
            "Results are saved to `results/segment_3_dataset_images/<run_id>/`.\n",
        ]
        print("  [OK] Updated Section 10 markdown header")
        break

# ─── 3. Add experiment-log-table markdown cell at the end ────────────────
# Check if it already exists
has_table = any(c.get("id") == "experiment-log-table" for c in nb["cells"])
if has_table:
    print("  [SKIP] experiment-log-table cell already exists")
else:
    table_cell = {
        "cell_type": "markdown",
        "id": "experiment-log-table",
        "metadata": {},
        "source": [
            "| Run | Layer | Neurons | Split | Samples | Batch | Steps | Res | Mode | Time | Throughput | GPU Mem (GB) | W&B |\n",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
        ],
    }
    nb["cells"].append(table_cell)
    print("  [OK] Added experiment-log-table markdown cell")

# ─── Write back ──────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"\n✅ Notebook updated ({len(nb['cells'])} cells)")

# Validate JSON
with open(NB_PATH, "r", encoding="utf-8") as f:
    json.load(f)
print("✅ JSON validation passed")
