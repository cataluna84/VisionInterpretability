"""
Add the missing append-experiment-log code cell between the
intro markdown and the log table cells.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "notebooks"
    / "cataluna84__segment_3_dataset_images_imagenet_validation.ipynb"
)


def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Find the log table cell index
    table_idx = None
    for i, c in enumerate(cells):
        if c.get("id") == "experiment-log-table":
            table_idx = i
            break

    if table_idx is None:
        print("[ERR] experiment-log-table cell not found")
        return

    # Check if append cell already exists
    for c in cells:
        if c.get("id") == "append-experiment-log":
            print("[SKIP] append-experiment-log cell already exists")
            return

    # Create the append code cell
    append_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "append-experiment-log",
        "metadata": {},
        "outputs": [],
        "source": [
            '# --- Append this run to the Master Experiment Log ---\n',
            'import json\n',
            '\n',
            '_nb_path = project_root / "notebooks" / "cataluna84__segment_3_dataset_images_imagenet_validation.ipynb"\n',
            '\n',
            '# Build the new row\n',
            "_wandb_url = run.url if hasattr(run, 'url') else 'N/A'\n",
            '_wandb_link = f"[W&B]({_wandb_url})" if _wandb_url != "N/A" else "N/A"\n',
            '_gpu_mem = f"{torch.cuda.max_memory_allocated() / (1024**3):.2f}" if torch.cuda.is_available() else "0"\n',
            '_image_dir = f"results/segment_3_dataset_images/{run_id}/"\n',
            '\n',
            '_row = (\n',
            '    f"| {run_id} "\n',
            '    f"| {total_processed:,} "\n',
            '    f"| {total_time:.1f} "\n',
            '    f"| {throughput_avg:.1f} "\n',
            '    f"| {_gpu_mem} "\n',
            '    f"| {_image_dir} "\n',
            '    f"| {_wandb_link} |\\n"\n',
            ')\n',
            '\n',
            '# Read notebook, append row, write back\n',
            "with open(_nb_path, 'r', encoding='utf-8') as f:\n",
            '    _nb = json.load(f)\n',
            '\n',
            '_found = False\n',
            "for _cell in _nb['cells']:\n",
            "    if _cell.get('id') == 'experiment-log-table':\n",
            "        _cell['source'].append(_row)\n",
            '        _found = True\n',
            '        break\n',
            '\n',
            'if _found:\n',
            "    with open(_nb_path, 'w', encoding='utf-8') as f:\n",
            '        json.dump(_nb, f, indent=1, ensure_ascii=False)\n',
            '    print(f"✅ Logged run {run_id} to Master Experiment Log")\n',
            '    print(f"   Images: {_image_dir}")\n',
            '    print(f"   W&B: {_wandb_url}")\n',
            'else:\n',
            '    print("⚠️ Could not find experiment-log-table cell.")\n',
        ],
    }

    # Insert before the table cell
    cells.insert(table_idx, append_cell)

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[OK] Inserted append-experiment-log cell at index {table_idx}")
    print(f"Saved: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
