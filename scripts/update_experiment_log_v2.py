"""
Comprehensive update: sequential Run ID, slim 7-col table, W&B hyperlink,
rename old image folder to 001/.
"""

import json
import shutil
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parent.parent
    / "notebooks"
    / "cataluna84__segment_3_dataset_images_imagenet_validation.ipynb"
)
RESULTS_DIR = (
    Path(__file__).parent.parent
    / "notebooks"
    / "results"
    / "segment_3_dataset_images"
)


def load_notebook(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(path: Path, nb: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {path}")


def find_cell(cells: list, cell_id: str) -> dict | None:
    for c in cells:
        if c.get("id") == cell_id:
            return c
    return None


def update_config_cell(cell: dict) -> None:
    """Replace composite run_id + timestamp with sequential run_id."""
    source = cell["source"]

    # Find end of CONFIG dict
    dict_end_idx = None
    for i in range(len(source) - 1, -1, -1):
        if source[i] == '}\n':
            dict_end_idx = i
            break

    if dict_end_idx is None:
        raise ValueError("Could not find end of CONFIG dict")

    cell["source"] = source[:dict_end_idx + 1] + [
        '\n',
        '# Generate sequential Run ID by counting existing log rows\n',
        'import json as _json\n',
        '_nb_path = project_root / "notebooks" / "cataluna84__segment_3_dataset_images_imagenet_validation.ipynb"\n',
        'with open(_nb_path, "r", encoding="utf-8") as _f:\n',
        '    _nb_data = _json.load(_f)\n',
        '_existing_rows = 0\n',
        'for _c in _nb_data["cells"]:\n',
        '    if _c.get("id") == "experiment-log-table":\n',
        '        _existing_rows = sum(\n',
        '            1 for line in _c["source"]\n',
        '            if line.startswith("| ") and not line.startswith("| ---") and not line.startswith("| Run")\n',
        '        )\n',
        '        break\n',
        'run_id = f"{_existing_rows + 1:03d}"\n',
        '\n',
        'print("Configuration:")\n',
        'for k, v in CONFIG.items():\n',
        '    print(f"  {k}: {v}")\n',
        'print(f"\\nRun ID: {run_id}")\n',
    ]
    cell["outputs"] = []
    cell["execution_count"] = None
    print("  [OK] CONFIG cell: sequential run_id, no timestamp")


def update_append_log_cell(cell: dict) -> None:
    """Uncomment and update: 7 columns, W&B as hyperlink, no timestamp."""
    cell["source"] = [
        '# --- Append this run to the Master Experiment Log ---\n',
        'import json\n',
        '\n',
        '_nb_path = project_root / "notebooks" / "cataluna84__segment_3_dataset_images_imagenet_validation.ipynb"\n',
        '\n',
        '# Build the new row\n',
        '_wandb_url = run.url if hasattr(run, \'url\') else \'N/A\'\n',
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
        'with open(_nb_path, \'r\', encoding=\'utf-8\') as f:\n',
        '    _nb = json.load(f)\n',
        '\n',
        '_found = False\n',
        'for _cell in _nb[\'cells\']:\n',
        '    if _cell.get(\'id\') == \'experiment-log-table\':\n',
        '        _cell[\'source\'].append(_row)\n',
        '        _found = True\n',
        '        break\n',
        '\n',
        'if _found:\n',
        '    with open(_nb_path, \'w\', encoding=\'utf-8\') as f:\n',
        '        json.dump(_nb, f, indent=1, ensure_ascii=False)\n',
        '    print(f"✅ Logged run {run_id} to Master Experiment Log")\n',
        '    print(f"   Images: {_image_dir}")\n',
        '    print(f"   W&B: {_wandb_url}")\n',
        'else:\n',
        '    print("⚠️ Could not find experiment-log-table cell.")\n',
    ]
    cell["outputs"] = []
    cell["execution_count"] = None
    print("  [OK] Append-log cell: 7 columns, W&B hyperlink, no timestamp")


def update_log_table_cell(cell: dict, wandb_url: str) -> None:
    """Update headers to 7 columns and convert existing row to 001."""
    wandb_link = f"[W&B]({wandb_url})" if wandb_url else "N/A"

    cell["source"] = [
        "### Master Experiment Log Table\n",
        "\n",
        "| Run ID | Samples | Time (s) | Throughput | GPU Mem (GB) | Image Dir | W&B |\n",
        "| --- | --- | --- | --- | --- | --- | --- |\n",
        f"| 001 | 50,000 | 977.0 | 51.2 | 5.12 | results/segment_3_dataset_images/001/ | {wandb_link} |\n",
    ]
    print("  [OK] Log table: 7-col headers + existing row converted to 001")


def rename_old_folder() -> None:
    """Rename the old long-named folder to 001/."""
    old_name = "mixed4a_validation_n10_s512_r512_baseline_20260210T112429"
    old_path = RESULTS_DIR / old_name
    new_path = RESULTS_DIR / "001"

    if old_path.exists():
        if new_path.exists():
            print(f"  [SKIP] Target folder {new_path} already exists")
        else:
            shutil.move(str(old_path), str(new_path))
            print(f"  [OK] Renamed: {old_name}/ → 001/")
    else:
        print(f"  [SKIP] Old folder not found: {old_path}")


def main():
    print("Applying: sequential Run ID, slim table, W&B hyperlink...\n")

    nb = load_notebook(NOTEBOOK_PATH)
    cells = nb["cells"]

    # 1. CONFIG cell
    config_cell = find_cell(cells, "config")
    if config_cell:
        update_config_cell(config_cell)
    else:
        print("  [ERR] CONFIG cell not found")

    # 2. Append-log cell
    append_cell = find_cell(cells, "append-experiment-log")
    if append_cell:
        update_append_log_cell(append_cell)
    else:
        print("  [ERR] append-experiment-log cell not found")

    # 3. Log table cell - extract existing W&B URL first
    table_cell = find_cell(cells, "experiment-log-table")
    wandb_url = ""
    if table_cell:
        # Extract W&B URL from existing row
        for line in table_cell["source"]:
            if "wandb.ai" in line:
                # Find the URL
                import re
                urls = re.findall(r'https://wandb\.ai/[^\s|]+', line)
                if urls:
                    wandb_url = urls[0].rstrip("/").rstrip("|").strip()
        update_log_table_cell(table_cell, wandb_url)
    else:
        print("  [ERR] experiment-log-table cell not found")

    # Remove the empty trailing markdown cell if present
    while cells and cells[-1].get("cell_type") == "markdown" and not any(cells[-1].get("source", [])):
        cells.pop()
        print("  [OK] Removed empty trailing cell")

    save_notebook(NOTEBOOK_PATH, nb)

    # 4. Rename old image folder
    rename_old_folder()

    print("\nDone! All changes applied.")


if __name__ == "__main__":
    main()
