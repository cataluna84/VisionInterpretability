"""
Script to add Master Experiment Log Table to the segment 3 notebook.

Changes:
1. Add Run ID generation to CONFIG cell
2. Update save cell to use run-specific subfolder
3. Add Section 10 cells (Configuration + Master Experiment Log) after cleanup
"""

import json
import sys
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_3_dataset_images_imagenet_validation.ipynb"


def load_notebook(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(path: Path, nb: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Saved: {path}")


def find_cell_index(cells: list, cell_id: str) -> int:
    """Find cell index by its 'id' field."""
    for i, cell in enumerate(cells):
        if cell.get("id") == cell_id:
            return i
    raise ValueError(f"Cell with id '{cell_id}' not found")


def modify_config_cell(cell: dict) -> None:
    """Change 1: Add Run ID generation to the CONFIG cell."""
    old_tail = [
        '"}\n"',
        '"\n"',
        '"print(\\"Configuration:\\")\n"',
        '"for k, v in CONFIG.items():\n"',
        '"    print(f\\"  {k}: {v}\\")\n"',
    ]
    new_tail = [
        '"}\n"',
        '"\n"',
        '"# Generate unique Run ID (primary key for this experiment)\n"',
        '"from datetime import datetime\n"',
        '"run_timestamp = datetime.now().strftime(\\"%Y%m%dT%H%M%S\\")\n"',
        '"run_id = (\n"',
        '"    f\\"{CONFIG[\'layer_name\']}_{CONFIG[\'dataset_split\']}\\"\\n"\n"',
        '    "    f\\"_n{CONFIG[\'num_neurons\']}_s{CONFIG[\'optim_steps\']}\\"\\n"\n"',
        '    "    f\\"_r{CONFIG[\'optim_resolution\']}_{CONFIG[\'optimization_mode\']}\\"\\n"\n"',
        '    "    f\\"_{run_timestamp}\\"\\n"\n"',
        '")\n"',
        '"\n"',
        '"print(\\"Configuration:\\")\n"',
        '"for k, v in CONFIG.items():\n"',
        '"    print(f\\"  {k}: {v}\\")\n"',
        '"print(f\\"\\\\nRun ID: {run_id}\\")\n"',
    ]
    # Instead of complex string matching, replace the source entirely
    source = cell["source"]

    # Find the line with "}\n" (end of CONFIG dict) - search from end
    dict_end_idx = None
    for i in range(len(source) - 1, -1, -1):
        if source[i].strip() == '"}\n",':
            # Found via raw comparison; use a different approach
            pass
        if source[i] == '}\n':
            dict_end_idx = i
            break

    if dict_end_idx is None:
        raise ValueError("Could not find end of CONFIG dict in config cell")

    # Build new source: keep everything up to and including "}\n",
    # then add the Run ID generation + config printing
    new_source = source[:dict_end_idx + 1] + [
        '\n',
        '# Generate unique Run ID (primary key for this experiment)\n',
        'from datetime import datetime\n',
        'run_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")\n',
        'run_id = (\n',
        '    f"{CONFIG[\'layer_name\']}_{CONFIG[\'dataset_split\']}"\n',
        '    f"_n{CONFIG[\'num_neurons\']}_s{CONFIG[\'optim_steps\']}"\n',
        '    f"_r{CONFIG[\'optim_resolution\']}_{CONFIG[\'optimization_mode\']}"\n',
        '    f"_{run_timestamp}"\n',
        ')\n',
        '\n',
        'print("Configuration:")\n',
        'for k, v in CONFIG.items():\n',
        '    print(f"  {k}: {v}")\n',
        'print(f"\\nRun ID: {run_id}")\n',
    ]

    cell["source"] = new_source
    # Clear cached outputs
    cell["outputs"] = []
    cell["execution_count"] = None
    print("  [OK] Change 1: Run ID generation added to CONFIG cell")


def modify_save_cell(cell: dict) -> None:
    """Change 2: Update save cell to use run-specific subfolder."""
    for i, line in enumerate(cell["source"]):
        if "segment_3_dataset_images" in line and "results_dir" in line and "run_id" not in line:
            cell["source"][i] = 'results_dir = project_root / "notebooks" / "results" / "segment_3_dataset_images" / run_id\n'
            break
    else:
        raise ValueError("Could not find results_dir line in save cell")

    # Also update the comment
    for i, line in enumerate(cell["source"]):
        if "# Create results directory" in line:
            cell["source"][i] = "# Create run-specific results directory (keyed by run_id)\n"
            break

    cell["outputs"] = []
    cell["execution_count"] = None
    print("  [OK] Change 2: Save cell updated to use run-specific subfolder")


def create_section10_cells() -> list:
    """Change 3: Create 3 new cells for Section 10."""

    markdown_cell = {
        "cell_type": "markdown",
        "id": "experiment-log-markdown",
        "metadata": {},
        "source": [
            "## 10. Configuration & Master Experiment Log\n",
            "\n",
            "This section records the **complete configuration** and a **Master Experiment Log Table** for this run.\n",
            "Each saved image is assigned a unique `image_id` derived from the `run_id` primary key, making it\n",
            "possible to trace any output PNG back to the exact hyperparameters that produced it.\n",
        ],
    }

    config_summary_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "config-summary",
        "metadata": {},
        "outputs": [],
        "source": [
            '# --- Configuration Parameters ---\n',
            'print("=" * 60)\n',
            'print("CONFIGURATION PARAMETERS")\n',
            'print("=" * 60)\n',
            'print(f"Run ID (Primary Key): {run_id}")\n',
            'print("-" * 60)\n',
            'for k, v in CONFIG.items():\n',
            '    print(f"  {k:30s}: {v}")\n',
            'print("=" * 60)\n',
        ],
    }

    log_table_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "master-experiment-log",
        "metadata": {},
        "outputs": [],
        "source": [
            '# --- Master Experiment Log Table ---\n',
            'import pandas as pd\n',
            '\n',
            'rows = []\n',
            'for n in range(CONFIG["num_neurons"]):\n',
            '    image_filename = f"neuron_{n}_spectrum.png"\n',
            '    image_path = f"results/segment_3_dataset_images/{run_id}/{image_filename}"\n',
            '    rows.append({\n',
            '        "run_id": run_id,\n',
            '        "image_id": f"{run_id}_neuron_{n}",\n',
            '        "neuron_index": n,\n',
            '        "image_path": image_path,\n',
            '        "layer_name": CONFIG["layer_name"],\n',
            '        "dataset_split": CONFIG["dataset_split"],\n',
            '        "num_neurons": CONFIG["num_neurons"],\n',
            '        "samples_per_category": CONFIG["samples_per_category"],\n',
            '        "batch_size": CONFIG["batch_size"],\n',
            '        "optim_resolution": CONFIG["optim_resolution"],\n',
            '        "optim_steps": CONFIG["optim_steps"],\n',
            '        "optimization_mode": CONFIG["optimization_mode"],\n',
            '        "total_samples": total_processed,\n',
            '        "total_time_sec": round(total_time, 2),\n',
            '        "avg_throughput": round(throughput_avg, 2),\n',
            '        "gpu": gpu_name,\n',
            '        "gpu_memory_peak_gb": round(gpu_memory_peak, 2),\n',
            '        "wandb_run_url": run.url if hasattr(run, \'url\') else "N/A",\n',
            '        "timestamp": run_timestamp,\n',
            '    })\n',
            '\n',
            'experiment_log = pd.DataFrame(rows)\n',
            '\n',
            'print("=" * 80)\n',
            'print("MASTER EXPERIMENT LOG")\n',
            'print("=" * 80)\n',
            'display(experiment_log)\n',
            'print(f"\\n{len(rows)} images logged for run: {run_id}")\n',
        ],
    }

    return [markdown_cell, config_summary_cell, log_table_cell]


def main():
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    nb = load_notebook(NOTEBOOK_PATH)
    cells = nb["cells"]

    # Change 1: Modify CONFIG cell
    config_idx = find_cell_index(cells, "config")
    modify_config_cell(cells[config_idx])

    # Change 2: Modify save-figures cell
    save_idx = find_cell_index(cells, "save-figures")
    modify_save_cell(cells[save_idx])

    # Change 3: Add Section 10 cells after the last cell (cleanup)
    new_cells = create_section10_cells()
    cells.extend(new_cells)
    print(f"  [OK] Change 3: Added {len(new_cells)} new cells (Section 10)")

    # Save
    save_notebook(NOTEBOOK_PATH, nb)
    print("\nAll changes applied successfully!")


if __name__ == "__main__":
    main()
