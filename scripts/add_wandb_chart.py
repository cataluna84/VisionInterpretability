#!/usr/bin/env python3
"""Add WandB Table logging for images to Segment 3 notebook."""
import json
from pathlib import Path

# New source code for the "save-figures" cell
NEW_SAVE_SOURCE = [
    "import os\n",
    "\n",
    "# Create results directory\n",
    "results_dir = project_root / \"notebooks\" / \"results\" / \"segment_3_dataset_images\"\n",
    "results_dir.mkdir(parents=True, exist_ok=True)\n",
    "print(f\"Saving results to: {results_dir}\")\n",
    "\n",
    "# Initialize WandB Table for logging chart\n",
    "wandb_table = wandb.Table(columns=[\"Neuron Index\", \"Activation Spectrum\"])\n",
    "\n",
    "if 'figures' in locals() and figures:\n",
    "    for n, fig in enumerate(figures):\n",
    "        # Save locally\n",
    "        save_path = results_dir / f\"neuron_{n}_spectrum.png\"\n",
    "        fig.savefig(save_path, dpi=150, bbox_inches='tight')\n",
    "        print(f\"  Saved: {save_path.name}\")\n",
    "        \n",
    "        # Add to WandB Table\n",
    "        # converting figure to image for logging\n",
    "        try:\n",
    "            wandb_table.add_data(n, wandb.Image(fig))\n",
    "        except Exception as e:\n",
    "            print(f\"  Warning: Could not add figure {n} to WandB table: {e}\")\n",
    "else:\n",
    "    print(\"No figures found to save.\")\n",
    "\n",
    "# Log the table to WandB\n",
    "if wandb.run is not None:\n",
    "    wandb.log({\"activation_spectrum_chart\": wandb_table})\n",
    "    print(\"Logged activation spectrum table to WandB.\")"
]

def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_3_dataset_images.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    updated = False
    for cell in notebook["cells"]:
        if cell.get("id") == "save-figures":
            cell["source"] = NEW_SAVE_SOURCE
            updated = True
            print("✅ Updated 'save-figures' cell with WandB table logging.")
            break
            
    if not updated:
        # Fallback: look for cell containing specific text if ID doesn't match
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source_str = "".join(cell.get("source", []))
                if "results_dir =" in source_str and "project_root" in source_str:
                    cell["source"] = NEW_SAVE_SOURCE
                    updated = True
                    print("✅ Updated saving cell (found by content) with WandB table logging.")
                    break

    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=4)
        print(f"Successfully modified {notebook_path.name}")
    else:
        print("❌ Could not find the 'save-figures' cell to update.")

if __name__ == "__main__":
    main()
