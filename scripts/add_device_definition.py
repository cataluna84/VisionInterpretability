#!/usr/bin/env python3
"""Add DEVICE definition to imports cell in Segment 2 notebook."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_2_activation_max.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    cells = notebook["cells"]
    
    # Find the imports cell (has "import torch" and "from lucent")
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "import torch" in source and "from lucent.optvis import render" in source:
                # This is the imports cell - add DEVICE definition before the plt.rcParams
                # Insert after the warnings.filterwarnings line
                
                new_source_lines = []
                added = False
                
                for line in cell["source"]:
                    new_source_lines.append(line)
                    
                    # Add DEVICE definition after warnings.filterwarnings
                    if "warnings.filterwarnings('ignore'" in line and not added:
                        new_source_lines.append("\n")
                        new_source_lines.append("# Configure device (GPU if available, else CPU)\n")
                        new_source_lines.append("DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n")
                        new_source_lines.append(f"print(f'Using device: {{DEVICE}}')\n")
                        added = True
                
                cell["source"] = new_source_lines
                print(f"✅ Added DEVICE definition to cell {i} (imports cell)")
                break
    else:
        print("❌ Could not find imports cell")
        return
    
    # Save the notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print("✅ Notebook updated successfully")
    print("   DEVICE will be set to 'cuda' if GPU available, else 'cpu'")


if __name__ == "__main__":
    main()
