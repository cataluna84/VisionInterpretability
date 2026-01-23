#!/usr/bin/env python3
"""Add plotly to the Colab setup cell in Segment 2 notebook."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_2_activation_max.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Find the setup cell and add plotly installation
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            
            # Look for the pip install line with torch-lucent
            if "pip install" in source and "torch-lucent" in source:
                # Add plotly to the install command
                new_source = []
                for line in cell["source"]:
                    if "pip install" in line and "torch-lucent" in line:
                        # Add plotly to the install
                        if "plotly" not in line:
                            line = line.rstrip().rstrip("\\n").rstrip() + " plotly\\n\",\n"
                    new_source.append(line)
                cell["source"] = new_source
                print("✅ Added plotly to existing pip install line")
                break
    else:
        # If no pip install found, add a setup cell at the beginning
        setup_cell = {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Install required dependencies (run once)\n",
                "# Uncomment the line below if running locally and plotly is not installed\n",
                "# !pip install plotly torch-lucent\n",
                "\n",
                "# For Colab, these should be pre-installed or installed in setup cell\n",
                "import sys\n",
                "try:\n",
                "    import plotly\n",
                "except ImportError:\n",
                "    !pip install -q plotly\n",
                "    import plotly\n",
                "\n",
                "print(f\"Plotly version: {plotly.__version__}\")\n"
            ]
        }
        
        # Insert after imports cell
        insert_at = 2  # After title and imports
        notebook["cells"].insert(insert_at, setup_cell)
        print(f"✅ Added plotly setup cell at position {insert_at}")
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print("   Run the setup cell to install plotly")


if __name__ == "__main__":
    main()
