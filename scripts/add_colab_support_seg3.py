#!/usr/bin/env python3
"""Add comprehensive Colab/Local setup cell to Segment 3 notebook."""
import json
from pathlib import Path


SETUP_CELL = {
    "cell_type": "code",
    "metadata": {
      "id": "colab-setup"
    },
    "execution_count": None,
    "outputs": [],
    "source": [
        "# @title 🚀 Environment Setup (Run this cell first!)\n",
        "# @markdown This cell sets up the environment for both Colab and local runs.\n",
        "\n",
        "import sys\n",
        "import os\n",
        "\n",
        "# Check if running in Google Colab\n",
        "IN_COLAB = 'google.colab' in sys.modules\n",
        "\n",
        "if IN_COLAB:\n",
        "    print(\"🌐 Running in Google Colab\")\n",
        "    \n",
        "    # Clone the repository if not already cloned\n",
        "    if not os.path.exists('VisionInterpretability'):\n",
        "        !git clone https://github.com/cataluna84/VisionInterpretability.git\n",
        "    \n",
        "    # Change to project directory\n",
        "    os.chdir('VisionInterpretability')\n",
        "    \n",
        "    # Install dependencies\n",
        "    !pip install -q torch torchvision matplotlib numpy pillow tqdm opencv-python requests\n",
        "    \n",
        "    # Add src to Python path\n",
        "    sys.path.insert(0, 'src')\n",
        "    print(\"✅ Colab setup complete!\")\n",
        "else:\n",
        "    print(\"💻 Running locally\")\n",
        "    # For local runs, add src to path if running from notebooks directory\n",
        "    if os.path.basename(os.getcwd()) == 'notebooks':\n",
        "        sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), 'src'))\n",
        "    elif 'src' not in sys.path:\n",
        "        sys.path.insert(0, 'src')\n",
        "    print(\"✅ Local setup complete!\")"
    ]
}

COLAB_BADGE = {
    "cell_type": "markdown",
    "metadata": {
      "id": "colab-badge"
    },
    "source": [
        "<a href=\"https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_3_dataset_images.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
    ]
}


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_3_dataset_images.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # 1. Remove old setup/badge cells if they exist (to avoid duplicates)
    cells_to_remove = []
    for i, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        # Identify existing setup cell by checks
        if (cell["cell_type"] == "code" and "@title 🚀 Environment Setup" in source):
             cells_to_remove.append(i)
        # Identify existing badge by link
        if (cell["cell_type"] == "markdown" and "colab-badge.svg" in source):
             cells_to_remove.append(i)

    for i in sorted(cells_to_remove, reverse=True):
        notebook["cells"].pop(i)
        print(f"Removed existing setup/badge cell at index {i}")

    # 2. Insert Badge at TOP (Index 0)
    # Check if first cell is header, insert badge inside it or before it?
    # User requested badge and URL. A separate cell is cleanest.
    notebook["cells"].insert(0, COLAB_BADGE)
    print("✅ Inserted Colab badge at top.")

    # 3. Insert Setup Cell after badge (Index 1)
    notebook["cells"].insert(1, SETUP_CELL)
    print("✅ Inserted Setup cell at index 1.")

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4) # Using indent 4 to match common formats, or we can use 1 like the other script
    
    print(f"\nSuccessfully updated {notebook_path.name}")


if __name__ == "__main__":
    main()
