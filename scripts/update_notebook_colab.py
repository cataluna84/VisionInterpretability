#!/usr/bin/env python3
"""Script to update the notebook with Colab setup cell and enhanced markdown.

This script programmatically modifies the Jupyter notebook to add:
1. An "Open in Colab" badge in the header
2. A Colab setup cell that handles repository cloning and dependency installation

Usage:
    python scripts/update_notebook_colab.py
"""
import json
from pathlib import Path


def main():
    """Update the notebook with Colab-compatible cells."""
    # Path to the notebook
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_1_intro.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return
    
    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Update the first markdown cell with Colab badge
    header_source = [
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_1_intro.ipynb)\n",
        "\n",
        "# Vision Interpretability: Decoding CNNs\n",
        "\n",
        "Welcome to this interactive tutorial on **Computer Vision** and **Convolutional Neural Networks (CNNs)**.\n",
        "\n",
        "## What You'll Learn\n",
        "\n",
        "1. **Image Representation** — How computers \"see\" images as tensors\n",
        "2. **Convolution Operations** — The math behind edge detection, blur, and sharpening\n",
        "3. **Building a CNN** — Train a model from scratch on ImageNette\n",
        "4. **Feature Visualization** — See what patterns each layer detects\n",
        "5. **Interpretability Methods** — Understand *why* a model makes predictions\n",
        "   - Saliency Maps (Vanilla Gradients)\n",
        "   - Grad-CAM (Class Activation Mapping)\n",
        "\n",
        "Let's decode the black box! 🧠\n"
    ]
    
    # Create the Colab setup cell
    colab_setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "colab_setup",
        "metadata": {},
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
            "    print(\"✅ Local setup complete!\")\n"
        ]
    }
    
    # Update the imports cell with docstring
    imports_source = [
        '"""Import required libraries and custom modules."""\n',
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torchvision\n",
        "import torchvision.transforms as transforms\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "import requests\n",
        "from io import BytesIO\n",
        "\n",
        "# Import our custom modules\n",
        "from segment_1_intro import models, visualize, data\n",
        "\n",
        "# Configuration\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        'print(f"🖥️ Using device: {device}")\n',
        "\n",
        "# Visualization settings\n",
        "plt.rcParams['figure.figsize'] = (10, 6)\n",
        "plt.rcParams['font.size'] = 12\n"
    ]
    
    # Apply changes
    # 1. Update the header markdown cell (first cell)
    notebook["cells"][0]["source"] = header_source
    
    # 2. Insert the Colab setup cell after the header
    notebook["cells"].insert(1, colab_setup_cell)
    
    # 3. Update the imports cell (now at index 2)
    notebook["cells"][2]["source"] = imports_source
    
    # Write the updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Updated notebook: {notebook_path}")
    print("   - Added 'Open in Colab' badge")
    print("   - Added Colab setup cell")
    print("   - Enhanced imports cell with docstring")


if __name__ == "__main__":
    main()
