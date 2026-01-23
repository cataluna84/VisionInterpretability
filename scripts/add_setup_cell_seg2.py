#!/usr/bin/env python3
"""Add comprehensive Colab/Local setup cell to Segment 2 notebook."""
import json
from pathlib import Path


SETUP_CELL = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# @title 🚀 Environment Setup (Run this cell first!)\n",
        "# @markdown This cell automatically detects your environment (Colab or local)\n",
        "# @markdown and sets up all required dependencies for activation maximization.\n",
        "\n",
        "r\"\"\"\n",
        "Environment Setup for Segment 2: Activation Maximization\n",
        "=========================================================\n",
        "\n",
        "This cell handles:\n",
        "1. Environment detection (Google Colab vs local Jupyter)\n",
        "2. Repository cloning (Colab only)\n",
        "3. Dependency installation (torch-lucent, plotly, etc.)\n",
        "4. Python path configuration\n",
        "\n",
        "Dependencies Required:\n",
        "    - torch-lucent: Feature visualization library (PyTorch port of Lucid)\n",
        "    - plotly: Interactive visualizations with hover tooltips\n",
        "    - torch, torchvision: Deep learning framework\n",
        "    - matplotlib, numpy, pillow: Standard scientific computing\n",
        "\n",
        "Compatibility:\n",
        "    - Google Colab: Full support with automatic setup\n",
        "    - Local Jupyter: Works with `uv sync` pre-installed dependencies\n",
        "    - JupyterLab: Full support\n",
        "\n",
        "Author: cataluna84\n",
        "Project: VisionInterpretability\n",
        "\"\"\"\n",
        "\n",
        "import sys\n",
        "import os\n",
        "\n",
        "# ============================================================================\n",
        "# 🔍 ENVIRONMENT DETECTION\n",
        "# ============================================================================\n",
        "IN_COLAB = 'google.colab' in sys.modules\n",
        "\n",
        "print(\"=\" * 70)\n",
        "print(\"🔬 SEGMENT 2: ACTIVATION MAXIMIZATION - ENVIRONMENT SETUP\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "if IN_COLAB:\n",
        "    # ========================================================================\n",
        "    # 🌐 GOOGLE COLAB SETUP\n",
        "    # ========================================================================\n",
        "    print(\"\\n🌐 Environment: Google Colab\")\n",
        "    print(\"-\" * 50)\n",
        "    \n",
        "    # Step 1: Clone repository if not already present\n",
        "    print(\"\\n📦 Step 1/4: Checking repository...\")\n",
        "    if not os.path.exists('VisionInterpretability'):\n",
        "        print(\"   Cloning VisionInterpretability repository...\")\n",
        "        !git clone -q https://github.com/cataluna84/VisionInterpretability.git\n",
        "        print(\"   ✅ Repository cloned\")\n",
        "    else:\n",
        "        print(\"   ✅ Repository already exists\")\n",
        "    \n",
        "    # Step 2: Change to project directory\n",
        "    print(\"\\n📂 Step 2/4: Changing to project directory...\")\n",
        "    os.chdir('VisionInterpretability')\n",
        "    print(f\"   ✅ Working directory: {os.getcwd()}\")\n",
        "    \n",
        "    # Step 3: Install dependencies\n",
        "    print(\"\\n📥 Step 3/4: Installing dependencies...\")\n",
        "    print(\"   Installing: torch-lucent, plotly, opencv-python...\")\n",
        "    !pip install -q torch-lucent plotly opencv-python pillow tqdm\n",
        "    print(\"   ✅ Dependencies installed\")\n",
        "    \n",
        "    # Step 4: Configure Python path\n",
        "    print(\"\\n🔧 Step 4/4: Configuring Python path...\")\n",
        "    if 'src' not in sys.path:\n",
        "        sys.path.insert(0, 'src')\n",
        "    print(\"   ✅ Python path configured\")\n",
        "    \n",
        "    print(\"\\n\" + \"=\" * 70)\n",
        "    print(\"✅ COLAB SETUP COMPLETE!\")\n",
        "    print(\"=\" * 70)\n",
        "    print(\"\\n💡 You can now run all cells in this notebook.\")\n",
        "    print(\"📊 GPU Status:\")\n",
        "    !nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo \"   No GPU detected (CPU mode)\")\n",
        "\n",
        "else:\n",
        "    # ========================================================================\n",
        "    # 💻 LOCAL ENVIRONMENT SETUP\n",
        "    # ========================================================================\n",
        "    print(\"\\n💻 Environment: Local Jupyter\")\n",
        "    print(\"-\" * 50)\n",
        "    \n",
        "    # Step 1: Configure Python path\n",
        "    print(\"\\n🔧 Step 1/2: Configuring Python path...\")\n",
        "    \n",
        "    # Handle running from notebooks/ directory\n",
        "    current_dir = os.path.basename(os.getcwd())\n",
        "    if current_dir == 'notebooks':\n",
        "        project_root = os.path.dirname(os.getcwd())\n",
        "        src_path = os.path.join(project_root, 'src')\n",
        "    else:\n",
        "        src_path = 'src'\n",
        "    \n",
        "    if src_path not in sys.path:\n",
        "        sys.path.insert(0, src_path)\n",
        "    print(f\"   ✅ Added to path: {src_path}\")\n",
        "    \n",
        "    # Step 2: Verify dependencies\n",
        "    print(\"\\n📦 Step 2/2: Verifying dependencies...\")\n",
        "    missing_deps = []\n",
        "    \n",
        "    try:\n",
        "        import torch\n",
        "        print(f\"   ✅ PyTorch {torch.__version__}\")\n",
        "    except ImportError:\n",
        "        missing_deps.append('torch')\n",
        "        print(\"   ❌ PyTorch not found\")\n",
        "    \n",
        "    try:\n",
        "        import lucent\n",
        "        print(\"   ✅ torch-lucent\")\n",
        "    except ImportError:\n",
        "        missing_deps.append('torch-lucent')\n",
        "        print(\"   ❌ torch-lucent not found\")\n",
        "    \n",
        "    try:\n",
        "        import plotly\n",
        "        print(f\"   ✅ Plotly {plotly.__version__}\")\n",
        "    except ImportError:\n",
        "        missing_deps.append('plotly')\n",
        "        print(\"   ❌ Plotly not found\")\n",
        "    \n",
        "    if missing_deps:\n",
        "        print(f\"\\n⚠️  Missing dependencies: {', '.join(missing_deps)}\")\n",
        "        print(\"   Run: pip install \" + ' '.join(missing_deps))\n",
        "    \n",
        "    print(\"\\n\" + \"=\" * 70)\n",
        "    print(\"✅ LOCAL SETUP COMPLETE!\")\n",
        "    print(\"=\" * 70)\n",
        "    print(\"\\n💡 Dependencies should be pre-installed via: uv sync\")\n",
        "    \n",
        "    # Check GPU availability\n",
        "    try:\n",
        "        import torch\n",
        "        if torch.cuda.is_available():\n",
        "            print(f\"\\n🎮 GPU: {torch.cuda.get_device_name(0)}\")\n",
        "            print(f\"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n",
        "        else:\n",
        "            print(\"\\n⚠️  No GPU detected - running on CPU (slower)\")\n",
        "    except:\n",
        "        pass\n"
    ]
}


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_2_activation_max.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Find and remove old setup cells (if any)
    cells_to_remove = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if ("Environment Setup" in source and "IN_COLAB" in source) or \
               ("try:" in source and "import plotly" in source and len(source) < 500):
                cells_to_remove.append(i)
    
    # Remove old cells in reverse order
    for i in sorted(cells_to_remove, reverse=True):
        notebook["cells"].pop(i)
        print(f"   Removed old setup cell at position {i}")
    
    # Find the position after the title markdown cell
    insert_pos = 1  # After the Colab badge + title cell
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell.get("source", []))
            if "Segment 2: Activation Maximization" in source:
                insert_pos = i + 1
                break
    
    # Insert the new setup cell
    notebook["cells"].insert(insert_pos, SETUP_CELL)
    print(f"✅ Added comprehensive setup cell at position {insert_pos}")
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print("\n   Features:")
    print("   - 🌐 Colab: Auto-clone repo, install deps, configure path")
    print("   - 💻 Local: Path config, dependency verification")
    print("   - 🎮 GPU detection and status")
    print("   - 📊 Full documentation with emojis")


if __name__ == "__main__":
    main()
