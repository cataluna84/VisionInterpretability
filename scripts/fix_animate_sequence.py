"""Fix animate_sequence CSS ID conflicts in Lucent notebooks.

This script injects a fix cell into Jupyter notebooks that redefines
the `animate_sequence` function to use unique CSS IDs (via uuid) instead
of the hardcoded '#animation' and '@keyframes play' identifiers.

This prevents the "overwrite" issue where multiple animations all display
the last rendered sprite sheet.
"""

import json
import sys
from pathlib import Path


def create_fix_cells():
    """Create the markdown and code cells for the animate_sequence fix."""
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Fix: Unique ID Animation Sequence\n",
            "\n",
            "The original `animate_sequence` function uses hardcoded CSS IDs "
            "(`#animation`) and keyframe names (`play`), causing all animations "
            "to conflict and display only the last one rendered. This fix "
            "generates unique identifiers for each animation instance."
        ]
    }
    
    fix_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# FIX: Override animate_sequence to use unique CSS IDs\n",
            "import uuid\n",
            "from string import Template\n",
            "import numpy as np\n",
            "from lucent.misc.io.showing import _display_html, _image_url\n",
            "\n",
            "def animate_sequence(sequence, domain=(0, 1), fmt='png'):\n",
            '    """Animate a sequence of images with unique CSS identifiers.\n',
            "    \n",
            "    This version generates a unique ID for each animation to prevent\n",
            "    CSS conflicts when multiple animations are rendered.\n",
            '    """\n',
            "    steps, height, width, _ = sequence.shape\n",
            "    sequence = np.concatenate(sequence, 1)\n",
            "    \n",
            "    # Generate unique ID for this animation instance\n",
            "    anim_id = 'animation_' + str(uuid.uuid4()).replace('-', '')\n",
            "    \n",
            "    code = Template('''\n",
            "    <style>\n",
            "        #${anim_id} {\n",
            "            width: ${width}px;\n",
            "            height: ${height}px;\n",
            "            background: url('$image_url') left center;\n",
            "            animation: play_${anim_id} 1s steps($steps) infinite alternate;\n",
            "        }\n",
            "        @keyframes play_${anim_id} {\n",
            "            100% { background-position: -${sequence_width}px; }\n",
            "        }\n",
            "    </style><div id='${anim_id}'></div>\n",
            "    ''').substitute(\n",
            "        anim_id=anim_id,\n",
            "        image_url=_image_url(sequence, domain=domain, fmt=fmt),\n",
            "        sequence_width=width*steps,\n",
            "        width=width,\n",
            "        height=height,\n",
            "        steps=steps,\n",
            "    )\n",
            "    _display_html(code)"
        ]
    }
    
    return md_cell, fix_cell


def inject_fix(notebook_path: Path) -> bool:
    """Inject the animate_sequence fix into a notebook.
    
    Args:
        notebook_path: Path to the Jupyter notebook.
        
    Returns:
        True if fix was injected, False if already present or not needed.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Check if fix already exists
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'uuid.uuid4()' in source and 'anim_id' in source:
                print(f"Fix already present in {notebook_path.name}")
                return False
    
    # Find the imports cell with animate_sequence
    insert_index = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'from lucent.misc.io.showing import animate_sequence' in source:
                insert_index = i + 1
                break
    
    if insert_index is None:
        print(f"No animate_sequence import found in {notebook_path.name}")
        return False
    
    # Create and insert fix cells
    md_cell, fix_cell = create_fix_cells()
    nb['cells'].insert(insert_index, md_cell)
    nb['cells'].insert(insert_index + 1, fix_cell)
    
    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    
    print(f"Fix successfully injected into {notebook_path.name} after cell {insert_index - 1}")
    return True


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        notebook_path = Path(sys.argv[1])
    else:
        # Default to neuron_interaction.ipynb
        script_dir = Path(__file__).parent
        notebook_path = script_dir.parent / "notebooks" / "lucent" / "neuron_interaction.ipynb"
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)
    
    inject_fix(notebook_path)


if __name__ == "__main__":
    main()
