"""Script to update the notebook to use plot_neuron_spectrum_distill from visualization module."""
import json

# New visualization cell that imports and uses plot_neuron_spectrum_distill
NEW_VIZ_CELL_SOURCE = [
    "from segment_3_dataset_images.visualization import plot_neuron_spectrum_distill\n",
    "\n",
    "# The plot_neuron_spectrum_distill function creates Distill.pub style visualizations\n",
    "# showing the full activation spectrum:\n",
    "# | Neg Optimized | Min Grid | Slight- | Slight+ | Max Grid | Pos Optimized |\n",
    "print(f\"Using Distill.pub style visualization for {CONFIG['num_neurons']} neurons\")"
]

# New plotting cell that uses plot_neuron_spectrum_distill
NEW_PLOT_CELL_SOURCE = [
    "# Visualize spectrum for all tracked neurons using Distill.pub style\n",
    "print(f\"Visualizing activation spectrum for {CONFIG['num_neurons']} neurons...\\n\")\n",
    "\n",
    "figures = []\n",
    "for n in range(CONFIG[\"num_neurons\"]):\n",
    "    spectrum = tracker.get_spectrum(n)\n",
    "    pos_img = optimized_images.get(n)\n",
    "    neg_img = optimized_negative_images.get(n)\n",
    "    \n",
    "    # Use Distill.pub style visualization from the visualization module\n",
    "    fig = plot_neuron_spectrum_distill(\n",
    "        neuron_idx=n,\n",
    "        layer_name=CONFIG[\"layer_name\"],\n",
    "        spectrum=spectrum,\n",
    "        optimized_img=pos_img,\n",
    "        negative_optimized_img=neg_img,\n",
    "    )\n",
    "    figures.append(fig)\n",
    "    plt.show()\n",
    "    print()"
]

# Read notebook
with open('notebooks/cataluna84__segment_3_dataset_images.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the cells
for i, cell in enumerate(nb['cells']):
    cell_id = cell.get('id', '')
    
    if cell_id == 'visualize-spectrum':
        print(f"Updating 'visualize-spectrum' cell at index {i} to import plot_neuron_spectrum_distill")
        nb['cells'][i]['source'] = NEW_VIZ_CELL_SOURCE
        nb['cells'][i]['outputs'] = []
        
    elif cell_id == 'plot-all-neurons':
        print(f"Updating 'plot-all-neurons' cell at index {i} to use plot_neuron_spectrum_distill")
        nb['cells'][i]['source'] = NEW_PLOT_CELL_SOURCE
        nb['cells'][i]['outputs'] = []

# Write updated notebook
with open('notebooks/cataluna84__segment_3_dataset_images.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("\nNotebook updated to use plot_neuron_spectrum_distill!")
