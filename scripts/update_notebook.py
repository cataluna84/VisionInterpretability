"""Script to update the notebook with negative optimization support."""
import json

# New source for the generate-optimized cell
NEW_OPTIMIZED_CELL_SOURCE = [
    "from lucent.optvis import render, objectives\n",
    "\n",
    "optimized_images = {}  # Positive optimization (maximize activation)\n",
    "optimized_negative_images = {}  # Negative optimization (minimize activation)\n",
    "\n",
    "if CONFIG[\"generate_optimized\"]:\n",
    "    print(f\"Generating optimized examples for {CONFIG['num_neurons']} neurons...\")\n",
    "    print(\"This generates TWO images per neuron:\")\n",
    "    print(\"  - Positive: What the neuron WANTS to see (maximize activation)\")\n",
    "    print(\"  - Negative: What the neuron AVOIDS (minimize activation)\")\n",
    "    print()\n",
    "    \n",
    "    for n in tqdm(range(CONFIG[\"num_neurons\"]), desc=\"Optimizing\"):\n",
    "        # --- Positive Optimization (maximize activation) ---\n",
    "        try:\n",
    "            objective = f\"{CONFIG['layer_name']}:{n}\"\n",
    "            result = render.render_vis(\n",
    "                model,\n",
    "                objective,\n",
    "                show_image=False,\n",
    "                show_inline=False,\n",
    "                thresholds=(512,),\n",
    "            )\n",
    "            \n",
    "            if result and len(result) > 0:\n",
    "                img_array = result[0][0]\n",
    "                img_array = (img_array * 255).astype(np.uint8)\n",
    "                optimized_images[n] = Image.fromarray(img_array)\n",
    "            else:\n",
    "                optimized_images[n] = None\n",
    "                \n",
    "        except Exception as e:\n",
    "            print(f\"  Failed positive optimization for neuron {n}: {e}\")\n",
    "            optimized_images[n] = None\n",
    "        \n",
    "        # --- Negative Optimization (minimize activation) ---\n",
    "        try:\n",
    "            # Create a negated channel objective using lucent's objectives API\n",
    "            positive_objective = objectives.channel(CONFIG['layer_name'], n)\n",
    "            negative_objective = -1 * positive_objective\n",
    "            \n",
    "            result = render.render_vis(\n",
    "                model,\n",
    "                negative_objective,\n",
    "                show_image=False,\n",
    "                show_inline=False,\n",
    "                thresholds=(512,),\n",
    "            )\n",
    "            \n",
    "            if result and len(result) > 0:\n",
    "                img_array = result[0][0]\n",
    "                img_array = (img_array * 255).astype(np.uint8)\n",
    "                optimized_negative_images[n] = Image.fromarray(img_array)\n",
    "            else:\n",
    "                optimized_negative_images[n] = None\n",
    "                \n",
    "        except Exception as e:\n",
    "            print(f\"  Failed negative optimization for neuron {n}: {e}\")\n",
    "            optimized_negative_images[n] = None\n",
    "    \n",
    "    pos_count = sum(1 for v in optimized_images.values() if v is not None)\n",
    "    neg_count = sum(1 for v in optimized_negative_images.values() if v is not None)\n",
    "    print(f\"Generated {pos_count} positive and {neg_count} negative optimized images\")\n",
    "else:\n",
    "    print(\"Skipping optimized example generation\")"
]

# New visualization function that includes the negative optimized column
NEW_VIZ_FUNCTION_SOURCE = [
    "def plot_neuron_spectrum(neuron_idx, spectrum, optimized_img=None, optimized_neg_img=None, figsize=(20, 4)):\n",
    "    \"\"\"\n",
    "    Plot the activation spectrum for a single neuron.\n",
    "    \n",
    "    Shows: Negative Optimized | Minimum | Slightly- | Slightly+ | Maximum | Positive Optimized\n",
    "    \"\"\"\n",
    "    categories = [\"minimum\", \"slight_negative\", \"slight_positive\", \"maximum\"]\n",
    "    titles = [\"Optimized\\n(avoids)\", \"Minimum\\n(most suppressing)\", \"Slightly Negative\\n(near threshold)\", \n",
    "              \"Slightly Positive\\n(barely activating)\", \"Maximum\\n(most activating)\", \"Optimized\\n(seeks)\"]\n",
    "    \n",
    "    # Determine grid size\n",
    "    k = CONFIG[\"samples_per_category\"]\n",
    "    ncols = 6  # neg_opt + 4 categories + pos_opt\n",
    "    nrows = k\n",
    "    \n",
    "    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)\n",
    "    fig.suptitle(f\"Neuron {neuron_idx} ({CONFIG['layer_name']})\", fontsize=14, fontweight=\"bold\")\n",
    "    \n",
    "    # Column 0: Negative optimized (what the neuron avoids)\n",
    "    col = 0\n",
    "    if nrows > 1:\n",
    "        axes[0, col].set_title(titles[col], fontsize=10, color=\"blue\")\n",
    "        for row in range(nrows):\n",
    "            axes[row, col].axis(\"off\")\n",
    "        if optimized_neg_img is not None:\n",
    "            mid = nrows // 2\n",
    "            axes[mid, col].imshow(optimized_neg_img)\n",
    "    else:\n",
    "        axes[col].set_title(titles[col], fontsize=10, color=\"blue\")\n",
    "        axes[col].axis(\"off\")\n",
    "        if optimized_neg_img is not None:\n",
    "            axes[col].imshow(optimized_neg_img)\n",
    "    \n",
    "    # Columns 1-4: Dataset samples\n",
    "    for col, cat in enumerate(categories, start=1):\n",
    "        samples = spectrum.get(cat, [])\n",
    "        \n",
    "        if nrows > 1:\n",
    "            axes[0, col].set_title(titles[col], fontsize=10)\n",
    "        else:\n",
    "            axes[col].set_title(titles[col], fontsize=10)\n",
    "        \n",
    "        for row in range(k):\n",
    "            ax = axes[row, col] if nrows > 1 else axes[col]\n",
    "            ax.axis(\"off\")\n",
    "            \n",
    "            if row < len(samples):\n",
    "                sample = samples[row]\n",
    "                if isinstance(sample.image, Image.Image):\n",
    "                    ax.imshow(sample.image)\n",
    "                    ax.set_xlabel(f\"act={sample.activation:.2f}\", fontsize=8)\n",
    "    \n",
    "    # Column 5: Positive optimized (what the neuron seeks)\n",
    "    col = 5\n",
    "    if nrows > 1:\n",
    "        axes[0, col].set_title(titles[col], fontsize=10, color=\"red\")\n",
    "        for row in range(nrows):\n",
    "            axes[row, col].axis(\"off\")\n",
    "        if optimized_img is not None:\n",
    "            mid = nrows // 2\n",
    "            axes[mid, col].imshow(optimized_img)\n",
    "    else:\n",
    "        axes[col].set_title(titles[col], fontsize=10, color=\"red\")\n",
    "        axes[col].axis(\"off\")\n",
    "        if optimized_img is not None:\n",
    "            axes[col].imshow(optimized_img)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    return fig"
]

# New plotting cell that uses both optimized images
NEW_PLOT_CELL_SOURCE = [
    "# Visualize spectrum for all tracked neurons\n",
    "print(f\"Visualizing activation spectrum for {CONFIG['num_neurons']} neurons...\\n\")\n",
    "\n",
    "figures = []\n",
    "for n in range(CONFIG[\"num_neurons\"]):\n",
    "    spectrum = tracker.get_spectrum(n)\n",
    "    opt_img = optimized_images.get(n)\n",
    "    opt_neg_img = optimized_negative_images.get(n)\n",
    "    \n",
    "    fig = plot_neuron_spectrum(n, spectrum, opt_img, opt_neg_img)\n",
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
    
    if cell_id == 'generate-optimized':
        print(f"Updating 'generate-optimized' cell at index {i}")
        nb['cells'][i]['source'] = NEW_OPTIMIZED_CELL_SOURCE
        nb['cells'][i]['outputs'] = []  # Clear old outputs
        
    elif cell_id == 'visualize-spectrum':
        print(f"Updating 'visualize-spectrum' cell at index {i}")
        nb['cells'][i]['source'] = NEW_VIZ_FUNCTION_SOURCE
        nb['cells'][i]['outputs'] = []
        
    elif cell_id == 'plot-all-neurons':
        print(f"Updating 'plot-all-neurons' cell at index {i}")
        nb['cells'][i]['source'] = NEW_PLOT_CELL_SOURCE
        nb['cells'][i]['outputs'] = []

# Also update WANDB logging to include negative images
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 'wandb-log':
        print(f"Updating 'wandb-log' cell at index {i}")
        nb['cells'][i]['source'] = [
            "print(\"Logging results to WANDB...\")\n",
            "\n",
            "# Log spectrum grids as images\n",
            "for n, fig in enumerate(figures):\n",
            "    wandb.log({f\"spectrum/neuron_{n}\": wandb.Image(fig)})\n",
            "    plt.close(fig)\n",
            "\n",
            "# Create summary table\n",
            "columns = [\"Neuron\", \"Category\", \"Rank\", \"Activation\", \"Label\", \"Image\"]\n",
            "table = wandb.Table(columns=columns)\n",
            "\n",
            "for n in range(CONFIG[\"num_neurons\"]):\n",
            "    spectrum = tracker.get_spectrum(n)\n",
            "    for category in [\"minimum\", \"slight_negative\", \"slight_positive\", \"maximum\"]:\n",
            "        samples = spectrum.get(category, [])\n",
            "        for rank, sample in enumerate(samples):\n",
            "            if isinstance(sample.image, Image.Image):\n",
            "                table.add_data(\n",
            "                    n,\n",
            "                    category,\n",
            "                    rank,\n",
            "                    sample.activation,\n",
            "                    sample.label,\n",
            "                    wandb.Image(sample.image),\n",
            "                )\n",
            "\n",
            "wandb.log({\"activation_spectrum_table\": table})\n",
            "\n",
            "# Log positive optimized images\n",
            "if CONFIG[\"generate_optimized\"]:\n",
            "    for n, img in optimized_images.items():\n",
            "        if img is not None:\n",
            "            wandb.log({f\"optimized_positive/neuron_{n}\": wandb.Image(img)})\n",
            "\n",
            "# Log negative optimized images\n",
            "if CONFIG[\"generate_optimized\"]:\n",
            "    for n, img in optimized_negative_images.items():\n",
            "        if img is not None:\n",
            "            wandb.log({f\"optimized_negative/neuron_{n}\": wandb.Image(img)})\n",
            "\n",
            "print(f\"Logged spectrum for {CONFIG['num_neurons']} neurons to WANDB\")"
        ]
        nb['cells'][i]['outputs'] = []

# Write updated notebook
with open('notebooks/cataluna84__segment_3_dataset_images.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("\nNotebook updated successfully!")
