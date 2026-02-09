"""Script to update CONFIG and optimization parameters in the segment_3 notebook."""

import json
from pathlib import Path

notebook_path = Path(r"c:\Users\cataluna84\Documents\Workspace\VisionInterpretability\notebooks\cataluna84__segment_3_dataset_images_imagenet_validation.ipynb")

# Read notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update cells
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = ''.join(cell['source'])
    
    # 1. Update CONFIG cell
    if 'CONFIG = {' in source and '"generate_optimized": True' in source:
        cell['source'] = [
            "# Configuration\n",
            "CONFIG = {\n",
            '    "num_neurons": 10,           # First 10 neurons of mixed4a (512 total)\n',
            '    "samples_per_category": 9,    # 9 samples per category (min, slight-, slight+, max)\n',
            '    "max_samples": None,         # Process ALL samples (WARNING: ~1.28M for train)\n',
            '                                 # Set to 1000 or 10000 for faster testing.\n',
            '    "batch_size": 128,             # Batch size for streaming inference\n',
            '    "layer_name": "mixed4a",      # Target layer in InceptionV1\n',
            '    "wandb_project": "vision-interpretability",\n',
            '    "wandb_run_name": "mixed4a-activation-spectrum-v3",\n',
            '    "generate_optimized": True,   # Generate gradient-ascent visualizations\n',
            '    # --- Dataset & Optimization Parameters ---\n',
            '    "dataset_split": "validation", # "train" (1.28M) or "validation" (50K)\n',
            '    "optim_resolution": 512,       # Image size for optimization (pixels)\n',
            '    "optim_steps": 512,            # Number of gradient ascent steps\n',
            "}\n",
            "\n",
            'print("Configuration:")\n',
            "for k, v in CONFIG.items():\n",
            '    print(f"  {k}: {v}")\n',
        ]
        # Clear outputs since config changed
        cell['outputs'] = []
        print("Updated CONFIG cell")
    
    # 2. Update ImageNetStreamer cell
    if 'ImageNetStreamer' in source and 'streamer = ImageNetStreamer' in source:
        cell['source'] = [
            "from segment_3_dataset_images.activation_pipeline import ImageNetStreamer\n",
            "\n",
            "# Setup streaming from HuggingFace\n",
            "streamer = ImageNetStreamer(\n",
            '    batch_size=CONFIG["batch_size"],\n',
            '    max_samples=CONFIG["max_samples"],\n',
            '    split=CONFIG["dataset_split"],\n',
            ")\n",
            "\n",
            "max_samples_str = f\"{CONFIG['max_samples']:,}\" if CONFIG['max_samples'] is not None else \"ALL\"\n",
            "split_size = \"~1.28M\" if CONFIG['dataset_split'] == \"train\" else \"~50K\"\n",
            "print(f\"Streaming up to {max_samples_str} samples from ImageNet-1k ({CONFIG['dataset_split']} split, {split_size})\")\n",
            "print(f\"Batch size: {CONFIG['batch_size']}\")\n",
            "\n",
            "if CONFIG['max_samples'] is not None:\n",
            "    print(f\"Estimated batches: {CONFIG['max_samples'] // CONFIG['batch_size']}\")\n",
            "else:\n",
            '    print("Estimated batches: Unknown (streaming infinite/all)")',
        ]
        cell['outputs'] = []
        print("Updated ImageNetStreamer cell")
    
    # 3. Update optimization cell
    if 'from lucent.optvis import render, objectives' in source and 'optimized_images = {}' in source:
        cell['source'] = [
            "from lucent.optvis import render, objectives, param\n",
            "\n",
            "optimized_images = {}  # Positive optimization (maximize activation)\n",
            "optimized_negative_images = {}  # Negative optimization (minimize activation)\n",
            "\n",
            'if CONFIG["generate_optimized"]:\n',
            '    print(f"Generating optimized examples for {CONFIG[\'num_neurons\']} neurons...")\n',
            '    print(f"Resolution: {CONFIG[\'optim_resolution\']}px, Steps: {CONFIG[\'optim_steps\']}")\n',
            '    print("This generates TWO images per neuron:")\n',
            '    print("  - Positive: What the neuron WANTS to see (maximize activation)")\n',
            '    print("  - Negative: What the neuron AVOIDS (minimize activation)")\n',
            "    print()\n",
            "    \n",
            '    for n in tqdm(range(CONFIG["num_neurons"]), desc="Optimizing"):\n',
            "        # --- Positive Optimization (maximize activation) ---\n",
            "        try:\n",
            '            objective = f"{CONFIG[\'layer_name\']}:{n}"\n',
            "            result = render.render_vis(\n",
            "                model,\n",
            "                objective,\n",
            "                show_image=False,\n",
            "                show_inline=False,\n",
            '                thresholds=(CONFIG["optim_steps"],),\n',
            '                param_f=lambda: param.image(CONFIG["optim_resolution"]),\n',
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
            '            print(f"  Failed positive optimization for neuron {n}: {e}")\n',
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
            '                thresholds=(CONFIG["optim_steps"],),\n',
            '                param_f=lambda: param.image(CONFIG["optim_resolution"]),\n',
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
            '            print(f"  Failed negative optimization for neuron {n}: {e}")\n',
            "            optimized_negative_images[n] = None\n",
            "    \n",
            "    pos_count = sum(1 for v in optimized_images.values() if v is not None)\n",
            "    neg_count = sum(1 for v in optimized_negative_images.values() if v is not None)\n",
            '    print(f"Generated {pos_count} positive and {neg_count} negative optimized images")\n',
            "else:\n",
            '    print("Skipping optimized example generation")',
        ]
        cell['outputs'] = []
        print("Updated optimization cell")

# Write notebook back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\n✅ Notebook updated successfully!")
