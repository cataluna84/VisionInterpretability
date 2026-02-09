"""Script to add performance metrics tracking and summary table to the segment_3 notebook."""

import json
from pathlib import Path

notebook_path = Path(r"c:\Users\cataluna84\Documents\Workspace\VisionInterpretability\notebooks\cataluna84__segment_3_dataset_images_imagenet_validation.ipynb")

# Read notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Track cell indices for insertion
cleanup_cell_idx = None

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    source = ''.join(cell['source'])
    
    # 1. Update CONFIG cell - add optimization_mode
    if 'CONFIG = {' in source and '"optim_steps"' in source:
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
            '    # --- Performance Mode ---\n',
            '    "optimization_mode": "baseline",  # "baseline", "compile", "optimized"\n',
            "}\n",
            "\n",
            'print("Configuration:")\n',
            "for k, v in CONFIG.items():\n",
            '    print(f"  {k}: {v}")\n',
        ]
        cell['outputs'] = []
        print("Updated CONFIG cell with optimization_mode")
    
    # 2. Update batch processing cell - add timing breakdown
    if 'for tensors, images, labels, ids in tqdm(streamer' in source:
        cell['source'] = [
            "# Process batches with TIMING and LOGGING\n",
            "import time\n",
            "\n",
            "total_processed = 0\n",
            "batch_count = 0\n",
            "\n",
            "# Timing accumulators for metrics\n",
            "time_data_loading = 0.0\n",
            "time_inference = 0.0\n",
            "time_tracking = 0.0\n",
            "\n",
            'print(f"Processing batches using device: {device}...")\n',
            'print(f"Optimization mode: {CONFIG[\'optimization_mode\']}")\n',
            "t_start_total = time.time()\n",
            "\n",
            '# Apply torch.compile if requested\n',
            'if CONFIG["optimization_mode"] in ["compile", "optimized"]:\n',
            '    try:\n',
            '        compiled_model = torch.compile(model, mode="reduce-overhead")\n',
            '        print("Model compiled with torch.compile!")\n',
            '    except Exception as e:\n',
            '        print(f"torch.compile failed: {e}, using eager mode")\n',
            '        compiled_model = model\n',
            "else:\n",
            "    compiled_model = model\n",
            "\n",
            'for tensors, images, labels, ids in tqdm(streamer, desc="Batches"):\n',
            "    t_batch_start = time.time()\n",
            "    \n",
            "    # 1. Move Data to GPU (Bottleneck: Host-to-Device transfer)\n",
            '    t_load_start = time.time()\n',
            '    if CONFIG["optimization_mode"] == "optimized":\n',
            "        tensors = tensors.to(device, non_blocking=True)\n",
            "    else:\n",
            "        tensors = tensors.to(device)\n",
            '    time_data_loading += time.time() - t_load_start\n',
            "    \n",
            "    # 2. Forward Pass (Inference)\n",
            '    t_infer_start = time.time()\n',
            "    with torch.no_grad():\n",
            "        _ = compiled_model(tensors)\n",
            '    if device.type == "cuda":\n',
            "        torch.cuda.synchronize()  # Accurate timing\n",
            '    time_inference += time.time() - t_infer_start\n',
            "    \n",
            "    # 3. Extract Activations\n",
            "    activations = extractor.get_max_activations_per_channel()\n",
            "    \n",
            "    # 4. Update Tracker (CPU)\n",
            '    t_track_start = time.time()\n',
            "    tracker.update(\n",
            '        activations[:, :CONFIG["num_neurons"]],\n',
            "        images,\n",
            "        ids,\n",
            "        labels,\n",
            "    )\n",
            '    time_tracking += time.time() - t_track_start\n',
            "    \n",
            "    # Timing calculations\n",
            "    batch_duration = time.time() - t_batch_start\n",
            "    current_batch_size = len(tensors)\n",
            "    throughput = current_batch_size / batch_duration\n",
            "    \n",
            "    total_processed += current_batch_size\n",
            "    batch_count += 1\n",
            "    \n",
            "    # 5. Logging to WandB\n",
            "    if batch_count % 10 == 0:\n",
            "        wandb.log({\n",
            '            "batch": batch_count,\n',
            '            "samples_processed": total_processed,\n',
            '            "batch_duration_sec": batch_duration,\n',
            '            "throughput_img_per_sec": throughput,\n',
            "        })\n",
            "\n",
            "    # Memory Cleanup\n",
            "    del tensors, images\n",
            "    if batch_count % 50 == 0:\n",
            "        torch.cuda.empty_cache()\n",
            "\n",
            "t_end_total = time.time()\n",
            "total_time = t_end_total - t_start_total\n",
            "throughput_avg = total_processed / total_time if total_time > 0 else 0\n",
            "\n",
            'print(f"\\nProcessed {total_processed:,} samples in {batch_count} batches")\n',
            'print(f"Total time: {total_time:.2f} seconds")\n',
            'print(f"Average throughput: {throughput_avg:.2f} img/sec")\n',
            'print(f"\\nTiming Breakdown:")\n',
            'print(f"  Data loading: {time_data_loading:.2f}s ({100*time_data_loading/total_time:.1f}%)")\n',
            'print(f"  Inference:    {time_inference:.2f}s ({100*time_inference/total_time:.1f}%)")\n',
            'print(f"  Tracking:     {time_tracking:.2f}s ({100*time_tracking/total_time:.1f}%)")\n',
        ]
        cell['outputs'] = []
        print("Updated batch processing cell with timing breakdown")
    
    # Find cleanup cell index
    if 'wandb.finish()' in source:
        cleanup_cell_idx = idx

# 3. Insert metrics summary cell after cleanup
if cleanup_cell_idx is not None:
    metrics_markdown_cell = {
        "cell_type": "markdown",
        "id": "metrics-markdown",
        "metadata": {},
        "source": [
            "## 9. Performance Metrics Summary\n",
            "\n",
            "This table summarizes the performance of the current run for benchmarking and optimization comparisons.\n"
        ]
    }
    
    metrics_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "metrics-summary",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Performance Metrics Summary Table\n",
            "import pandas as pd\n",
            "\n",
            "# Collect metrics\n",
            "gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"\n",
            "gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0\n",
            "\n",
            "metrics_data = {\n",
            '    "Metric": [\n',
            '        "Total Samples Processed",\n',
            '        "Total Time (sec)",\n',
            '        "Average Throughput (img/sec)",\n',
            '        "Data Loading Time (sec)",\n',
            '        "Inference Time (sec)",\n',
            '        "Tracking Time (sec)",\n',
            '        "Dataset Split",\n',
            '        "Optimization Mode",\n',
            '        "GPU",\n',
            '        "GPU Memory Peak (GB)",\n',
            '        "Batch Size",\n',
            "    ],\n",
            '    "Value": [\n',
            '        f"{total_processed:,}",\n',
            '        f"{total_time:.2f}",\n',
            '        f"{throughput_avg:.2f}",\n',
            '        f"{time_data_loading:.2f}",\n',
            '        f"{time_inference:.2f}",\n',
            '        f"{time_tracking:.2f}",\n',
            '        CONFIG["dataset_split"],\n',
            '        CONFIG["optimization_mode"],\n',
            "        gpu_name,\n",
            '        f"{gpu_memory_peak:.2f}",\n',
            '        str(CONFIG["batch_size"]),\n',
            "    ]\n",
            "}\n",
            "\n",
            "metrics_df = pd.DataFrame(metrics_data)\n",
            "\n",
            'print("\\n" + "="*60)\n',
            'print("PERFORMANCE METRICS SUMMARY")\n',
            'print("="*60)\n',
            "print(metrics_df.to_string(index=False))\n",
            'print("="*60)\n',
            "\n",
            "# Log to WandB\n",
            "if wandb.run is not None:\n",
            '    wandb.log({"performance_metrics": wandb.Table(dataframe=metrics_df)})\n',
            '    print("\\nMetrics logged to WandB.")\n',
        ]
    }
    
    # Insert after cleanup cell
    nb['cells'].insert(cleanup_cell_idx + 1, metrics_markdown_cell)
    nb['cells'].insert(cleanup_cell_idx + 2, metrics_code_cell)
    print("Added metrics summary cells after cleanup")
else:
    print("WARNING: Could not find cleanup cell")

# Write notebook back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\n✅ Notebook updated with performance metrics!")
