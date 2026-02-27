"""
Step 1: Read the current notebook and check the experiment-log-table cell.
Step 2: Ensure it has exactly: headers + separator + Run 001 data row.
Step 3: Write back.
"""
import json
import sys

NB = r"c:\Users\cataluna84\Documents\Workspace\VisionInterpretability\notebooks\cataluna84__segment_3_dataset_images_imagenet_validation.ipynb"

# Step 1: Read
print("Reading notebook...")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)
print(f"  Total cells: {len(nb['cells'])}")

# Step 2: Find and update the experiment-log-table cell
found = False
for cell in nb["cells"]:
    if cell.get("id") == "experiment-log-table":
        found = True
        print(f"  Found experiment-log-table cell")
        print(f"  Current source lines: {len(cell['source'])}")
        for i, line in enumerate(cell["source"]):
            print(f"    [{i}] {line.rstrip()}")

        # Set the correct content
        cell["source"] = [
            "| Run | Layer | Neurons | Split | Samples | Batch | Steps | Res | Mode | Time | Throughput | GPU Mem (GB) | W&B |\n",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
            "| 001 | mixed4a | 10 | validation | 50,000 | 96 | 1024 | 1024 | baseline | 997.5s | 50.1 | 5.12 | [Link](https://wandb.ai/cataluna84/vision-interpretability/runs/h5ehmgar) |\n",
        ]
        print("  Updated to: headers + separator + Run 001 row")
        break

if not found:
    print("ERROR: experiment-log-table cell not found!")
    sys.exit(1)

# Step 3: Write back
print("Writing notebook...")
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

# Validate
with open(NB, "r", encoding="utf-8") as f:
    validated = json.load(f)
print(f"  Validated: {len(validated['cells'])} cells")

# Double-check the cell is correct after re-reading
for cell in validated["cells"]:
    if cell.get("id") == "experiment-log-table":
        print(f"  Final source lines: {len(cell['source'])}")
        for i, line in enumerate(cell["source"]):
            print(f"    [{i}] {line.rstrip()}")
        break

print("\nDone!")
