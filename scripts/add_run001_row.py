"""Add Run 003 row to the Master Experiment Log table."""
import json

NB = r"c:\Users\cataluna84\Documents\Workspace\VisionInterpretability\notebooks\cataluna84__segment_3_dataset_images_imagenet_validation.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

ROW = "| 003 | mixed4a | 10 | validation | 50,000 | 96 | 256 | 224 | baseline | 1196.8s | 41.8 | 5.12 | [Link](https://wandb.ai/cataluna84/vision-interpretability/runs/ztji08nw) |\n"

for cell in nb["cells"]:
    if cell.get("id") == "experiment-log-table":
        print("Current table:")
        for line in cell["source"]:
            print(f"  {line.rstrip()}")
        cell["source"].append(ROW)
        print(f"\nAppended Run 003")
        break

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Done!")
