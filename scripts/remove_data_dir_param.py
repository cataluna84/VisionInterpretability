#!/usr/bin/env python3
"""Remove data_dir parameters from notebook load_imagenette calls."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_1_intro.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove data_dir="../data" from all load_imagenette calls
    replacements = [
        (', data_dir=\\"../data\\"', ''),
        (', data_dir="../data"', ''),
        ('data_dir=\\"../data\\", ', ''),
        ('data_dir="../data", ', ''),
    ]
    
    changes = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changes += 1
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✅ Removed data_dir parameter from {changes} locations")
    print("   Data directory is now controlled by global DATA_DIR in data.py")


if __name__ == "__main__":
    main()
