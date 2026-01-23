#!/usr/bin/env python3
"""Update notebook to use 320px image size instead of 128."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_1_intro.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace image_size=128 with image_size=320
    changes = content.count('image_size=128')
    content = content.replace('image_size=128', 'image_size=320')
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✅ Updated {changes} occurrences: image_size=128 → image_size=320")
    print("   Using full 320px resolution for better feature visualization")


if __name__ == "__main__":
    main()
