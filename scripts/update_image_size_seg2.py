#!/usr/bin/env python3
"""Update image sizes from 128 to 320 across the Segment 2 notebook."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_2_activation_max.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace various patterns
    replacements = [
        # Function defaults
        ('image_size: int = 128', 'image_size: int = 320'),
        ('image_size=128', 'image_size=320'),
        # Documented values
        ('Image size: 128x128', 'Image size: 320x320'),
        ('128x128', '320x320'),
        # Circuit visualization (smaller for speed) - keep reasonable
        ('image_size: int = 96', 'image_size: int = 128'),
        ('image_size=96', 'image_size=128'),
    ]
    
    changes = 0
    for old, new in replacements:
        count = content.count(old)
        if count > 0:
            content = content.replace(old, new)
            changes += count
            print(f"  Replaced {count}x: '{old}' → '{new}'")
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✅ Updated {changes} occurrences of image sizes")
    print("   Main visualizations: 128 → 320 (higher quality)")
    print("   Circuit visualizations: 96 → 128 (balanced)")


if __name__ == "__main__":
    main()
