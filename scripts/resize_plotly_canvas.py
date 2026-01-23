#!/usr/bin/env python3
"""Increase Plotly canvas and subplot sizes in Segment 2 notebook."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_2_activation_max.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Update canvas dimensions (2x larger)
    replacements = [
        # Main figure size
        ('height=700', 'height=1400'),
        ('width=1000', 'width=2000'),
        # Reduce spacing to give more room to images
        ('horizontal_spacing=0.03', 'horizontal_spacing=0.02'),
        ('vertical_spacing=0.08', 'vertical_spacing=0.04'),
    ]
    
    changes = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changes += 1
            print(f"  {old} → {new}")
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✅ Updated {changes} size parameters")
    print("   Canvas: 1000×700 → 2000×1400 (2x larger)")
    print("   Spacing: Reduced for larger subplots")


if __name__ == "__main__":
    main()
