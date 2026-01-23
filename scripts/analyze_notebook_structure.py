#!/usr/bin/env python3
"""Analyze notebook structure with proper UTF-8 encoding."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_2_activation_max.ipynb"
    
    # Read with proper UTF-8 encoding
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    cells = notebook["cells"]
    
    print(f"Total cells: {len(cells)}\n")
    print("Cell Structure:\n")
    print(f"{'#':>3} {'Type':4} | Content Preview")
    print("-" * 80)
    
    for i, cell in enumerate(cells):
        cell_type = cell["cell_type"]
        source = "".join(cell.get("source", []))
        
        # Extract section heading if it's markdown
        preview = ""
        if cell_type == "markdown":
            # Get first line (usually the heading)
            first_line = source.split("\n")[0]
            if first_line.startswith("#"):
                preview = first_line[:70]
            else:
                preview = source[:70].replace("\n", " ")
        else:
            # For code cells, show first meaningful line
            lines = [l.strip() for l in source.split("\n") if l.strip() and not l.strip().startswith("#")]
            if lines:
                preview = f"CODE: {lines[0][:60]}"
            else:
                preview = "CODE: (comments only)"
        
        print(f"{i:3d} {cell_type:4s} | {preview}")
    
    print("\n" + "=" * 80)
    print("Analysis complete")


if __name__ == "__main__":
    main()
