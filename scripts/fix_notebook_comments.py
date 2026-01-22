#!/usr/bin/env python3
"""Fix notebook comments and data path.

Fixes:
1. Change "Hugging Face" references to "fast.ai S3"
2. Update data path to use project root instead of ./data
"""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_1_intro.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    changes = []
    
    # Fix 1: Hugging Face -> fast.ai S3
    if "Hugging Face" in content:
        content = content.replace("from Hugging Face", "from fast.ai S3")
        content = content.replace("Hugging Face", "fast.ai")
        changes.append("Fixed 'Hugging Face' -> 'fast.ai S3'")
    
    # Fix 2: Update data path to project root (../data instead of ./data)
    # This ensures data is stored at VisionInterpretability/data/ not notebooks/data/
    if 'data_dir="./data"' in content or "data_dir='./data'" in content:
        content = content.replace('data_dir="./data"', 'data_dir="../data"')
        content = content.replace("data_dir='./data'", "data_dir='../data'")
        changes.append("Fixed data path: './data' -> '../data' (project root)")
    
    # Also fix any hardcoded "./data" in load_imagenette calls without explicit data_dir
    # Look for potential issues in the notebook
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    if changes:
        print("✅ Notebook updated:")
        for change in changes:
            print(f"   - {change}")
    else:
        print("ℹ️ No changes needed")


if __name__ == "__main__":
    main()
