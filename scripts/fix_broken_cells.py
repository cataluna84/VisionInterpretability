#!/usr/bin/env python3
"""Fix broken notebook cells caused by incorrect data_dir insertion."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_1_intro.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    changes = 0
    
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            fixed_source = []
            skip_next = False
            
            for i, line in enumerate(cell["source"]):
                if skip_next:
                    skip_next = False
                    continue
                    
                # Check for broken pattern: function call ending with ), followed by data_dir on next line
                if line.strip().endswith('),'):
                    # Check if next line is orphaned data_dir
                    if i + 1 < len(cell["source"]) and 'data_dir=' in cell["source"][i + 1]:
                        # This is broken - remove the trailing comma and skip the data_dir line
                        line = line.replace('),', ')')
                        skip_next = True
                        changes += 1
                
                # Also clean up any remaining orphaned data_dir lines
                if line.strip().startswith('data_dir='):
                    continue  # Skip orphaned data_dir lines
                
                fixed_source.append(line)
            
            cell["source"] = fixed_source
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Fixed {changes} broken cells in notebook")
    print("   Removed orphaned data_dir= lines that were causing IndentationError")


if __name__ == "__main__":
    main()
