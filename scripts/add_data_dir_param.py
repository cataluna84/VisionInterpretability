#!/usr/bin/env python3
"""Add data_dir parameter to notebook for project root data storage."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_1_intro.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    changes = 0
    
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Find load_imagenette calls without data_dir parameter
            if "load_imagenette(" in source and "data_dir=" not in source:
                # Add data_dir="../data" to each load_imagenette call
                new_source = []
                for line in cell["source"]:
                    if "load_imagenette(" in line and "data_dir=" not in line:
                        # Find the closing parenthesis or newline
                        if line.strip().endswith(")"):
                            # Single line call - insert before closing paren
                            line = line.rstrip(")\n") + ', data_dir="../data")\n'
                        elif line.strip().endswith(","):
                            # Multi-line call - line already has comma
                            new_source.append(line)
                            line = None  # We'll add it on next iteration
                        else:
                            # Need to check next lines
                            pass
                    
                    if line:
                        new_source.append(line)
                
                # For multi-line calls, we need a smarter approach
                # Let's just add a parameter after batch_size
                new_source = []
                for i, line in enumerate(cell["source"]):
                    if 'batch_size=' in line and 'load_imagenette' in "".join(cell["source"][:i+1]):
                        # Add data_dir on next line or same line
                        if line.rstrip().endswith(","):
                            new_source.append(line)
                        else:
                            new_source.append(line.rstrip() + ",\n")
                        # Check if next line exists and isn't closing paren
                        if i + 1 < len(cell["source"]) and ')' not in cell["source"][i+1]:
                            # Insert data_dir before next param
                            new_source.append('    data_dir="../data",\n')
                            changes += 1
                        elif i + 1 < len(cell["source"]) and ')' in cell["source"][i+1]:
                            # Insert before closing paren
                            new_source.append('    data_dir="../data"\n')
                            changes += 1
                    else:
                        new_source.append(line)
                
                if new_source and new_source != cell["source"]:
                    cell["source"] = new_source
    
    # Simpler approach: just do string replacement on the entire notebook JSON
    with open(notebook_path, "w", encoding="utf-8") as f:
        content = json.dumps(notebook, indent=1)
        
        # Find and replace load_imagenette calls
        # Pattern: load_imagenette(...batch_size=32) -> add data_dir
        import re
        
        # Match load_imagenette calls and add data_dir parameter
        def add_data_dir(match):
            call = match.group(0)
            if 'data_dir=' in call:
                return call  # Already has data_dir
            
            # Add data_dir before the closing paren
            # Find last parameter
            if ', variant=' in call:
                return call.replace(', variant=', ', data_dir="../data", variant=')
            elif 'batch_size=32' in call:
                return call.replace('batch_size=32', 'batch_size=32, data_dir="../data"')
            else:
                return call.replace(')', ', data_dir="../data")')
        
        # More targeted: just replace the specific patterns
        replacements = [
            ('load_imagenette(split="train", image_size=128, batch_size=32)',
             'load_imagenette(split="train", image_size=128, batch_size=32, data_dir="../data")'),
            ('load_imagenette(split="validation", image_size=128, batch_size=32)',
             'load_imagenette(split="validation", image_size=128, batch_size=32, data_dir="../data")'),
            ('load_imagenette(split=\\"train\\", image_size=128, batch_size=32)',
             'load_imagenette(split=\\"train\\", image_size=128, batch_size=32, data_dir=\\"../data\\")'),
            ('load_imagenette(split=\\"validation\\", image_size=128, batch_size=32)',
             'load_imagenette(split=\\"validation\\", image_size=128, batch_size=32, data_dir=\\"../data\\")'),
        ]
        
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                changes += 1
        
        f.write(content)
    
    print(f"✅ Updated {changes} load_imagenette() calls with data_dir='../data'")
    print("   Data will now be stored in project root: VisionInterpretability/data/")


if __name__ == "__main__":
    main()
