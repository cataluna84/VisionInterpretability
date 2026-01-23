#!/usr/bin/env python3
"""Generate detailed cell-by-cell flow analysis."""
import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_2_activation_max.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    cells = notebook["cells"]
    
    print("=" * 100)
    print("SEGMENT 2 NOTEBOOK - CELL FLOW ANALYSIS")
    print("=" * 100)
    print()
    
    # Track section numbers encountered
    sections_order = []
    issues = []
    
    for i, cell in enumerate(cells):
        cell_type = cell["cell_type"]
        source = "".join(cell.get("source", []))
        
        # Extract section info
        section_info = ""
        if cell_type == "markdown":
            # Look for section headers
            lines = source.split("\n")
            for line in lines:
                if line.startswith("## ") and not line.startswith("## Abstract"):
                    # Found a main section
                    section_info = line.strip()
                    sections_order.append((i, line.strip()))
                    break
                elif line.startswith("### "):
                    section_info = line.strip()
                    break
                elif line.startswith("#### "):
                    section_info = line.strip()
                    break
            
            if not section_info:
                section_info = lines[0][:50] if lines else "(empty)"
        else:
            # Code cell - show function name if any
            for line in source.split("\n"):
                if line.strip().startswith("def "):
                    section_info = "def " + line.split("def ")[1].split("(")[0] + "()"
                    break
            if not section_info:
                section_info = "(code block)"
        
        # Check for issues
        prev_cell_type = cells[i-1]["cell_type"] if i > 0 else "markdown"
        if cell_type == "code" and prev_cell_type == "code":
            issues.append(f"Cell {i}: Code cell follows code cell (missing header)")
        
        print(f"{i:2d}. [{cell_type[:4].upper()}] {section_info[:80]}")
    
    print()
    print("=" * 100)
    print("SECTION ORDER ANALYSIS")
    print("=" * 100)
    
    print("\nSections in order of appearance:")
    for i, section in sections_order:
        print(f"  Cell {i:2d}: {section}")
    
    print()
    print("=" * 100)
    print("ISSUES DETECTED")
    print("=" * 100)
    
    # Check section ordering
    for idx, (cell_idx, section) in enumerate(sections_order):
        if "## 5.3" in section or "## 5.4" in section or "### 5.3" in section or "### 5.4" in section:
            # Check if 6 came before
            for prev_idx, (prev_cell_idx, prev_section) in enumerate(sections_order[:idx]):
                if "## 6." in prev_section:
                    issues.append(f"Section order error: '{section}' (cell {cell_idx}) appears after '## 6' (cell {prev_cell_idx})")
        
        if "## 8." in section:
            for prev_idx, (prev_cell_idx, prev_section) in enumerate(sections_order[:idx]):
                if "## 7." not in prev_section:
                    pass  # Check if 7 came before
            # Actually check if 7 comes AFTER
            for later_idx, (later_cell_idx, later_section) in enumerate(sections_order[idx+1:]):
                if "## 7." in later_section:
                    issues.append(f"Section order error: '{section}' (cell {cell_idx}) appears before '## 7' (cell {later_cell_idx})")
    
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ No issues detected")
    
    print()


if __name__ == "__main__":
    main()
