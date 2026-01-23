#!/usr/bin/env python3
"""
Phase 3: Complete remaining section updates, add Conclusion and References.

Updates remaining sections and adds final sections 6-7.
"""
import json
from pathlib import Path


def create_conclusion_cell() -> dict:
    """Create Conclusion section."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## 7. Conclusion\n",
            "\n",
            "### 7.1 Summary of Findings\n",
            "\n",
            "This work successfully demonstrated **activation maximization** as a powerful technique\n",
            "for visualizing and interpreting the learned representations of deep neural networks.\n",
            "Through systematic analysis of 10 neurons in InceptionV1's `mixed4a` layer, we observed\n",
            "that individual neurons develop highly selective responses to specific visual features,\n",
            "including edges, textures, curves, and object parts.\n",
            "\n",
            "**Key findings:**\n",
            "\n",
            "1. **Interpretable features**: Neurons learn semantically meaningful, visually coherent patterns\n",
            "2. **Compositional hierarchies**: Early layers detect primitives; later layers combine them into complex features\n",
            "3. **Biological correspondence**: Learned representations mirror primate visual cortex (V1 → V2/V4 → IT)\n",
            "4. **Reproducibility**: Results consistent with Distill.pub Circuits research\n",
            "\n",
            "### 7.2 Implications for Interpretability Research\n",
            "\n",
            "The success of activation maximization in revealing interpretable features suggests that\n",
            "deep networks develop *natural* internal representations rather than arbitrary encodings.\n",
            "This finding has several implications:\n",
            "\n",
            "- **Model transparency**: Feature visualization can aid in understanding model decisions\n",
            "- **Debugging**: Visual inspection of neuron preferences can reveal dataset biases\n",
            "- **Architecture design**: Insights from feature hierarchies can inform network design\n",
            "- **Mechanistic interpretability**: Understanding *how* networks solve tasks, not just *what* they compute\n",
            "\n",
            "### 7.3 Limitations and Future Work\n",
            "\n",
            "While activation maximization provides valuable insights, several limitations warrant consideration:\n",
            "\n",
            "**Limitations:**\n",
            "- **Polysemanticity**: Individual neurons may respond to multiple unrelated concepts\n",
            "- **Context-dependence**: Isolated neuron visualizations miss interactions between neurons\n",
            "- **Optimization artifacts**: High-frequency patterns may not reflect natural image statistics\n",
            "\n",
            "**Future directions:**\n",
            "- Extend analysis to additional layers (mixed3a/b, mixed5a/b) for complete hierarchy\n",
            "- Investigate neuron interactions using circuit analysis methods\n",
            "- Compare features across different architectures (ResNet, Vision Transformers)\n",
            "- Apply techniques to domain-specific models (medical imaging, satellite imagery)\n",
            "\n",
            "### 7.4 Concluding Remarks\n",
            "\n",
            "Activation maximization bridges the gap between the remarkable capabilities of deep learning\n",
            "and our understanding of how these systems achieve their performance. By making the \"black box\"\n",
            "more transparent, we enable safer deployment, more effective debugging, and deeper scientific\n",
            "insights into both artificial and biological vision systems.\n",
            "\n",
            "---\n"
        ]
    }


def create_references_cell() -> dict:
    """Create References section."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## References\n",
            "\n",
            "### Primary Literature\n",
            "\n",
            "1. **Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S.** (2020).  \n",
            "   *Zoom In: An Introduction to Circuits.*  \n",
            "   Distill. https://distill.pub/2020/circuits/zoom-in/\n",
            "\n",
            "2. **Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S.** (2020).  \n",
            "   *An Overview of Early Vision in InceptionV1.*  \n",
            "   Distill. https://distill.pub/2020/circuits/early-vision/\n",
            "\n",
            "3. **Olah, C., Satyanarayan, A., Johnson, I., Carter, S., Schubert, L., Ye, K., & Mordvintsev, A.** (2018).  \n",
            "   *The Building Blocks of Interpretability.*  \n",
            "   Distill. https://distill.pub/2018/building-blocks/\n",
            "\n",
            "4. **Mordvintsev, A., Pezzotti, N., Schubert, L., & Olah, C.** (2018).  \n",
            "   *Differentiable Image Parameterizations.*  \n",
            "   Distill. https://distill.pub/2018/differentiable-parameterizations/\n",
            "\n",
            "5. **Olah, C., Mordvintsev, A., & Schubert, L.** (2017).  \n",
            "   *Feature Visualization.*  \n",
            "   Distill. https://distill.pub/2017/feature-visualization/\n",
            "\n",
            "### Foundational Works\n",
            "\n",
            "6. **Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A.** (2015).  \n",
            "   *Going deeper with convolutions.*  \n",
            "   Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1-9.\n",
            "\n",
            "7. **Erhan, D., Bengio, Y., Courville, A., & Vincent, P.** (2009).  \n",
            "   *Visualizing higher-layer features of a deep network.*  \n",
            "   University of Montreal, Technical Report 1341.\n",
            "\n",
            "8. **Simonyan, K., Vedaldi, A., & Zisserman, A.** (2013).  \n",
            "   *Deep inside convolutional networks: Visualising image classification models and saliency maps.*  \n",
            "   arXiv preprint arXiv:1312.6034.\n",
            "\n",
            "9. **Zeiler, M. D., & Fergus, R.** (2014).  \n",
            "   *Visualizing and understanding convolutional networks.*  \n",
            "   European Conference on Computer Vision (ECCV), 818-833.\n",
            "\n",
            "### Software Libraries\n",
            "\n",
            "10. **Lucent Library** (PyTorch port of Lucid)  \n",
            "    https://github.com/greentfrapp/lucent\n",
            "\n",
            "11. **PyTorch Framework**  \n",
            "    Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An imperative style, high-performance deep learning library.*  \n",
            "    Advances in Neural Information Processing Systems 32, 8024-8035.\n",
            "\n",
            "---\n",
            "\n",
            "*End of Notebook*\n"
        ]
    }


def main():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_2_activation_max.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    cells = notebook["cells"]
    changes = []
    
    # Update remaining section headers
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown":
            source = "".join(cell.get("source", []))
            
            # Update "## 5. Analysis and Interpretation" → "## 6. Results & Analysis"
            if "## 5. Analysis and Interpretation" in source:
                new_source = source.replace(
                    "## 5. Analysis and Interpretation",
                    "## 6. Results & Analysis"
                )
                new_source = new_source.replace(
                    "### What Do These Neurons Detect?",
                    "### 6.1 Feature Interpretation"
                )
                new_source = new_source.replace(
                    "Looking at the generated visualizations, we can interpret what each\nneuron has learned to detect.",
                    "Analysis of the generated visualizations reveals the specific visual patterns\neach neuron has learned to detect."
                )
                cell["source"] = new_source.split("\n")
                cell["source"] = [line + "\n" if i < len(cell["source"])-1 else line for i, line in enumerate(cell["source"])]
                changes.append(f"Cell {i}: Updated '## 5. Analysis' → '## 6. Results & Analysis'")
            
            # Update "## 6. Advanced: Custom Optimization" → "### 5.3 Advanced Optimization Techniques"
            elif "## 6. Advanced: Custom Optimization Parameters" in source:
                new_source = source.replace(
                    "## 6. Advanced: Custom Optimization Parameters",
                    "### 5.3 Advanced Optimization Techniques"
                )
                new_source = new_source.replace(
                    "We can customize the optimization process to generate different styles\nof visualizations.",
                    "The optimization process can be customized through various parameters\nto generate visualizations with different characteristics."
                )
                cell["source"] = new_source.split("\n")
                cell["source"] = [line + "\n" if i < len(cell["source"])-1 else line for i, line in enumerate(cell["source"])]
                changes.append(f"Cell {i}: Updated '## 6. Advanced' → '### 5.3'")
            
            # Update subsections under Advanced
            elif "### Image Parameterization" in source and "FFT" in source:
                new_source = source.replace("### Image Parameterization", "#### Image Parameterization")
                cell["source"] = new_source.split("\n")
                cell["source"] = [line + "\n" if i < len(cell["source"])-1 else line for i, line in enumerate(cell["source"])]
                changes.append(f"Cell {i}: Updated to #### subsection")
            
            elif "### Transformations" in source and "robustness" in source:
                new_source = source.replace("### Transformations", "#### Transformations")
                cell["source"] = new_source.split("\n")
                cell["source"] = [line + "\n" if i < len(cell["source"])-1 else line for i, line in enumerate(cell["source"])]
                changes.append(f"Cell {i}: Updated to #### subsection")
            
            # Remove/integrate "## 7. High-Resolution Visualization" - part of 5.3
            elif "## 7. High-Resolution Visualization" in source:
                new_source = source.replace(
                    "## 7. High-Resolution Visualization",
                    "#### High-Resolution Visualization"
                )
                new_source = new_source.replace(
                    "For publication-quality images, we can generate higher resolution\nvisualizations with more optimization steps.",
                    "For publication-quality results, higher resolution visualizations\nwith extended optimization can be generated."
                )
                cell["source"] = new_source.split("\n")
                cell["source"] = [line + "\n" if i < len(cell["source"])-1 else line for i, line in enumerate(cell["source"])]
                changes.append(f"Cell {i}: Downgraded '## 7.' → '####' (part of 5.3)")
            
            # Renumber "## 9. Interactive Feature Circuit" but keep it after exper which become 5.4
            elif "## 9. Interactive Feature Circuit Visualization" in source:
                new_source = source.replace(
                    "## 9. Interactive Feature Circuit Visualization",
                    "### 5.4 Interactive Feature Circuit Visualization"
                )
                cell["source"] = new_source.split("\n")
                cell["source"] = [line + "\n" if i < len(cell["source"])-1 else line for i, line in enumerate(cell["source"])]
                changes.append(f"Cell {i}: Updated '## 9.' → '### 5.4'")
            
  # Update subsections under Circuit Viz
            elif "### Theoretical Foundation: Compositional Feature Hierarchies" in source:
                new_source = source.replace(
                    "### Theoretical Foundation: Compositional Feature Hierarchies",
                    "#### Theoretical Foundation"
                )
                cell["source"] = new_source.split("\n")
                cell["source"] = [line + "\n" if i < len(cell["source"])-1 else line for i, line in enumerate(cell["source"])]
                changes.append(f"Cell {i}: Updated to #### subsection")
            
            elif "#### Mathematical Framework" in source and "Polysemantic" in source:
                # Keep as is - already ####
                pass
            
            elif "### Feature Hierarchy in InceptionV1" in source and "Receptive Field" in source:
                new_source = source.replace(
                    "### Feature Hierarchy in InceptionV1",
                    "#### Feature Hierarchy"
                )
                cell["source"] = new_source.split("\n")
                cell["source"] = [line + "\n" if i < len(cell["source"])-1 else line for i, line in enumerate(cell["source"])]
                changes.append(f"Cell {i}: Updated to #### subsection")
    
    # Find the end of the notebook (before or after Summary section if it exists)
    # Insert Conclusion and References before the last cell
    insert_pos = len(cells)
    
    conclusion_cell = create_conclusion_cell()
    references_cell = create_references_cell()
    
    cells.insert(insert_pos, conclusion_cell)
    cells.insert(insert_pos + 1, references_cell)
    changes.append(f"Added Conclusion section at position {insert_pos}")
    changes.append(f"Added References section at position {insert_pos + 1}")
    
    # Save updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Completed remaining section updates")
    print(f"   Made {len(changes)} changes:")
    for change in changes:
        print(f"   - {change}")
    print(f"\n✅ Total cells in notebook: {len(cells)}")


if __name__ == "__main__":
    main()
