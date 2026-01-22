#!/usr/bin/env python3
"""Script to enhance notebook markdown cells with theory, formulas, and descriptions.

This script updates the Jupyter notebook's markdown cells with comprehensive
theoretical explanations, mathematical formulas, and research-oriented content
for each section of the Vision Interpretability tutorial.

Usage:
    python scripts/enhance_notebook_theory.py
"""
import json
from pathlib import Path


def main():
    """Enhance notebook markdown cells with theory and formulas."""
    notebook_path = Path(__file__).parent.parent / "notebooks" / "cataluna84__segment_1_intro.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    # Enhanced markdown content for each section
    enhanced_markdowns = {
        # Section 1: Image Representation (find by checking first line of source)
        "## 1. How Images Are Represented": [
            "## 1. How Images Are Represented\n",
            "\n",
            "### Theoretical Background\n",
            "\n",
            "Digital images are represented as **multi-dimensional arrays** (tensors) where each element corresponds to a pixel intensity. In deep learning frameworks like PyTorch, images follow the **CHW convention**:\n",
            "\n",
            "$$\\mathbf{X} \\in \\mathbb{R}^{C \\times H \\times W}$$\n",
            "\n",
            "where:\n",
            "- $C$ = Number of color channels (3 for RGB: Red, Green, Blue)\n",
            "- $H$ = Height in pixels\n",
            "- $W$ = Width in pixels\n",
            "\n",
            "### Pixel Value Representation\n",
            "\n",
            "| Format | Range | Description |\n",
            "|--------|-------|-------------|\n",
            "| Raw (uint8) | [0, 255] | Original image format |\n",
            "| Normalized | [0, 1] | After `ToTensor()` transform |\n",
            "| Standardized | ~[-2.5, 2.5] | After ImageNet normalization |\n",
            "\n",
            "**ImageNet normalization** is standard for pretrained models:\n",
            "\n",
            "$$x_{\\text{norm}} = \\frac{x - \\mu}{\\sigma}$$\n",
            "\n",
            "where $\\mu = [0.485, 0.456, 0.406]$ and $\\sigma = [0.229, 0.224, 0.225]$ (per channel).\n",
            "\n",
            "### Why This Matters\n",
            "\n",
            "Understanding tensor representations is fundamental because:\n",
            "1. **Convolutions** operate on spatial dimensions (H, W)\n",
            "2. **Batch processing** adds a 4th dimension: $(N, C, H, W)$\n",
            "3. **Interpretability methods** compute gradients with respect to these pixel values\n"
        ],
        
        # Section 2: Convolution Operations
        "## 2. Convolution Operations Deep Dive": [
            "## 2. Convolution Operations Deep Dive\n",
            "\n",
            "### Mathematical Definition\n",
            "\n",
            "A **2D convolution** (technically cross-correlation in deep learning) applies a kernel $\\mathbf{K}$ to an input image $\\mathbf{I}$ to produce an output feature map $\\mathbf{S}$:\n",
            "\n",
            "$$S(i, j) = (\\mathbf{I} * \\mathbf{K})(i, j) = \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1} I(i+m, j+n) \\cdot K(m, n)$$\n",
            "\n",
            "where $M \\times N$ is the kernel size.\n",
            "\n",
            "### Output Size Calculation\n",
            "\n",
            "Given input size $i$, kernel size $k$, stride $s$, and padding $p$:\n",
            "\n",
            "$$o = \\left\\lfloor \\frac{i + 2p - k}{s} \\right\\rfloor + 1$$\n",
            "\n",
            "### Common Kernels and Their Effects\n",
            "\n",
            "| Kernel | Purpose | Mathematical Property |\n",
            "|--------|---------|----------------------|\n",
            "| **Sobel** | Edge detection | Approximates gradient $\\nabla I$ |\n",
            "| **Laplacian** | All edges | Second derivative $\\nabla^2 I$ |\n",
            "| **Gaussian** | Blur/smoothing | Low-pass filter |\n",
            "| **Sharpen** | Enhance edges | High-pass amplification |\n",
            "\n",
            "### Sobel Operators\n",
            "\n",
            "Detect horizontal and vertical edges:\n",
            "\n",
            "$$G_x = \\begin{bmatrix} -1 & 0 & +1 \\\\ -2 & 0 & +2 \\\\ -1 & 0 & +1 \\end{bmatrix}, \\quad G_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ +1 & +2 & +1 \\end{bmatrix}$$\n",
            "\n",
            "Edge magnitude: $G = \\sqrt{G_x^2 + G_y^2}$\n",
            "\n",
            "### Key Insight\n",
            "\n",
            "> In CNNs, kernels are **learned** through backpropagation rather than hand-designed, allowing the network to discover optimal feature detectors for the task.\n"
        ],
        
        # Section 3: Building & Training a CNN
        "## 3. Building & Training a CNN on ImageNette": [
            "## 3. Building & Training a CNN on ImageNette\n",
            "\n",
            "### CNN Architecture Theory\n",
            "\n",
            "A Convolutional Neural Network learns a hierarchy of features through stacked layers:\n",
            "\n",
            "$$\\mathbf{h}^{(l)} = \\sigma\\left(\\mathbf{W}^{(l)} * \\mathbf{h}^{(l-1)} + \\mathbf{b}^{(l)}\\right)$$\n",
            "\n",
            "where:\n",
            "- $\\mathbf{h}^{(l)}$ = Activation at layer $l$\n",
            "- $\\mathbf{W}^{(l)}$ = Learnable convolution kernels\n",
            "- $\\sigma$ = Non-linear activation (e.g., ReLU)\n",
            "- $*$ = Convolution operation\n",
            "\n",
            "### The Feature Hierarchy\n",
            "\n",
            "Research by Zeiler & Fergus (2014) and Olah et al. (2017) shows:\n",
            "\n",
            "| Layer Depth | Features Learned | Examples |\n",
            "|-------------|-----------------|----------|\n",
            "| Layer 1 | Low-level | Edges, colors, gradients |\n",
            "| Layer 2-3 | Mid-level | Textures, patterns, curves |\n",
            "| Layer 4+ | High-level | Object parts, faces, wheels |\n",
            "| Final | Semantic | Full objects, scenes |\n",
            "\n",
            "### ImageNette Dataset\n",
            "\n",
            "A subset of ImageNet with 10 easily classifiable classes:\n",
            "\n",
            "| Class | ImageNet ID | Category |\n",
            "|-------|------------|----------|\n",
            "| Tench | n01440764 | Fish |\n",
            "| English Springer | n02102040 | Dog |\n",
            "| Cassette Player | n02979186 | Electronics |\n",
            "| Chain Saw | n03000684 | Tool |\n",
            "| Church | n03028079 | Building |\n",
            "| French Horn | n03394916 | Instrument |\n",
            "| Garbage Truck | n03417042 | Vehicle |\n",
            "| Gas Pump | n03425413 | Object |\n",
            "| Golf Ball | n03445777 | Sports |\n",
            "| Parachute | n03888257 | Equipment |\n",
            "\n",
            "### Training Objective\n",
            "\n",
            "We minimize the **Cross-Entropy Loss**:\n",
            "\n",
            "$$\\mathcal{L} = -\\sum_{c=1}^{C} y_c \\log(\\hat{y}_c)$$\n",
            "\n",
            "where $y_c$ is the one-hot target and $\\hat{y}_c$ is the predicted probability for class $c$.\n"
        ],
        
        # Section: Training note
        "### Training the Model": [
            "### Training the Model\n",
            "\n",
            "Training uses the **Adam optimizer**, which adapts learning rates per-parameter:\n",
            "\n",
            "$$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$\n",
            "\n",
            "where $\\hat{m}_t$ and $\\hat{v}_t$ are bias-corrected first and second moment estimates.\n",
            "\n",
            "⚠️ **Note**: Training takes a few minutes. For a quick demo, we train for just 3 epochs.\n",
            "In practice, you'd train for 20+ epochs to achieve better accuracy.\n"
        ],
        
        # Section 4: Feature Maps
        "## 4. Feature Map Visualization": [
            "## 4. Feature Map Visualization\n",
            "\n",
            "### What Are Feature Maps?\n",
            "\n",
            "**Feature maps** (activation maps) are the outputs of convolutional layers. For a layer with $K$ filters:\n",
            "\n",
            "$$\\mathbf{A}^k = \\sigma(\\mathbf{W}^k * \\mathbf{X} + b^k), \\quad k = 1, \\ldots, K$$\n",
            "\n",
            "Each feature map $\\mathbf{A}^k$ highlights regions where specific patterns are detected.\n",
            "\n",
            "### Interpretation by Layer\n",
            "\n",
            "| Layer | Feature Map Shows | Spatial Resolution |\n",
            "|-------|------------------|-------------------|\n",
            "| `conv1` | Edges, color blobs | High (close to input) |\n",
            "| `conv2` | Texture patterns | Medium |\n",
            "| `conv3` | Object parts | Low (more abstract) |\n",
            "\n",
            "### Why Visualize Feature Maps?\n",
            "\n",
            "1. **Debugging**: Verify the network is learning meaningful features\n",
            "2. **Interpretation**: Understand *where* the network focuses attention\n",
            "3. **Research**: Analyze the feature hierarchy in novel architectures\n",
            "\n",
            "> *\"Early layers detect low-level features (edges, colors, textures). Later layers detect high-level concepts (object parts, shapes).\"* — Zeiler & Fergus, 2014\n"
        ],
        
        # Section 5: Filters
        "## 5. Filter (Kernel) Visualization": [
            "## 5. Filter (Kernel) Visualization\n",
            "\n",
            "### Theory: What Filters Learn\n",
            "\n",
            "Convolutional filters are the **learned parameters** $\\mathbf{W} \\in \\mathbb{R}^{C_{out} \\times C_{in} \\times k \\times k}$.\n",
            "\n",
            "For the **first layer** (RGB input with $C_{in}=3$), each filter can be visualized as a small RGB image showing what pattern the filter responds to.\n",
            "\n",
            "### Trained vs. Random Filters\n",
            "\n",
            "| Filter Type | Appearance | Interpretation |\n",
            "|-------------|-----------|----------------|\n",
            "| **Random (untrained)** | Noise-like | No meaningful patterns |\n",
            "| **Trained** | Structure | Edge detectors, color selectivity |\n",
            "\n",
            "### Common First-Layer Filter Types\n",
            "\n",
            "After training, first-layer filters typically include:\n",
            "- **Gabor-like filters**: Oriented edges at various angles\n",
            "- **Color-opponent filters**: Red-green, blue-yellow channels\n",
            "- **Gradient filters**: Similar to Sobel operators\n",
            "\n",
            "This emergence is consistent across different CNN architectures and datasets.\n"
        ],
        
        # Section 6: Saliency Maps
        "## 6. Gradient-Based Interpretability: Saliency Maps": [
            "## 6. Gradient-Based Interpretability: Saliency Maps\n",
            "\n",
            "### Theoretical Foundation\n",
            "\n",
            "**Saliency maps** (also called *vanilla gradients* or *input attribution*) identify which input pixels most influence the model's prediction by computing:\n",
            "\n",
            "$$S = \\left| \\frac{\\partial y^c}{\\partial \\mathbf{x}} \\right|$$\n",
            "\n",
            "where:\n",
            "- $y^c$ = Pre-softmax score for target class $c$\n",
            "- $\\mathbf{x}$ = Input image pixels\n",
            "- $S$ = Saliency map (same spatial dimensions as input)\n",
            "\n",
            "### Algorithm\n",
            "\n",
            "1. **Forward pass**: Compute prediction $y^c$\n",
            "2. **Backward pass**: Compute gradient $\\nabla_{\\mathbf{x}} y^c$\n",
            "3. **Aggregate channels**: Take max or mean across RGB\n",
            "4. **Absolute value**: $S = |\\nabla_{\\mathbf{x}} y^c|$\n",
            "\n",
            "### Mathematical Interpretation\n",
            "\n",
            "The gradient represents the **local sensitivity** of the output to input perturbations:\n",
            "\n",
            "$$\\Delta y^c \\approx \\nabla_{\\mathbf{x}} y^c \\cdot \\Delta \\mathbf{x}$$\n",
            "\n",
            "High gradient magnitude → Small input change causes large output change → **Important pixel**.\n",
            "\n",
            "### Limitations\n",
            "\n",
            "- **Noisy**: Gradients can be high-frequency and visually noisy\n",
            "- **Saturation**: ReLU gradients are zero where activations are negative\n",
            "- **Local**: Only captures first-order effects\n",
            "\n",
            "Advanced methods like **Integrated Gradients** and **SmoothGrad** address these issues.\n"
        ],
        
        # Section 7: Grad-CAM
        "## 7. Class Activation Mapping (Grad-CAM)": [
            "## 7. Class Activation Mapping (Grad-CAM)\n",
            "\n",
            "### Theoretical Foundation\n",
            "\n",
            "**Gradient-weighted Class Activation Mapping (Grad-CAM)** produces coarse localization maps highlighting important regions for a target class $c$. Unlike saliency maps, Grad-CAM operates on **feature maps** rather than input pixels.\n",
            "\n",
            "### The Grad-CAM Formula\n",
            "\n",
            "**Step 1: Compute importance weights**\n",
            "\n",
            "$$\\alpha_k^c = \\underbrace{\\frac{1}{Z} \\sum_i \\sum_j}_{\\text{global avg pool}} \\frac{\\partial y^c}{\\partial A_{ij}^k}$$\n",
            "\n",
            "where:\n",
            "- $A^k$ = Feature map $k$ from the target convolutional layer\n",
            "- $y^c$ = Score for class $c$ (before softmax)\n",
            "- $Z$ = Number of pixels in the feature map\n",
            "\n",
            "**Step 2: Weighted combination with ReLU**\n",
            "\n",
            "$$L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)$$\n",
            "\n",
            "The ReLU ensures we only visualize features with **positive influence** on the target class.\n",
            "\n",
            "### Why Grad-CAM Works\n",
            "\n",
            "| Property | Saliency Maps | Grad-CAM |\n",
            "|----------|--------------|----------|\n",
            "| Resolution | Pixel-level (high) | Feature-level (coarse) |\n",
            "| Noise | High-frequency noise | Smooth heatmap |\n",
            "| Interpretation | Which pixels matter | Which *regions* matter |\n",
            "| Class-discriminative | Yes | Yes |\n",
            "\n",
            "### Reference\n",
            "\n",
            "> Selvaraju et al., \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\", ICCV 2017.\n"
        ],
        
        # Section 8: InceptionV1
        "## 8. Deep Interpretability with InceptionV1": [
            "## 8. Deep Interpretability with InceptionV1\n",
            "\n",
            "### InceptionV1 (GoogLeNet) Architecture\n",
            "\n",
            "InceptionV1 introduced the **Inception module**, which processes input through parallel convolutions of different sizes:\n",
            "\n",
            "```\n",
            "        Input\n",
            "          │\n",
            "    ┌─────┼─────┬─────┐\n",
            "    │     │     │     │\n",
            "   1×1   3×3   5×5  MaxPool\n",
            "    │     │     │     │\n",
            "    └─────┴─────┴─────┘\n",
            "          │\n",
            "     Concatenate\n",
            "```\n",
            "\n",
            "### Why InceptionV1 for Interpretability?\n",
            "\n",
            "1. **Multi-scale features**: Parallel branches capture patterns at different spatial scales\n",
            "2. **Rich representations**: Concatenation creates diverse feature maps\n",
            "3. **Distill.pub research**: Extensively studied for feature visualization (Olah et al., 2017)\n",
            "4. **Pretrained on ImageNet**: 1000 classes, robust feature learning\n",
            "\n",
            "### Network Depth and Interpretability\n",
            "\n",
            "| Layer | Example Content | Interpretability |\n",
            "|-------|----------------|------------------|\n",
            "| `conv1` | Gabor-like edges | Easy to interpret |\n",
            "| `inception3a` | Texture combinations | Moderate |\n",
            "| `inception4a` | Object parts | Harder to interpret |\n",
            "| `inception5b` | Full objects | Most abstract |\n",
            "\n",
            "### Key Insight\n",
            "\n",
            "> The deeper the layer, the more **class-specific** and **abstract** the features become. Grad-CAM on `inception5b` highlights semantic regions relevant to the predicted class.\n"
        ],
        
        # Section 9: Summary
        "## 9. Summary": [
            "## 9. Summary\n",
            "\n",
            "### Methods Covered\n",
            "\n",
            "| Technique | Question Answered | Mathematical Core | Output |\n",
            "|-----------|------------------|-------------------|--------|\n",
            "| **Feature Maps** | Where did the model detect patterns? | $A^k = \\sigma(W^k * X + b^k)$ | Activation grids |\n",
            "| **Filter Visualization** | What patterns is it looking for? | Visualize $W^k$ directly | Kernel weights as images |\n",
            "| **Saliency Maps** | Which pixels influenced the prediction? | $S = \\|\\nabla_x y^c\\|$ | Gradient-based heatmap |\n",
            "| **Grad-CAM** | Which *regions* were important? | $L^c = \\text{ReLU}(\\sum_k \\alpha_k^c A^k)$ | Coarse localization heatmap |\n",
            "\n",
            "### The Interpretability Stack\n",
            "\n",
            "```\n",
            "┌──────────────────────────────────┐\n",
            "│     High-Level Explanations      │  ← Grad-CAM, CAM\n",
            "├──────────────────────────────────┤\n",
            "│       Feature Attribution        │  ← Saliency, Integrated Gradients\n",
            "├──────────────────────────────────┤\n",
            "│     Internal Representations     │  ← Feature maps, filter visualization\n",
            "├──────────────────────────────────┤\n",
            "│         Model Parameters         │  ← Weight analysis\n",
            "└──────────────────────────────────┘\n",
            "```\n",
            "\n",
            "### Practical Applications\n",
            "\n",
            "1. **Model debugging**: Find spurious correlations (e.g., model focusing on background)\n",
            "2. **Trust building**: Show users *why* a medical AI made a diagnosis\n",
            "3. **Research**: Understand how CNNs develop hierarchical representations\n",
            "4. **Adversarial robustness**: Detect when models rely on fragile features\n",
            "\n",
            "### Further Reading\n",
            "\n",
            "- [Distill.pub Feature Visualization](https://distill.pub/2017/feature-visualization/)\n",
            "- [Grad-CAM Paper (ICCV 2017)](https://arxiv.org/abs/1610.02391)\n",
            "- [Captum Library](https://captum.ai/) — PyTorch interpretability toolkit\n"
        ]
    }
    
    # Apply enhancements
    cells_updated = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            source_text = "".join(cell["source"])
            for key, new_content in enhanced_markdowns.items():
                if source_text.strip().startswith(key):
                    cell["source"] = new_content
                    cells_updated += 1
                    print(f"  ✓ Enhanced: {key[:50]}...")
                    break
    
    # Write the updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"\n✅ Enhanced {cells_updated} markdown cells with theory and formulas")
    print(f"   Output: {notebook_path}")


if __name__ == "__main__":
    main()
