# Vision Interpretability Project - AI Context

> [!IMPORTANT]
> **READ THIS FIRST**: This file contains the architectural context, design philosophy, and development guidelines for the Vision Interpretability project. All changes should align with the patterns defined here.

## Project Overview
This project consists of **two interactive Jupyter notebooks** designed to demystify Convolutional Neural Networks (CNNs) through a "code-first, visual-first" approach.

**Key Goals:**
1.  **Visual Fidelity**: High-resolution visualizations so features are clearly visible
2.  **Interactive Learning**: Every concept has runnable code and visual output
3.  **Transparency**: Implement interpretability methods from scratch with mathematical rigor
4.  **Colab-Ready**: One-click execution on Google Colab

## Architecture & Tech Stack
- **Language**: Python 3.13+
- **Deep Learning Framework**: PyTorch 2.5+
- **Package Manager**: UV
- **Notebooks**: Two segments covering different aspects of interpretability
- **Feature Visualization**: torch-lucent (Segment 2)

## Directory Structure
```
VisionInterpretability/
├── AGENTS.md                                  # YOU ARE HERE
├── README.md                                  # Public documentation
├── pyproject.toml                             # Dependencies (UV)
├── notebooks/
│   ├── cataluna84__segment_1_intro.ipynb      # Segment 1: CNN Basics
│   └── cataluna84__segment_2_activation_max.ipynb  # Segment 2: Feature Viz
├── src/segment_1_intro/                       # Reusable modules (Segment 1 only)
│   ├── __init__.py
│   ├── data.py                                # ImageNette loading
│   ├── models.py                              # SimpleCNN, InceptionV1, training
│   └── visualize.py                           # Grad-CAM, Saliency Maps
├── scripts/
│   ├── update_notebook_colab.py               # Add Colab setup cells
│   └── enhance_notebook_theory.py             # Add formulas & theory
└── docs/                                      # Conceptual documentation
```

## Notebook Segments

### Segment 1: CNN Basics & Interpretability
**File**: `notebooks/cataluna84__segment_1_intro.ipynb`  
**Dependencies**: `segment_1_intro.{data, models, visualize}`

**Topics Covered:**
- Image tensor representation: $(C, H, W)$ format
- Convolution operations with mathematical formulas
- CNN architecture & training on ImageNette
- Feature maps & filter visualization  
- Saliency maps: $S = |\nabla_x y^c|$
- Grad-CAM: $L^c = \text{ReLU}(\sum_k \alpha_k^c A^k)$

**Features:**
✅ "Open in Colab" badge  
✅ Auto-setup cell (clone repo, install deps)  
✅ LaTeX formulas & research references  
✅ PEP-8 Google-style docstrings

### Segment 2: Activation Maximization
**File**: `notebooks/cataluna84__segment_2_activation_max.ipynb`  
**Dependencies**: `torch-lucent` (NO local .py files)

**Topics Covered:**
- Activation maximization: $\mathbf{x}^* = \arg\max_{\mathbf{x}} a_{l,k}(f(\mathbf{x}))$
- Gradient ascent optimization
- FFT-based image parameterization
- Total variation & L2 regularization
- Reproducing Distill.pub Circuits visualizations

**Features:**
✅ Self-contained (no local imports)  
✅ Uses Lucent library  
✅ Complete theory with formulas

## Module Reference (Segment 1 Only)

### `data.py` - Dataset Loading
- `load_imagenette(split, image_size, batch_size)` - Load ImageNette
- `get_imagenette_transforms(image_size, is_train)` - Preprocessing transforms
- `IMAGENETTE_CLASSES` - List of 10 class names

### `models.py` - Models & Training
- `load_inception_v1(pretrained)` - Load GoogLeNet
- `load_simple_cnn(num_classes)` - 3-layer CNN with BatchNorm
- `train_model(model, train_loader, val_loader, epochs)` - Full training loop
- `get_predictions(model, images)` - Get predictions and probabilities

### `visualize.py` - Visualization & Interpretability
- `compute_saliency_map(model, input, target_class)` - Vanilla gradients
- `GradCAM(model, target_layer)` - Grad-CAM class with hooks
- `visualize_feature_maps(activation, num_maps)` - Display activations
- `visualize_filters(layer, num_filters)` - Show learned kernels
- `denormalize_image(tensor)` - Convert to displayable format

## Development Guidelines

### 1. Code Style
- **Type Hints**: All functions use type annotations
- **Docstrings**: Google-style with `Args`, `Returns`, `Example`
- **Modularity**: Segment 1 uses `src/`, Segment 2 is self-contained
- **Raw Strings**: Use `r"""..."""` for docstrings with LaTeX

### 2. Visualization Standards
- **Matplotlib**: Consistent `fig, ax = plt.subplots()` pattern
- **LaTeX**: Use `$...$` for inline math, `$$...$$` for display
- **Color Maps**: `viridis` for heatmaps, `gray` for single-channel

### 3. Dependency Management
- Use `uv add <package>` to add dependencies
- `pyproject.toml` is source of truth
- Segment 2 requires `torch-lucent` in addition to core deps

## Interpretability Concepts

| Concept | Definition | Formula |
|---------|-----------|----------|
| **Filters** | Learned convolutional kernels | $\mathbf{W}^{(l)} \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$ |
| **Feature Maps** | Layer activations | $A^k = \sigma(W^k * X + b^k)$ |
| **Saliency** | Input gradient magnitude | $S = \|\nabla_x y^c\|$ |
| **Grad-CAM** | Weighted activation map | $L^c = \text{ReLU}(\sum_k \alpha_k^c A^k)$ |
| **Activation Max** | Optimal input for neuron | $\mathbf{x}^* = \arg\max_{\mathbf{x}} a_{l,k}(f(\mathbf{x}))$ |
