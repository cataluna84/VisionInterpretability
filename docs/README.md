# Vision Interpretability Documentation

*A complete reference for understanding CNNs through two interactive notebooks*

---

## Project Overview

This project provides **two Jupyter notebooks** for understanding how Convolutional Neural Networks interpret visual information.

### Quick Start

| Notebook | Description | Launch |
|----------|-------------|--------|
| **Segment 1: CNN Basics** | Convolutions, training, Grad-CAM, saliency maps | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_1_intro.ipynb) |
| **Segment 2: Activation Max** | Feature visualization, Distill.pub Circuits | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_2_activation_max.ipynb) |

---

## Part 1: CNN Fundamentals

### Image Representation

Images are tensors: $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$

- $C$ = Channels (3 for RGB)
- $H, W$ = Height, Width in pixels

Normalized with ImageNet stats: $x_{\text{norm}} = (x - \mu) / \sigma$

### Convolution Operation

$$S(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)$$

**Output size**: $o = \lfloor(i + 2p - k)/s\rfloor + 1$

### Feature Hierarchy

| Layer | Detects |
|-------|---------|
| Conv1 | Edges, colors |
| Conv2-3 | Textures, patterns |
| Conv4+ | Object parts |
| Final | Complete objects |

---

## Part 2: Interpretability Methods

### Saliency Maps (Segment 1)

*Which pixels matter?*

$$S = \left| \frac{\partial y^c}{\partial \mathbf{x}} \right|$$

### Grad-CAM (Segment 1)

*Which regions matter?*

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

### Activation Maximization (Segment 2)

*What does a neuron prefer?*

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} a_{l,k}(f(\mathbf{x}))$$

**Update**: $\mathbf{x}_{t+1} = \mathbf{x}_t + \eta \cdot \nabla_{\mathbf{x}} a_{l,k}$

**Regularization**:
- Total Variation: $\mathcal{L}_{\text{TV}} = \sum |x_{i+1,j} - x_{i,j}|$
- L2 penalty: $\mathcal{L}_{L2} = \|\mathbf{x}\|_2^2$

---

## Module Reference (Segment 1 Only)

```python
from segment_1_intro import data, models, visualize

# Data
train_loader = data.load_imagenette(split="train")

# Models
model = models.load_simple_cnn(num_classes=10)
models.train_model(model, train_loader, val_loader, epochs=5)

# Interpretability
saliency = visualize.compute_saliency_map(model, image, target_class=3)
gradcam = visualize.GradCAM(model, model.conv3)
heatmap = gradcam(image, target_class=3)
```

**Segment 2**: Uses `torch-lucent` only (no local imports)

---

## Dependencies

**Both notebooks**: torch, torchvision, matplotlib, numpy  
**Segment 1 only**: opencv-python, scikit-learn, tqdm  
**Segment 2 only**: **torch-lucent**

---

## Changelog (2026-01-22)

- ✅ Added Colab one-click setup to both notebooks
- ✅ Added LaTeX formulas & theory to all markdown cells
- ✅ Created PEP-8 docstrings for all Python modules
- ✅ Fixed package name: `vision_interpret` → `segment_1_intro`
- ✅ Updated AGENTS.md and README.md with Colab badges

---

## References

**Segment 1**:
- Selvaraju et al., "Grad-CAM", ICCV 2017
- Zeiler & Fergus, "Visualizing CNNs", ECCV 2014

**Segment 2**:
- Olah et al., "Feature Visualization", Distill 2017
- Olah et al., "Circuits", Distill 2020
