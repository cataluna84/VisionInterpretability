# Vision Interpretability Documentation

*A complete reference for understanding CNNs through four interactive notebooks*

---

## Project Overview

This project provides **four main Jupyter notebooks** (plus 8 Lucent tutorials) for understanding how Convolutional Neural Networks interpret visual information.

### Quick Start

| Notebook | Description | Launch |
|----------|-------------|--------|
| **Segment 1: CNN Basics** | Convolutions, training, Grad-CAM, saliency maps | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_1_intro.ipynb) |
| **Segment 2: Activation Max** | Feature visualization, Distill.pub Circuits | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_2_activation_max.ipynb) |
| **Segment 3: Dataset Examples** | Activation spectrum, Distill.pub layout | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_3_dataset_images.ipynb) |
| **Segment 3b: Faccent** | Faccent optimization techniques | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_3_faccent.ipynb) |

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

### Dataset Examples (Segment 3)

*What dataset images activate a neuron?*

Shows 6 categories per neuron:
- Negative optimized (gradient descent)
- Minimum activation examples
- Slightly negative (near threshold)
- Slightly positive (near threshold)
- Maximum activation examples
- Positive optimized (gradient ascent)

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

**Segment 3**:
```python
from segment_3_dataset_images import (
    ActivationSpectrumTrackerV2,
    FeatureOptimizer,
    plot_neuron_spectrum_distill,
)

tracker = ActivationSpectrumTrackerV2(num_neurons=10)
optimizer = FeatureOptimizer(model)
fig = plot_neuron_spectrum_distill(...)
```

---

## Dependencies

**Both notebooks**: torch, torchvision, matplotlib, numpy  
**Segment 1 only**: opencv-python, scikit-learn, tqdm  
**Segment 2 only**: **torch-lucent**
**Segment 3 only**: **torch-lucent**, **wandb**

---

## Changelog (2026-01-30)

- ✅ Added Segment 3b: Faccent optimization notebook
- ✅ Added Lucent tutorial notebooks (8 notebooks)
- ✅ Added `faccent/` library to segment_3_dataset_images module
- ✅ Updated all documentation with accurate directory structures

### Previous (2026-01-29)

- ✅ Added Segment 3: Dataset Examples & Activation Spectrum
- ✅ Added `segment_3_dataset_images` module with `visualization.py`
- ✅ Created Distill.pub style 6-column visualization
- ✅ Added negative optimization method to FeatureOptimizer

### Previous (2026-01-22)

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
