# Vision Interpretability: Decoding CNNs

A comprehensive, interactive deep dive into how Convolutional Neural Networks (CNNs) "see" the world through **two tutorial notebooks** covering fundamentals and advanced feature visualization.

## 📓 Notebooks

### Segment 1: CNN Basics & Interpretability
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_1_intro.ipynb)

**Topics:**
- Image tensors & convolution mathematics  
- Training a simple CNN on ImageNette
- Filter & feature map visualization
- Saliency maps (vanilla gradients)
- Grad-CAM class activation mapping

**Features:** ✅ Auto-setup for Colab | ✅ LaTeX formulas | ✅ Research references

---

### Segment 2: Activation Maximization
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_2_activation_max.ipynb)

**Topics:**
- Gradient ascent optimization for feature visualization
- Reproducing Distill.pub Circuits research
- FFT vs pixel parameterization
- Total variation & L2 regularization

**Features:** ✅ Uses torch-lucent library | ✅ Self-contained (no local deps) | ✅ Publication-quality visuals

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
Click either badge above → Run all cells. Setup is automatic!

### Option 2: Local Setup

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv)

```bash
# Clone the repository
git clone https://github.com/cataluna84/VisionInterpretability.git
cd VisionInterpretability

# Install dependencies
uv sync

# Start Jupyter
uv run jupyter lab
```

Then open either notebook in `notebooks/`.

## 📁 Project Structure

```
VisionInterpretability/
├── notebooks/
│   ├── cataluna84__segment_1_intro.ipynb           # Part 1: CNN Basics
│   └── cataluna84__segment_2_activation_max.ipynb  # Part 2: Feature Viz
├── src/segment_1_intro/              # Python modules (for Segment 1)
│   ├── data.py       # ImageNette dataset loading
│   ├── models.py     # SimpleCNN, InceptionV1, training
│   └── visualize.py  # Grad-CAM, Saliency Maps, plotting
├── scripts/          # Notebook enhancement scripts
└── pyproject.toml    # Dependencies (UV)
```

## 📦 Python Modules (Segment 1 Only)

### `segment_1_intro.data`
```python
from segment_1_intro import data

train_loader = data.load_imagenette(split="train", batch_size=32)
classes = data.IMAGENETTE_CLASSES  # 10 ImageNet classes
```

### `segment_1_intro.models`
```python
from segment_1_intro import models

model = models.load_simple_cnn(num_classes=10)
history = models.train_model(model, train_loader, val_loader, epochs=5)
```

### `segment_1_intro.visualize`
```python
from segment_1_intro import visualize

# Saliency map
saliency = visualize.compute_saliency_map(model, image, target_class=3)

# Grad-CAM
gradcam = visualize.GradCAM(model, model.conv3)
heatmap = gradcam(image, target_class=3)
```

## 📊 Dependencies

### Core (Both Segments)
- PyTorch >= 2.5.0
- torchvision >= 0.20.0  
- matplotlib >= 3.9.0
- numpy >= 2.0.0

### Segment 1 Specific
- opencv-python >= 4.13.0
- scikit-learn >= 1.5.0
- tqdm >= 4.66.0

### Segment 2 Specific  
- **torch-lucent >= 0.1.8** — Feature visualization library (PyTorch port of Lucid)

## 🎯 What You'll Learn

| Section | Notebook | Key Concepts |
|---------|----------|-------------|
| **Image Representation** | Segment 1 | Tensors $(C, H, W)$, normalization |
| **Convolutions** | Segment 1 | Kernels, stride, padding, formulas |
| **CNN Training** | Segment 1 | SimpleCNN on ImageNette |
| **Feature Maps** | Segment 1 | Layer activations, what CNNs detect |
| **Saliency Maps** | Segment 1 | $S = \|\nabla_x y^c\|$ |
| **Grad-CAM** | Segment 1 | $L^c = \text{ReLU}(\sum_k \alpha_k^c A^k)$ |
| **Activation Max** | Segment 2 | Gradient ascent, FFT parameterization |
| **Feature Viz** | Segment 2 | Reproducing Distill.pub Circuits |

## 📖 References

### Segment 1
- Selvaraju et al., [Grad-CAM](https://arxiv.org/abs/1610.02391), ICCV 2017
- Zeiler & Fergus, [Visualizing CNNs](https://arxiv.org/abs/1311.2901), ECCV 2014

### Segment 2  
- Olah et al., [Feature Visualization](https://distill.pub/2017/feature-visualization/), Distill 2017
- Olah et al., [Circuits](https://distill.pub/2020/circuits/), Distill 2020

## 📜 License

MIT License

