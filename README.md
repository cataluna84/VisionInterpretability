# Vision Interpretability: Decoding CNNs

A comprehensive, interactive deep dive into how Convolutional Neural Networks (CNNs) "see" the world through **three tutorial notebooks** covering fundamentals and advanced feature visualization.

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

---

### Segment 3: Dataset Examples & Activation Spectrum
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_3_dataset_images.ipynb)

**Topics:**
- Finding dataset examples across activation spectrum
- Minimum, slightly negative, slightly positive, maximum examples
- Distill.pub style 6-column visualization layout

**Features:** ✅ Streaming ImageNet | ✅ W&B logging | ✅ Publication-quality Distill.pub visuals

---

### Segment 3b: Faccent Optimization
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_3_faccent.ipynb)

**Topics:**
- Feature visualization with Faccent library
- Advanced optimization techniques
- Class activation mapping (CAM)

**Features:** ✅ Faccent library | ✅ Advanced parametrization | ✅ CAM visualization

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
│   ├── cataluna84__segment_2_activation_max.ipynb  # Part 2: Feature Viz
│   ├── cataluna84__segment_3_dataset_images.ipynb  # Part 3: Dataset Examples
│   ├── cataluna84__segment_3_faccent.ipynb         # Part 3b: Faccent Optimization
│   ├── lucent/                     # Lucent tutorial notebooks
│   │   ├── tutorial.ipynb          # Getting started with Lucent
│   │   ├── activation_grids.ipynb  # Activation grid visualizations
│   │   ├── diversity.ipynb         # Feature diversity analysis
│   │   ├── feature_inversion.ipynb # Feature inversion techniques
│   │   ├── GAN_parametrization.ipynb   # GAN-based parametrization
│   │   ├── neuron_interaction.ipynb    # Neuron interaction analysis
│   │   ├── style_transfer.ipynb    # Neural style transfer
│   │   └── modelzoo.ipynb          # Model zoo examples
│   ├── results/                    # Notebook output artifacts
│   └── wandb/                      # W&B experiment logs
├── src/segment_1_intro/            # Python modules (for Segment 1)
│   ├── __init__.py
│   ├── data.py       # ImageNette dataset loading
│   ├── models.py     # SimpleCNN, InceptionV1, training
│   └── visualize.py  # Grad-CAM, Saliency Maps, plotting
├── src/segment_3_dataset_images/   # Python modules (for Segment 3)
│   ├── __init__.py
│   ├── activation_pipeline.py  # Activation extraction, spectrum tracking
│   ├── visualization.py        # Distill.pub style plotting
│   └── faccent/                # Feature visualization library
│       ├── cam.py              # Class activation mapping
│       ├── mask.py             # Masking utilities
│       ├── objectives.py       # Optimization objectives
│       ├── param.py            # Image parameterization
│       ├── render.py           # Rendering engine
│       ├── transform.py        # Image transforms
│       ├── utils.py            # Utility functions
│       └── modelzoo/           # Pretrained model loaders
│           └── inceptionv1/    # InceptionV1 model
├── scripts/                    # Notebook enhancement scripts
│   ├── add_circuit_visualization.py
│   ├── add_colab_support_seg3.py
│   ├── add_data_dir_param.py
│   ├── add_device_definition.py
│   ├── add_performance_docs.py
│   ├── add_plotly_setup.py
│   ├── add_setup_cell_seg2.py
│   ├── add_wandb_chart.py
│   ├── analyze_flow.py
│   ├── analyze_notebook_structure.py
│   ├── check_gpu.py
│   ├── complete_restructure.py
│   ├── enhance_notebook_theory.py
│   ├── fix_animate_sequence.py
│   ├── update_notebook.py
│   └── update_notebook_distill.py
├── data/                       # Dataset files
│   ├── imagenette2-320/        # ImageNette dataset
│   └── segment_3_test_images/  # Test images for Segment 3
├── docs/                       # Documentation
└── pyproject.toml              # Dependencies (UV)
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

### `segment_3_dataset_images`
```python
from segment_3_dataset_images import (
    ActivationSpectrumTrackerV2,
    FeatureOptimizer,
    plot_neuron_spectrum_distill,
)

tracker = ActivationSpectrumTrackerV2(num_neurons=10, samples_per_category=9)
optimizer = FeatureOptimizer(model)
fig = plot_neuron_spectrum_distill(
    neuron_idx=0,
    layer_name="mixed4a",
    spectrum=tracker.get_spectrum(0),
    optimized_img=optimizer.optimize_neuron("mixed4a", 0),
    negative_optimized_img=optimizer.optimize_neuron_negative("mixed4a", 0),
)
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

### Segment 3 Specific
- **torch-lucent >= 0.1.8** — Feature visualization
- **wandb >= 0.18.0** — Experiment tracking

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
| **Dataset Examples** | Segment 3 | Activation spectrum, min/max/near-threshold |
| **Distill.pub Layout** | Segment 3 | 6-column visualization |

## 📖 References

### Segment 1
- Selvaraju et al., [Grad-CAM](https://arxiv.org/abs/1610.02391), ICCV 2017
- Zeiler & Fergus, [Visualizing CNNs](https://arxiv.org/abs/1311.2901), ECCV 2014

### Segment 2  
- Olah et al., [Feature Visualization](https://distill.pub/2017/feature-visualization/), Distill 2017
- Olah et al., [Circuits](https://distill.pub/2020/circuits/), Distill 2020

## 📜 License

MIT License

