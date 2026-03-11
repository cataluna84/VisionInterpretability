# Vision Interpretability: Decoding CNNs

A comprehensive, interactive deep dive into how Convolutional Neural Networks (CNNs) "see" the world through **four tutorial notebooks** plus **two production-ready canonical notebooks** for local execution, covering fundamentals, feature visualization, and dataset-driven analysis.

## 📓 Notebooks

### Tutorial Notebooks (Colab-Optimized)

| Notebook | Description | Launch |
|----------|-------------|--------|
| **Segment 1: CNN Basics & Interpretability** | Convolutions, training on ImageNette, filter & feature map viz, saliency maps, Grad-CAM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_1_intro.ipynb) |
| **Segment 2: Activation Maximization** | Gradient ascent feature viz, FFT parameterization, Distill.pub Circuits reproduction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_2_activation_max.ipynb) |
| **Segment 3: Dataset Examples** | Activation spectrum analysis, min/max/near-threshold examples, Distill.pub 6-column layout | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_3_dataset_images.ipynb) |
| **Segment 3b: Faccent Optimization** | Feature viz with Faccent library, CAM visualization, advanced parametrization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cataluna84/VisionInterpretability/blob/main/notebooks/cataluna84__segment_3_faccent.ipynb) |

---

### Canonical Notebooks (Local Windows Execution)

These are the **production-ready, locally-runnable** versions designed for Windows with GPU support. They include checkpoint/resume, memory management, and robustness features not present in the Colab tutorials.

| Notebook | Description |
|----------|-------------|
| **Segment 2 Canonical** | Batch activation maximization for every neuron in an InceptionV1 layer via Lucent. Optimizes random images via gradient ascent with FFT parameterization, saves lossless `.png` results. Features: resume support (skips already-generated neurons), GPU memory cleanup (`gc.collect` + `torch.cuda.empty_cache`), and configurable layer/neuron ranges. |
| **Segment 3 Canonical** | Two-pass top-K dataset image extraction pipeline over ~1.28M ImageNet images. **Pass 1** streams local WDS shards, computes per-channel activations with AMP (FP16), and maintains min-heaps of the top-K scoring images. **Pass 2** re-streams to save full 224×224 images and spatially-cropped patches around max-activation regions. Features: atomic checkpoint/resume, 4-layer shard integrity verification, DataLoader smoke tests, and configurable channel ranges. |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Tutorials)
Click any Colab badge above → Run all cells. Setup is automatic!

### Option 2: Local Setup

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/cataluna84/VisionInterpretability.git
cd VisionInterpretability

# Install dependencies
uv sync

# Start Jupyter
uv run jupyter lab
```

Then open any notebook in `notebooks/`.

### Running Canonical Notebooks Locally

```bash
# Segment 2: Generate activation maximization images
uv run jupyter lab notebooks/Segment_2_canonical_Windows.ipynb

# Segment 3: Run the top-K pipeline (requires ~144 GB for ImageNet WDS shards)
uv run jupyter lab notebooks/Segment_3_canonical_Windows.ipynb
```

> **Note:** Segment 3 Canonical downloads ImageNet-1k WDS shards to `data/imagenet-1k-wds/` on first run. The download resumes automatically if interrupted.

---

## 📁 Project Structure

```
VisionInterpretability/
├── README.md                                      # This file (GitHub front page)
├── AGENTS.md                                      # AI context & architecture guide
├── pyproject.toml                                 # Dependencies (UV)
├── uv.lock                                        # Locked dependency versions
│
├── notebooks/
│   ├── cataluna84__segment_1_intro.ipynb           # Tutorial: CNN Basics & Interpretability
│   ├── cataluna84__segment_2_activation_max.ipynb  # Tutorial: Activation Maximization
│   ├── cataluna84__segment_3_dataset_images.ipynb  # Tutorial: Dataset Examples
│   ├── cataluna84__segment_3_faccent.ipynb         # Tutorial: Faccent Optimization
│   ├── Segment_2_canonical_Windows.ipynb           # Canonical: Batch activation max (local)
│   ├── Segment_3_canonical_Windows.ipynb           # Canonical: Top-K image extraction (local)
│   ├── cataluna84__segment_2_activation_max_canonical.ipynb  # Earlier canonical draft
│   ├── cataluna84__segment_3_dataset_images_v2.ipynb         # Earlier v2 draft
│   ├── lucent/                                    # Lucent tutorial notebooks (8)
│   │   ├── tutorial.ipynb                         # Getting started with Lucent
│   │   ├── activation_grids.ipynb                 # Activation grid visualizations
│   │   ├── diversity.ipynb                        # Feature diversity analysis
│   │   ├── feature_inversion.ipynb                # Feature inversion techniques
│   │   ├── GAN_parametrization.ipynb              # GAN-based parametrization
│   │   ├── neuron_interaction.ipynb               # Neuron interaction analysis
│   │   ├── style_transfer.ipynb                   # Neural style transfer
│   │   └── modelzoo.ipynb                         # Model zoo examples
│   └── results/                                   # Notebook output artifacts
│       ├── checkpoints/                           # Pipeline checkpoints
│       └── dataset_images/                        # Extracted top-K images
│
├── src/
│   ├── segment_1_intro/                           # Python modules (Segment 1)
│   │   ├── __init__.py
│   │   ├── data.py                                # ImageNette dataset loading
│   │   ├── models.py                              # SimpleCNN, InceptionV1, training
│   │   └── visualize.py                           # Grad-CAM, saliency maps, plotting
│   └── segment_3_dataset_images/                  # Python modules (Segment 3)
│       ├── __init__.py
│       ├── activation_pipeline.py                 # Activation extraction, spectrum tracking
│       ├── visualization.py                       # Distill.pub style plotting
│       └── faccent/                               # Feature visualization library
│           ├── __init__.py
│           ├── cam.py                             # Class activation mapping
│           ├── mask.py                            # Masking utilities
│           ├── objectives.py                      # Optimization objectives
│           ├── objectives_util.py                 # Objective helpers
│           ├── param.py                           # Image parameterization (FFT, pixel)
│           ├── param_util.py                      # Parameterization helpers
│           ├── render.py                          # Rendering engine
│           ├── transform.py                       # Image transforms
│           ├── utils.py                           # Utility functions
│           ├── clean_decorrelated.npy             # Decorrelation matrix
│           └── modelzoo/                          # Pretrained model loaders
│               ├── __init__.py
│               ├── util.py                        # Model loading utilities
│               ├── imagenet_labels.txt            # ImageNet class labels
│               ├── inceptionv1/                   # InceptionV1 model definition
│               └── misc/                          # Miscellaneous model helpers
│
├── scripts/                                       # Notebook enhancement & utility scripts (35)
│
├── data/                                          # Dataset files
│   ├── imagenet-1k-wds/                           # ImageNet-1k WebDataset shards (~144 GB)
│   ├── segment_3_test_images/                     # Test images for Segment 3
│   ├── dog_cat.png                                # Sample image
│   ├── transfer_big_ben.png                       # Style transfer content image
│   ├── transfer_picasso.png                       # Style transfer style image
│   └── transfer_vangogh.png                       # Style transfer style image
│
└── docs/                                          # Documentation
    ├── README.md                                  # Conceptual reference & formulas
    └── performance_optimization.md                # Performance tuning guide
```

---

## 📦 Python Modules

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

---

## 📊 Dependencies

All dependencies are managed via [uv](https://github.com/astral-sh/uv) and defined in `pyproject.toml`.

### Core
| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥ 2.5.0 | Deep learning framework |
| torchvision | ≥ 0.20.0 | Vision models & transforms |
| matplotlib | ≥ 3.9.0 | Plotting & visualization |
| numpy | ≥ 2.0.0 | Numerical computing |
| pillow | ≥ 10.4.0 | Image processing |
| scipy | ≥ 1.14.0 | Scientific computing |

### Machine Learning & Visualization
| Package | Version | Purpose |
|---------|---------|---------|
| torch-lucent | ≥ 0.1.8 | Feature visualization (Lucid port) |
| captum | ≥ 0.7.0 | Model interpretability |
| timm | ≥ 1.0.24 | Pretrained vision models |
| scikit-learn | ≥ 1.5.0 | ML utilities |
| opencv-python | ≥ 4.13.0 | Computer vision operations |
| kornia | ≥ 0.4.1 | Differentiable CV operations |
| einops | ≥ 0.8.1 | Tensor operations |

### Data & Experiment Tracking
| Package | Version | Purpose |
|---------|---------|---------|
| datasets | ≥ 2.20.0 | HuggingFace datasets |
| webdataset | ≥ 0.2.100 | WebDataset streaming |
| huggingface-hub | ≥ 0.30.0 | HF model/dataset hub |
| wandb | ≥ 0.18.0 | Experiment tracking |
| pandas | ≥ 2.2.0 | Data manipulation |
| pyarrow | ≥ 23.0.0 | Columnar data format |

### Development & Notebooks
| Package | Version | Purpose |
|---------|---------|---------|
| jupyterlab | ≥ 4.2.0 | Notebook environment |
| ipywidgets | ≥ 8.1.0 | Interactive widgets |
| plotly | ≥ 6.5.2 | Interactive plots |
| kaleido | ≥ 1.2.0 | Plotly image export |
| nbformat | ≥ 5.10.4 | Notebook manipulation |
| tqdm | ≥ 4.66.0 | Progress bars |
| imageio | ≥ 2.37.2 | Image I/O |
| requests | ≥ 2.32.0 | HTTP requests |
| python-dotenv | ≥ 1.2.1 | Environment variables |

---

## 🎯 What You'll Learn

| Topic | Notebook | Key Concepts |
|-------|----------|-------------|
| **Image Representation** | Segment 1 | Tensors $(C, H, W)$, normalization |
| **Convolutions** | Segment 1 | Kernels, stride, padding, formulas |
| **CNN Training** | Segment 1 | SimpleCNN on ImageNette |
| **Feature Maps** | Segment 1 | Layer activations, what CNNs detect |
| **Saliency Maps** | Segment 1 | $S = \|\nabla_x y^c\|$ |
| **Grad-CAM** | Segment 1 | $L^c = \text{ReLU}(\sum_k \alpha_k^c A^k)$ |
| **Activation Maximization** | Segment 2 | Gradient ascent, FFT parameterization |
| **Feature Visualization** | Segment 2 | Reproducing Distill.pub Circuits |
| **Batch Neuron Viz** | Seg 2 Canonical | Per-neuron activation max with resume |
| **Dataset Examples** | Segment 3 | Activation spectrum, min/max/near-threshold |
| **Distill.pub Layout** | Segment 3 | 6-column visualization |
| **Top-K Extraction** | Seg 3 Canonical | Two-pass pipeline, min-heaps, spatial crops |

---

## 📖 References

- Selvaraju et al., [Grad-CAM](https://arxiv.org/abs/1610.02391), ICCV 2017
- Zeiler & Fergus, [Visualizing CNNs](https://arxiv.org/abs/1311.2901), ECCV 2014
- Olah et al., [Feature Visualization](https://distill.pub/2017/feature-visualization/), Distill 2017
- Olah et al., [Circuits](https://distill.pub/2020/circuits/), Distill 2020

## 📜 License

MIT License
