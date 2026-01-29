# Vision Interpretability Project - AI Context

> [!IMPORTANT]
> **READ THIS FIRST**: This file contains the architectural context, design philosophy, and development guidelines for the Vision Interpretability project. All changes should align with the patterns defined here.

## Project Overview
This project consists of **three interactive Jupyter notebooks** designed to demystify Convolutional Neural Networks (CNNs) through a "code-first, visual-first" approach.

**Key Goals:**
1.  **Visual Fidelity**: High-resolution visualizations so features are clearly visible
2.  **Interactive Learning**: Every concept has runnable code and visual output
3.  **Transparency**: Implement interpretability methods from scratch with mathematical rigor
4.  **Colab-Ready**: One-click execution on Google Colab

## Architecture & Tech Stack
- **Language**: Python 3.13+
- **Deep Learning Framework**: PyTorch 2.5+
- **Package Manager**: UV
- **Notebooks**: Three segments covering different aspects of interpretability
- **Feature Visualization**: torch-lucent (Segment 2 & 3)

## Directory Structure
```
VisionInterpretability/
├── AGENTS.md                                  # YOU ARE HERE
├── README.md                                  # Public documentation
├── pyproject.toml                             # Dependencies (UV)
├── notebooks/
│   ├── cataluna84__segment_1_intro.ipynb      # Segment 1: CNN Basics
│   ├── cataluna84__segment_2_activation_max.ipynb  # Segment 2: Feature Viz
│   ├── cataluna84__segment_3_dataset_images.ipynb  # Segment 3: Dataset Examples
│   ├── cataluna84__segment_3_faccent.ipynb    # Segment 3b: Faccent Optimization
│   ├── lucent/                                # Lucent tutorial notebooks (8 notebooks)
│   │   ├── tutorial.ipynb
│   │   ├── activation_grids.ipynb
│   │   ├── diversity.ipynb
│   │   ├── feature_inversion.ipynb
│   │   ├── GAN_parametrization.ipynb
│   │   ├── neuron_interaction.ipynb
│   │   ├── style_transfer.ipynb
│   │   └── modelzoo.ipynb
│   ├── results/                               # Notebook output artifacts
│   └── wandb/                                 # W&B experiment logs
├── src/segment_1_intro/                       # Reusable modules (Segment 1)
│   ├── __init__.py
│   ├── data.py                                # ImageNette loading
│   ├── models.py                              # SimpleCNN, InceptionV1, training
│   └── visualize.py                           # Grad-CAM, Saliency Maps
├── src/segment_3_dataset_images/              # Reusable modules (Segment 3)
│   ├── __init__.py
│   ├── activation_pipeline.py                 # Activation extraction, spectrum tracking
│   ├── visualization.py                       # Distill.pub style plotting
│   └── faccent/                               # Feature visualization library
│       ├── cam.py, mask.py, objectives.py     # Core modules
│       ├── param.py, render.py, transform.py  # Rendering modules
│       ├── utils.py                           # Utilities
│       └── modelzoo/                          # InceptionV1 model
├── scripts/                                   # Notebook enhancement scripts (16 files)
│   ├── enhance_notebook_theory.py
│   ├── add_colab_support_seg3.py
│   ├── update_notebook.py
│   └── ... (13 more)
├── data/                                      # Dataset files
│   ├── imagenette2-320/                       # ImageNette dataset
│   └── segment_3_test_images/                 # Test images
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

### Segment 3: Dataset Examples & Activation Spectrum
**File**: `notebooks/cataluna84__segment_3_dataset_images.ipynb`  
**Dependencies**: `segment_3_dataset_images.{activation_pipeline, visualization}`

**Topics Covered:**
- Finding dataset examples across activation spectrum
- Minimum, slightly negative, slightly positive, maximum examples
- Negative and positive optimized visualizations
- Distill.pub style 6-column layout

**Features:**
✅ Streaming ImageNet data  
✅ W&B experiment logging  
✅ Publication-quality Distill.pub visualizations

### Segment 3b: Faccent Optimization
**File**: `notebooks/cataluna84__segment_3_faccent.ipynb`  
**Dependencies**: `segment_3_dataset_images.faccent`

**Topics Covered:**
- Feature visualization with Faccent library
- Advanced optimization techniques
- Class activation mapping (CAM)

**Features:**
✅ Faccent library integration  
✅ Advanced parametrization options


## Module Reference (Segment 1)

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

## Module Reference (Segment 3)

### `activation_pipeline.py` - Activation Extraction & Tracking
- `ActivationExtractor(model, layer_name)` - Extract layer activations via hooks
- `ActivationSpectrumTrackerV2(num_neurons, samples_per_category)` - Track activation spectrum
- `ImageNetStreamer(batch_size, max_samples)` - Stream ImageNet samples
- `FeatureOptimizer(model, device)` - Generate optimized visualizations
  - `optimize_neuron(layer, channel)` - Positive (max) optimization
  - `optimize_neuron_negative(layer, channel)` - Negative (min) optimization
- `WANDBExperimentLogger(project, run_name)` - W&B logging
- `run_pipeline(config)` - Full pipeline execution

### `visualization.py` - Distill.pub Style Plotting
- `plot_neuron_spectrum_distill(neuron_idx, layer_name, spectrum, ...)` - 6-column layout

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
