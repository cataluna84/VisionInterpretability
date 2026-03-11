# Vision Interpretability Project - AI Context

> [!IMPORTANT]
> **READ THIS FIRST**: This file contains the architectural context, design philosophy, and development guidelines for the Vision Interpretability project. All changes should align with the patterns defined here.

## Project Overview
This project consists of **four interactive tutorial notebooks** plus **two canonical production notebooks** designed to demystify Convolutional Neural Networks (CNNs) through a "code-first, visual-first" approach.

**Key Goals:**
1.  **Visual Fidelity**: High-resolution visualizations so features are clearly visible
2.  **Interactive Learning**: Every concept has runnable code and visual output
3.  **Transparency**: Implement interpretability methods from scratch with mathematical rigor
4.  **Colab-Ready**: Tutorial notebooks have one-click execution on Google Colab
5.  **Local Execution**: Canonical notebooks run locally on Windows with GPU, featuring checkpoint/resume and memory management

## Architecture & Tech Stack
- **Language**: Python 3.13+
- **Deep Learning Framework**: PyTorch 2.5+
- **Package Manager**: UV (all commands use `uv run`)
- **Notebooks**: Four tutorial segments + two canonical Windows notebooks
- **Feature Visualization**: torch-lucent (Segments 2 & 3), faccent (Segment 3b)
- **Data**: ImageNette (Segment 1), ImageNet-1k WebDataset shards (Segment 3 Canonical)

## Directory Structure
```
VisionInterpretability/
├── AGENTS.md                                      # YOU ARE HERE
├── README.md                                      # Public documentation (GitHub front page)
├── pyproject.toml                                 # Dependencies (UV)
├── uv.lock                                        # Locked dependency versions
│
├── notebooks/
│   ├── cataluna84__segment_1_intro.ipynb           # Tutorial: CNN Basics
│   ├── cataluna84__segment_2_activation_max.ipynb  # Tutorial: Activation Max
│   ├── cataluna84__segment_3_dataset_images.ipynb  # Tutorial: Dataset Examples
│   ├── cataluna84__segment_3_faccent.ipynb         # Tutorial: Faccent Optimization
│   ├── Segment_2_canonical_Windows.ipynb           # Canonical: Batch activation max (local)
│   ├── Segment_3_canonical_Windows.ipynb           # Canonical: Top-K extraction (local)
│   ├── cataluna84__segment_2_activation_max_canonical.ipynb  # Earlier canonical draft
│   ├── cataluna84__segment_3_dataset_images_v2.ipynb         # Earlier v2 draft
│   ├── lucent/                                    # Lucent tutorial notebooks (8)
│   │   ├── tutorial.ipynb
│   │   ├── activation_grids.ipynb
│   │   ├── diversity.ipynb
│   │   ├── feature_inversion.ipynb
│   │   ├── GAN_parametrization.ipynb
│   │   ├── neuron_interaction.ipynb
│   │   ├── style_transfer.ipynb
│   │   └── modelzoo.ipynb
│   └── results/                                   # Notebook output artifacts
│       ├── checkpoints/                           # Pipeline checkpoints
│       └── dataset_images/                        # Extracted top-K images
│
├── src/
│   ├── segment_1_intro/                           # Reusable modules (Segment 1)
│   │   ├── __init__.py
│   │   ├── data.py                                # ImageNette loading
│   │   ├── models.py                              # SimpleCNN, InceptionV1, training
│   │   └── visualize.py                           # Grad-CAM, Saliency Maps
│   └── segment_3_dataset_images/                  # Reusable modules (Segment 3)
│       ├── __init__.py
│       ├── activation_pipeline.py                 # Activation extraction, spectrum tracking
│       ├── visualization.py                       # Distill.pub style plotting
│       └── faccent/                               # Feature visualization library
│           ├── __init__.py
│           ├── cam.py                             # Class activation mapping
│           ├── mask.py                            # Masking utilities
│           ├── objectives.py, objectives_util.py  # Optimization objectives
│           ├── param.py, param_util.py            # Image parameterization
│           ├── render.py                          # Rendering engine
│           ├── transform.py                       # Image transforms
│           ├── utils.py                           # Utilities
│           ├── clean_decorrelated.npy             # Decorrelation matrix
│           └── modelzoo/                          # InceptionV1 model
│               ├── __init__.py, util.py
│               ├── imagenet_labels.txt
│               ├── inceptionv1/                   # Model definition
│               └── misc/                          # Model helpers
│
├── scripts/                                       # Notebook enhancement & utility scripts (35)
│
├── data/                                          # Dataset files
│   ├── imagenet-1k-wds/                           # ImageNet-1k WDS shards (~144 GB)
│   ├── segment_3_test_images/                     # Test images
│   ├── dog_cat.png                                # Sample image
│   ├── transfer_big_ben.png                       # Style transfer content
│   ├── transfer_picasso.png                       # Style transfer style
│   └── transfer_vangogh.png                       # Style transfer style
│
└── docs/                                          # Conceptual documentation
    ├── README.md                                  # Formula reference & changelog
    └── performance_optimization.md                # Performance tuning guide
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

### Segment 2 Canonical (Local Windows)
**File**: `notebooks/Segment_2_canonical_Windows.ipynb`
**Dependencies**: `torch-lucent`

**Topics Covered:**
- Batch activation maximization for every neuron in a chosen InceptionV1 layer
- FFT-parameterized gradient ascent via Lucent
- Lossless `.png` output per neuron

**Features:**
✅ Resume support (skips neurons with existing `.png` files)
✅ GPU memory management (`gc.collect` + `torch.cuda.empty_cache`)
✅ Configurable layer and neuron ranges
✅ Designed for local Windows execution with GPU

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

### Segment 3 Canonical (Local Windows)
**File**: `notebooks/Segment_3_canonical_Windows.ipynb`
**Dependencies**: `torch-lucent`, `webdataset`, `huggingface-hub`

**Topics Covered:**
- Two-pass top-K dataset image extraction over ~1.28M ImageNet images
- Pass 1: Stream local WDS shards, compute per-channel activations with AMP (FP16), maintain min-heaps
- Pass 2: Re-stream to save full images and spatially-cropped patches
- Atomic checkpoint/resume system
- 4-layer shard integrity verification

**Features:**
✅ Local ImageNet-1k WDS shards (auto-download via `huggingface_hub`)
✅ Atomic checkpointing (`.tmp` → `os.replace` → `.pkl`)
✅ AMP (FP16) for halved activation memory
✅ DataLoader smoke test before pipeline execution
✅ Configurable channel ranges and batch sizes
✅ Designed for 8 GB GPU memory budget

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
- Use `uv run <command>` to run any project command
- `pyproject.toml` is source of truth
- Segment 2 requires `torch-lucent` in addition to core deps
- Segment 3 Canonical requires `webdataset` and `huggingface-hub`

### 4. Data Management
- `data/imagenet-1k-wds/` contains local ImageNet WDS shards (~144 GB), downloaded at runtime
- `data/imagenette2-320/` is gitignored (downloaded at runtime)
- Style transfer images (`data/transfer_*.png`) are tracked

## Interpretability Concepts

| Concept | Definition | Formula |
|---------|-----------|----------|
| **Filters** | Learned convolutional kernels | $\mathbf{W}^{(l)} \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$ |
| **Feature Maps** | Layer activations | $A^k = \sigma(W^k * X + b^k)$ |
| **Saliency** | Input gradient magnitude | $S = \|\nabla_x y^c\|$ |
| **Grad-CAM** | Weighted activation map | $L^c = \text{ReLU}(\sum_k \alpha_k^c A^k)$ |
| **Activation Max** | Optimal input for neuron | $\mathbf{x}^* = \arg\max_{\mathbf{x}} a_{l,k}(f(\mathbf{x}))$ |
| **Top-K Extraction** | Highest-activating dataset images | $\text{score}_c(x) = \max_{i,j} A^c_{i,j}(x)$ |
