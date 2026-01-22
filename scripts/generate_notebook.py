"""
Notebook Generator for Vision Interpretability Tutorial.
Generates the main Jupyter notebook with all sections.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os


def create_notebook():
    """Generate the complete vision interpretability notebook."""
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13"
        }
    }

    cells = []

    # ========================================================================
    # SECTION 1: Introduction
    # ========================================================================
    cells.append(new_markdown_cell("""# Vision Interpretability: Decoding CNNs

Welcome to this interactive tutorial on **Computer Vision** and **Convolutional Neural Networks (CNNs)**.

## What You'll Learn

1. **Image Representation** — How computers "see" images as tensors
2. **Convolution Operations** — The math behind edge detection, blur, and sharpening
3. **Building a CNN** — Train a model from scratch on ImageNette
4. **Feature Visualization** — See what patterns each layer detects
5. **Interpretability Methods** — Understand *why* a model makes predictions
   - Saliency Maps (Vanilla Gradients)
   - Grad-CAM (Class Activation Mapping)

Let's decode the black box! 🧠
"""))

    # Setup cell
    cells.append(new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Import our custom modules
from vision_interpret import models, visualize, data

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Visualization settings
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
"""))

    # ========================================================================
    # SECTION 2: Image Representation
    # ========================================================================
    cells.append(new_markdown_cell("""## 1. How Images Are Represented

To a computer, an image is a **3D tensor** with shape `(Channels, Height, Width)`:
- **Channels**: Usually 3 (Red, Green, Blue)
- **Height/Width**: Pixel dimensions

Each pixel value ranges from 0-255 (or 0-1 after normalization).
"""))

    cells.append(new_code_cell("""# Download a sample image
url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb&w=512"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('RGB')

# Display original image
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Convert to tensor and show individual channels
img_tensor = transforms.ToTensor()(img)
print(f"Tensor Shape: {img_tensor.shape} (C, H, W)")
print(f"Value Range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

# Show RGB channels
channel_names = ["Red Channel", "Green Channel", "Blue Channel"]
cmaps = ["Reds", "Greens", "Blues"]

for i, (name, cmap) in enumerate(zip(channel_names, cmaps)):
    axes[i+1].imshow(img_tensor[i], cmap=cmap)
    axes[i+1].set_title(name)
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()
"""))

    # ========================================================================
    # SECTION 3: Convolution Operations
    # ========================================================================
    cells.append(new_markdown_cell("""## 2. Convolution Operations Deep Dive

A **convolution** slides a small filter (kernel) across the image to extract features.
Different kernels detect different patterns:
- **Edge Detection**: Find boundaries between objects
- **Blur/Smoothing**: Reduce noise
- **Sharpening**: Enhance edges
"""))

    cells.append(new_code_cell("""def manual_convolution(image_tensor, kernel):
    \"\"\"
    Simple manual 2D convolution for demonstration.
    image_tensor: (1, H, W) -> Grayscale
    kernel: (K, K) -> Filter
    \"\"\"
    c, h, w = image_tensor.shape
    kh, kw = kernel.shape
    output_h, output_w = h - kh + 1, w - kw + 1
    output = torch.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            region = image_tensor[0, i:i+kh, j:j+kw]
            output[i, j] = torch.sum(region * kernel)
    
    return output

# Define various filters
kernels = {
    "Sobel X (Vertical Edges)": torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32),
    "Sobel Y (Horizontal Edges)": torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32),
    "Laplacian (All Edges)": torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32),
    "Gaussian Blur": torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16,
    "Sharpen": torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32),
}

# Convert to grayscale
gray_img = transforms.Grayscale()(img_tensor)

# Apply each filter
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(gray_img.squeeze(), cmap='gray')
axes[0].set_title("Original (Grayscale)")
axes[0].axis('off')

for idx, (name, kernel) in enumerate(kernels.items(), 1):
    result = manual_convolution(gray_img, kernel)
    axes[idx].imshow(result.abs(), cmap='gray')
    axes[idx].set_title(name)
    axes[idx].axis('off')

plt.suptitle("Convolution Filter Effects", fontsize=16)
plt.tight_layout()
plt.show()
"""))

    # ========================================================================
    # SECTION 4: Building & Training CNN on ImageNette
    # ========================================================================
    cells.append(new_markdown_cell("""## 3. Building & Training a CNN on ImageNette

Now let's build a simple 3-layer CNN and train it on the **ImageNette** dataset 
(a subset of ImageNet with 10 easy-to-classify classes).

### ImageNette Classes:
1. Tench (fish)
2. English Springer (dog)
3. Cassette Player
4. Chain Saw
5. Church
6. French Horn
7. Garbage Truck
8. Gas Pump
9. Golf Ball
10. Parachute
"""))

    cells.append(new_code_cell("""# Load ImageNette dataset from Hugging Face
print("📦 Loading ImageNette dataset from Hugging Face...")
train_loader = data.load_imagenette(split="train", image_size=128, batch_size=32)
val_loader = data.load_imagenette(split="validation", image_size=128, batch_size=32)

print(f"✅ Training samples: ~{len(train_loader) * 32}")
print(f"✅ Validation samples: ~{len(val_loader) * 32}")
print(f"📋 Classes: {data.IMAGENETTE_CLASSES}")
"""))

    cells.append(new_code_cell("""# Visualize some training samples
sample_images, sample_labels = data.get_sample_images(train_loader, num_samples=8)
classes = data.IMAGENETTE_CLASSES

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    img = visualize.denormalize_image(sample_images[i])
    ax.imshow(img)
    ax.set_title(classes[sample_labels[i]])
    ax.axis('off')

plt.suptitle("ImageNette Training Samples", fontsize=16)
plt.tight_layout()
plt.show()
"""))

    cells.append(new_code_cell("""# Create our SimpleCNN model
model = models.load_simple_cnn(num_classes=10)
print(model)
print(f"\\n📊 Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
"""))

    cells.append(new_markdown_cell("""### Training the Model

⚠️ **Note**: Training takes a few minutes. For a quick demo, we train for just 3 epochs.
In practice, you'd train for 20+ epochs to achieve better accuracy.
"""))

    cells.append(new_code_cell("""# Train the model (reduced epochs for demo)
EPOCHS = 3  # Increase to 10-20 for better results

print(f"🚀 Training SimpleCNN for {EPOCHS} epochs...")
history = models.train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    learning_rate=0.001,
    device=device,
    show_progress=True
)

print("\\n✅ Training complete!")
"""))

    cells.append(new_code_cell("""# Plot training history
visualize.plot_training_history(history)
"""))

    # ========================================================================
    # SECTION 5: Feature Map Visualization
    # ========================================================================
    cells.append(new_markdown_cell("""## 4. Feature Map Visualization

**Feature maps** are the activations after each convolutional layer.
They show *where* in the image the network detected specific patterns.

- **Early layers**: Detect edges, colors, textures
- **Deep layers**: Detect high-level concepts (eyes, wheels, etc.)
"""))

    cells.append(new_code_cell("""# Get a sample image for visualization
sample_img, sample_label = sample_images[0:1], sample_labels[0]
print(f"Analyzing image of: {classes[sample_label]}")

# Register hooks to capture activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks on each conv layer
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))
model.conv3.register_forward_hook(get_activation('conv3'))

# Forward pass
model.eval()
with torch.no_grad():
    _ = model(sample_img.to(device))

# Visualize feature maps from each layer
for layer_name in ['conv1', 'conv2', 'conv3']:
    acts = activations[layer_name].cpu()
    print(f"\\n{layer_name} output shape: {acts.shape}")
    visualize.visualize_feature_maps(acts, num_maps=16, title=f"Feature Maps: {layer_name}")
"""))

    # ========================================================================
    # SECTION 6: Filter Visualization
    # ========================================================================
    cells.append(new_markdown_cell("""## 5. Filter (Kernel) Visualization

**Filters** are the learnable weights of convolutional layers.
Visualizing them shows what patterns the network has learned to detect.
"""))

    cells.append(new_code_cell("""# Visualize learned filters from the first conv layer
print("First Layer Filters (RGB - what patterns is the model looking for?):")
visualize.visualize_filters(model.conv1, num_filters=32)
"""))

    cells.append(new_code_cell("""# Compare with random (untrained) filters
random_model = models.load_simple_cnn(num_classes=10)  # Fresh model
print("Random (Untrained) Filters:")
visualize.visualize_filters(random_model.conv1, num_filters=32)
"""))

    # ========================================================================
    # SECTION 7: Saliency Maps
    # ========================================================================
    cells.append(new_markdown_cell("""## 6. Gradient-Based Interpretability: Saliency Maps

**Saliency maps** (Vanilla Gradients) show which input pixels most influence the prediction.

**How it works**:
1. Forward pass to get prediction
2. Backward pass to compute gradients with respect to input pixels
3. High gradient = high importance
"""))

    cells.append(new_code_cell("""# Compute saliency map for our sample image
model.eval()
saliency = visualize.compute_saliency_map(
    model=model,
    input_tensor=sample_img,
    target_class=sample_label.item(),
    device=device
)

# Visualize
visualize.visualize_saliency(
    input_tensor=sample_images[0],
    saliency_map=saliency,
    title=f"Saliency Map for '{classes[sample_label]}'"
)
"""))

    # ========================================================================
    # SECTION 8: Grad-CAM
    # ========================================================================
    cells.append(new_markdown_cell("""## 7. Class Activation Mapping (Grad-CAM)

**Grad-CAM** produces a heatmap highlighting the regions most important for a specific class.

**How it works**:
1. Compute gradients of target class with respect to final conv layer
2. Global average pool the gradients to get importance weights
3. Weighted sum of feature maps → Heatmap
4. ReLU to keep only positive contributions

This is more interpretable than saliency maps because it focuses on *semantic regions*.
"""))

    cells.append(new_code_cell("""# Compute Grad-CAM for our trained SimpleCNN
# We need to get the last conv layer
target_layer = model.conv3

gradcam = visualize.GradCAM(model, target_layer)
heatmap = gradcam(
    input_tensor=sample_img,
    target_class=sample_label.item(),
    device=device
)

# Visualize
visualize.visualize_gradcam(
    input_tensor=sample_images[0],
    heatmap=heatmap,
    predicted_class=classes[sample_label],
    title=f"Grad-CAM: Why is this a '{classes[sample_label]}'?"
)
"""))

    # ========================================================================
    # SECTION 9: InceptionV1 Deep Interpretability
    # ========================================================================
    cells.append(new_markdown_cell("""## 8. Deep Interpretability with InceptionV1

Let's apply these techniques to a professional, pretrained model: **InceptionV1 (GoogLeNet)**.

This model was trained on the full ImageNet dataset (1000 classes) and has much more 
sophisticated feature detectors.
"""))

    cells.append(new_code_cell("""# Load pretrained InceptionV1
inception = models.load_inception_v1(pretrained=True).to(device)
print("✅ InceptionV1 (GoogLeNet) loaded!")

# Download a sample image for InceptionV1 demo
from PIL import Image
import requests
from io import BytesIO

sample_url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb&w=512"
response = requests.get(sample_url)
sample_pil_image = Image.open(BytesIO(response.content)).convert('RGB')

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(sample_pil_image)
plt.axis('off')
plt.title("Input Image for InceptionV1")
plt.show()

# Preprocess image for Inception (224x224)
inception_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transform to PIL image
inception_input = inception_transform(sample_pil_image).unsqueeze(0).to(device)

# Get prediction
with torch.no_grad():
    output = inception(inception_input)
    probs = F.softmax(output[0], dim=0)
    top5_prob, top5_idx = probs.topk(5)

# Load ImageNet labels
import json
import urllib.request
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(LABELS_URL) as url:
    imagenet_labels = json.loads(url.read().decode())

print("🏆 Top 5 Predictions:")
for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx)):
    print(f"  {i+1}. {imagenet_labels[idx]}: {prob:.2%}")
"""))

    cells.append(new_code_cell("""# Visualize InceptionV1 first layer filters
print("InceptionV1 First Layer Filters:")
visualize.visualize_filters(inception.conv1.conv, num_filters=64)
"""))

    cells.append(new_code_cell("""# Grad-CAM on InceptionV1
# Target the last conv layer before the GAP (inception5b)
inception_target_layer = inception.inception5b

inception_gradcam = visualize.GradCAM(inception, inception_target_layer.branch4[1])
inception_heatmap = inception_gradcam(
    input_tensor=inception_input,
    target_class=top5_idx[0].item(),
    device=device
)

# Visualize
visualize.visualize_gradcam(
    input_tensor=inception_input.squeeze(0),
    heatmap=inception_heatmap,
    predicted_class=imagenet_labels[top5_idx[0]],
    title=f"Grad-CAM on InceptionV1"
)
"""))

    # ========================================================================
    # SECTION 10: Summary
    # ========================================================================
    cells.append(new_markdown_cell("""## 9. Summary

### What We Learned

| Technique | Question Answered | Output |
|-----------|------------------|--------|
| **Feature Maps** | Where did the model detect patterns? | Activation grids |
| **Filter Visualization** | What patterns is it looking for? | Kernel weights as images |
| **Saliency Maps** | Which pixels influenced the prediction? | Gradient-based heatmap |
| **Grad-CAM** | Which *regions* were important for the class? | Coarse localization heatmap |

### Key Insights

1. **Early layers** detect low-level features (edges, colors, textures)
2. **Deep layers** detect high-level concepts (object parts, shapes)
3. **Grad-CAM** is more interpretable than raw saliency because it's spatially coarse
4. These techniques help us:
   - Debug model failures
   - Build trust in AI systems
   - Understand what the model "sees"

### Next Steps

- Try different images and see how the heatmaps change
- Experiment with different target classes in Grad-CAM
- Explore other interpretability methods: Integrated Gradients, SHAP, LIME
"""))

    cells.append(new_code_cell("""print("🎉 Congratulations! You've completed the Vision Interpretability tutorial!")
print("\\n📚 Resources:")
print("  - Grad-CAM Paper: https://arxiv.org/abs/1610.02391")
print("  - Captum (PyTorch Interpretability): https://captum.ai/")
print("  - ImageNette Dataset: https://huggingface.co/datasets/frgfm/imagenette")
"""))

    # Create notebook
    nb.cells = cells

    os.makedirs("notebooks", exist_ok=True)
    output_path = "notebooks/vision_interpretability.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    
    print(f"✅ Notebook created: {output_path}")
    print(f"📊 Total cells: {len(cells)}")


if __name__ == "__main__":
    create_notebook()
