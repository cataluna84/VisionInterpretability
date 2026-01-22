r"""Visualization utilities for CNN interpretability.

This module provides functions and classes for visualizing convolutional
neural network internals and generating interpretability visualizations.
Key capabilities include:

- **Image Display**: Denormalization and display of tensors.
- **Filter Visualization**: View learned convolutional kernels.
- **Feature Maps**: Visualize intermediate layer activations.
- **Saliency Maps**: Gradient-based input attribution using vanilla gradients.
- **Grad-CAM**: Class Activation Mapping for spatial localization.

Mathematical Background:
    **Saliency Maps** compute the gradient of the class score with respect
    to input pixels:

    .. math::

        S = \\left| \\frac{\\partial y_c}{\\partial x} \\right|

    **Grad-CAM** computes importance weights and generates a heatmap:

    .. math::

        L^c_{Grad-CAM} = ReLU\\left(\\sum_k \\alpha_k^c A^k\\right)

    where :math:`\\alpha_k^c = \\frac{1}{Z} \\sum_i \\sum_j \\frac{\\partial y^c}{\\partial A^k_{ij}}`

Example:
    Compute and visualize a saliency map::

        from segment_1_intro import visualize

        saliency = visualize.compute_saliency_map(model, image, target_class=5)
        visualize.visualize_saliency(image, saliency, title="Dog Saliency")

    Use Grad-CAM for localization::

        gradcam = visualize.GradCAM(model, model.conv3)
        heatmap = gradcam(image, target_class=5, device='cuda')
        visualize.visualize_gradcam(image, heatmap, "dog")

Note:
    Most functions expect normalized tensors with ImageNet statistics.
    Use ``denormalize_image()`` to convert back for display.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import cv2


# ============================================================================
# Basic Visualization Functions
# ============================================================================

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized PyTorch tensor to a displayable numpy image.

    Reverses ImageNet normalization (mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]) and transposes from CHW to HWC format.

    Args:
        tensor: Normalized image tensor with shape ``(C, H, W)``.

    Returns:
        NumPy array with shape ``(H, W, C)`` and values in range [0, 1].

    Example:
        >>> img_np = denormalize_image(normalized_tensor)
        >>> plt.imshow(img_np)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def show_image(
    tensor: torch.Tensor,
    title: str = None,
    ax: Optional[plt.Axes] = None
):
    """Display a tensor image using matplotlib.

    Handles both normalized RGB tensors and raw arrays. Creates a new
    figure if no axes object is provided.

    Args:
        tensor: Image tensor with shape ``(C, H, W)`` or ``(H, W)``.
        title: Optional title for the plot.
        ax: Optional matplotlib Axes to plot on. If None, creates new figure.

    Example:
        >>> show_image(image_tensor, title="Sample Image")
    """
    if tensor.shape[0] == 3:  # RGB normalized image
        img = denormalize_image(tensor)
    else:
        img = tensor.detach().cpu().numpy()
    
    if ax is None:
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        ax.imshow(img)
        if title:
            ax.set_title(title)
        ax.axis("off")


def visualize_filters(
    model_layer,
    num_filters: int = 64,
    figsize: Tuple[int, int] = (10, 10)
):
    """Visualize learned convolutional filter weights.

    Displays the weight tensors of a Conv2d layer as images. For RGB
    input layers (3 channels), shows color filters. For deeper layers,
    shows grayscale representations.

    Args:
        model_layer: A ``torch.nn.Conv2d`` layer to visualize.
        num_filters: Maximum number of filters to display.
        figsize: Figure size as (width, height) tuple.

    Example:
        >>> visualize_filters(model.conv1, num_filters=32)
    """
    weights = model_layer.weight.data.cpu()
    num_filters = min(num_filters, weights.shape[0])
    
    # Normalize to 0-1 for display
    w_min, w_max = weights.min(), weights.max()
    weights = (weights - w_min) / (w_max - w_min)
    
    n_grids = int(np.ceil(np.sqrt(num_filters)))
    fig, axes = plt.subplots(n_grids, n_grids, figsize=figsize)
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            if weights.shape[1] == 3:  # RGB filters (first layer)
                img = weights[i].permute(1, 2, 0)
                ax.imshow(img)
            else:  # Grayscale for deeper layers
                ax.imshow(weights[i, 0], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.suptitle(f"First {num_filters} Filters", fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Feature Map Visualization
# ============================================================================

def visualize_feature_maps(
    activation: torch.Tensor,
    num_maps: int = 16,
    figsize: Tuple[int, int] = (12, 12),
    title: str = "Feature Maps"
):
    """
    Visualize feature maps from a convolutional layer.
    
    Args:
        activation: Feature map tensor (C, H, W) or (1, C, H, W)
        num_maps: Number of feature maps to display
        figsize: Figure size
        title: Plot title
    """
    # Remove batch dimension if present
    if activation.dim() == 4:
        activation = activation.squeeze(0)
    
    activation = activation.detach().cpu()
    num_maps = min(num_maps, activation.shape[0])
    n_grids = int(np.ceil(np.sqrt(num_maps)))
    
    fig, axes = plt.subplots(n_grids, n_grids, figsize=figsize)
    
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(activation[i], cmap='viridis')
            ax.set_title(f"Map {i}")
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Saliency Maps / Vanilla Gradients
# ============================================================================

def compute_saliency_map(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute saliency map using vanilla gradients.
    
    The saliency map shows the gradient of the target class score
    with respect to the input image pixels.
    
    Args:
        model: Neural network model
        input_tensor: Input image tensor (1, C, H, W) or (C, H, W)
        target_class: Target class index. If None, uses predicted class.
        device: Device to use for computation
    
    Returns:
        Saliency map tensor (H, W)
    """
    model.eval()
    
    # Ensure batch dimension
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)
    
    # Forward pass
    output = model(input_tensor)
    
    # Handle different output formats (tuple from InceptionV1, tensor from simple CNNs)
    if isinstance(output, tuple):
        output = output[0]
    
    # Get target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass for target class
    model.zero_grad()
    score = output[0, target_class]
    score.backward()
    
    # Get gradients and compute saliency
    gradients = input_tensor.grad.data.abs()
    
    # Take max across color channels
    saliency, _ = torch.max(gradients.squeeze(0), dim=0)
    
    return saliency.detach().cpu()


def visualize_saliency(
    input_tensor: torch.Tensor,
    saliency_map: torch.Tensor,
    title: str = "Saliency Map",
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Visualize saliency map alongside the original image.
    
    Args:
        input_tensor: Original input tensor (C, H, W)
        saliency_map: Saliency map (H, W)
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    img = denormalize_image(input_tensor)
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Saliency map
    saliency_np = saliency_map.numpy()
    axes[1].imshow(saliency_np, cmap='hot')
    axes[1].set_title("Saliency Map")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(saliency_np, cmap='hot', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Grad-CAM Implementation
# ============================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).

    Generates a coarse localization heatmap highlighting regions important
    for a specific class prediction. Works by computing gradients of the
    target class score with respect to feature maps in a target layer.

    The Grad-CAM formula is:

    .. math::

        L^c_{Grad-CAM} = ReLU\\left(\\sum_k \\alpha_k^c A^k\\right)

    where :math:`\\alpha_k^c` are the importance weights computed via
    global average pooling of gradients.

    Attributes:
        model: The neural network model.
        target_layer: The convolutional layer used for computing CAM.
        gradients: Stored gradients from backward pass.
        activations: Stored activations from forward pass.

    Example:
        >>> gradcam = GradCAM(model, model.layer4[-1])
        >>> heatmap = gradcam(image, target_class=243, device='cuda')
        >>> visualize_gradcam(image, heatmap, "bulldog")

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """Initialize Grad-CAM with model and target layer.

        Args:
            model: The neural network model to interpret.
            target_layer: The convolutional layer to compute CAM from.
                Typically the last conv layer before global pooling.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image (1, C, H, W) or (C, H, W)
            target_class: Target class index. If None, uses predicted class.
            device: Device for computation
        
        Returns:
            Grad-CAM heatmap (H, W) - values in [0, 1]
        """
        self.model.eval()
        
        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle tuple outputs (InceptionV1)
        if isinstance(output, tuple):
            output = output[0]
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu()


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: Optional[int] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Convenience function to compute Grad-CAM.
    
    Args:
        model: Neural network model
        input_tensor: Input image tensor
        target_layer: Target convolutional layer
        target_class: Target class (None = predicted class)
        device: Computation device
    
    Returns:
        Grad-CAM heatmap tensor (H, W)
    """
    gradcam = GradCAM(model, target_layer)
    return gradcam(input_tensor, target_class, device)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: torch.Tensor,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    
    Args:
        image: Original image (H, W, C) in range [0, 1]
        heatmap: Heatmap tensor (H, W) in range [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use
    
    Returns:
        Overlay image (H, W, C) in range [0, 1]
    """
    # Convert heatmap to numpy
    heatmap_np = heatmap.numpy() if isinstance(heatmap, torch.Tensor) else heatmap
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap_np, (image.shape[1], image.shape[0]))
    
    # Convert to uint8 for colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def visualize_gradcam(
    input_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    predicted_class: str = "",
    title: str = "Grad-CAM",
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize Grad-CAM heatmap with original image and overlay.
    
    Args:
        input_tensor: Original input tensor (C, H, W)
        heatmap: Grad-CAM heatmap (H, W)
        predicted_class: Name of predicted class for display
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    img = denormalize_image(input_tensor)
    axes[0].imshow(img)
    axes[0].set_title(f"Original\n{predicted_class}")
    axes[0].axis("off")
    
    # Heatmap only
    heatmap_np = heatmap.numpy() if isinstance(heatmap, torch.Tensor) else heatmap
    heatmap_resized = cv2.resize(heatmap_np, (img.shape[1], img.shape[0]))
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    # Overlay
    overlay = overlay_heatmap(img, heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Training Visualization
# ============================================================================

def plot_training_history(
    history: dict,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def show_predictions(
    images: torch.Tensor,
    true_labels: List[str],
    pred_labels: List[str],
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Display a grid of images with true and predicted labels.
    
    Args:
        images: Batch of images (N, C, H, W)
        true_labels: List of true class names
        pred_labels: List of predicted class names
        figsize: Figure size
    """
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flat if n > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < n:
            img = denormalize_image(images[i])
            ax.imshow(img)
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            ax.set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}", color=color)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
