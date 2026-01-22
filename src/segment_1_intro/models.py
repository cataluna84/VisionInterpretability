"""Model definitions and training utilities for Vision Interpretability.

This module provides neural network architectures and training functions for
demonstrating CNN interpretability concepts. It includes a simple 3-layer CNN
for educational purposes and utilities to load pretrained InceptionV1 (GoogLeNet).

Example:
    Create and train a SimpleCNN on ImageNette::

        from segment_1_intro import models, data

        model = models.load_simple_cnn(num_classes=10)
        train_loader = data.load_imagenette(split='train')
        val_loader = data.load_imagenette(split='val')

        history = models.train_model(
            model, train_loader, val_loader, epochs=5, device='cuda'
        )

    Load pretrained InceptionV1::

        inception = models.load_inception_v1(pretrained=True)

Attributes:
    SimpleCNN: A simple 3-layer CNN class for educational demonstrations.

Note:
    Training functions use Adam optimizer and CrossEntropyLoss by default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from typing import Tuple, Optional
from tqdm import tqdm


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_inception_v1(pretrained: bool = True) -> torch.nn.Module:
    """Load InceptionV1 (GoogLeNet) model for feature visualization.

    Returns a pretrained GoogLeNet model, which was one of the first deep
    networks to use inception modules (parallel convolutions of different
    sizes). This architecture is commonly used in interpretability research.

    The model outputs 1000 ImageNet classes and has auxiliary classifiers
    that are disabled in eval mode.

    Args:
        pretrained: If True, loads ImageNet pretrained weights.
            If False, initializes with random weights.

    Returns:
        A GoogLeNet model instance in evaluation mode.

    Example:
        >>> inception = load_inception_v1(pretrained=True)
        >>> inception.eval()
        >>> output = inception(torch.randn(1, 3, 224, 224))
        >>> print(output.shape)  # torch.Size([1, 1000])

    Note:
        In torchvision, InceptionV1 is implemented as ``googlenet``.
        Input images should be 224x224 and normalized with ImageNet stats.
    """
    weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.googlenet(weights=weights)
    model.eval()
    return model


def load_simple_cnn(num_classes: int = 10) -> torch.nn.Module:
    """Create a simple 3-layer CNN for educational demonstrations.

    This function returns a lightweight CNN designed for learning about
    convolutional neural networks and interpretability methods. The
    architecture uses:

    - 3 convolutional layers (32 → 64 → 128 channels)
    - Batch normalization after each conv layer
    - MaxPooling (2x2) after each conv block
    - Adaptive average pooling for input size flexibility
    - 2 fully connected layers with dropout

    Args:
        num_classes: Number of output classes. Default is 10 for ImageNette.

    Returns:
        A SimpleCNN model instance ready for training.

    Example:
        >>> model = load_simple_cnn(num_classes=10)
        >>> print(sum(p.numel() for p in model.parameters()))  # ~1.3M params
        >>> output = model(torch.randn(1, 3, 128, 128))
        >>> print(output.shape)  # torch.Size([1, 10])
    """

    class SimpleCNN(nn.Module):
        """A simple 3-layer CNN for educational purposes.

        This network is designed to be small enough for quick training while
        still demonstrating core CNN concepts like convolution, pooling,
        batch normalization, and dropout.

        Architecture::

            Input (3, H, W)
              ↓
            Conv1 (32 filters) → BN → ReLU → MaxPool
              ↓
            Conv2 (64 filters) → BN → ReLU → MaxPool
              ↓
            Conv3 (128 filters) → BN → ReLU → MaxPool
              ↓
            AdaptiveAvgPool (4x4)
              ↓
            FC1 (512) → ReLU → Dropout
              ↓
            FC2 (num_classes)

        Attributes:
            conv1, conv2, conv3: Convolutional layers.
            bn1, bn2, bn3: Batch normalization layers.
            fc1, fc2: Fully connected layers.
        """
        def __init__(self):
            super().__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            
            self.pool = nn.MaxPool2d(2, 2)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) # Output will always be 128 x 4 x 4
            self.dropout = nn.Dropout(0.25)
            
            # Fully connected layers
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            
            # Adaptive pooling ensures consistent size regardless of input dimensions
            x = self.adaptive_pool(x)
            
            # Flatten and fully connected
            x = torch.flatten(x, 1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
        
        def get_last_conv_layer(self):
            """Returns the last convolutional layer for Grad-CAM."""
            return self.conv3

    return SimpleCNN()


# ============================================================================
# Training Utilities
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
    show_progress: bool = True
) -> Tuple[float, float]:
    """Train a model for one epoch.

    Performs a complete pass through the training dataset, computing
    forward and backward passes for each batch.

    Args:
        model: The neural network model to train.
        dataloader: DataLoader containing training data.
        optimizer: Optimizer instance (e.g., Adam, SGD).
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: Device to train on ('cpu' or 'cuda').
        show_progress: If True, displays a tqdm progress bar.

    Returns:
        A tuple of (average_loss, accuracy_percentage) for the epoch.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> criterion = nn.CrossEntropyLoss()
        >>> loss, acc = train_one_epoch(model, train_loader, optimizer, criterion)
        >>> print(f"Loss: {loss:.4f}, Acc: {acc:.2f}%")
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(dataloader, desc="Training", leave=False) if show_progress else dataloader
    
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if show_progress:
            iterator.set_postfix(loss=loss.item(), acc=100.*correct/total)
    
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
    show_progress: bool = True
) -> Tuple[float, float]:
    """Evaluate a model on a validation or test dataset.

    Performs inference on the entire dataset without gradient computation,
    calculating average loss and accuracy metrics.

    Args:
        model: The neural network model to evaluate.
        dataloader: DataLoader containing validation/test data.
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: Device to use for computation ('cpu' or 'cuda').
        show_progress: If True, displays a tqdm progress bar.

    Returns:
        A tuple of (average_loss, accuracy_percentage).

    Example:
        >>> criterion = nn.CrossEntropyLoss()
        >>> val_loss, val_acc = evaluate(model, val_loader, criterion, 'cuda')
        >>> print(f"Validation Acc: {val_acc:.2f}%")
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(dataloader, desc="Evaluating", leave=False) if show_progress else dataloader
    
    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: str = "cpu",
    show_progress: bool = True
) -> dict:
    """Complete training loop with optional validation.

    Trains a model for the specified number of epochs using Adam optimizer
    and CrossEntropyLoss. Tracks training and validation metrics.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        epochs: Number of training epochs. Default is 5.
        learning_rate: Learning rate for Adam optimizer. Default is 0.001.
        device: Device to train on ('cpu' or 'cuda').
        show_progress: If True, displays progress bars during training.

    Returns:
        A dictionary containing training history with keys:
            - 'train_loss': List of training losses per epoch.
            - 'train_acc': List of training accuracies per epoch.
            - 'val_loss': List of validation losses (if val_loader provided).
            - 'val_acc': List of validation accuracies (if val_loader provided).

    Example:
        >>> history = train_model(model, train_loader, val_loader, epochs=10)
        >>> print(f"Final Val Acc: {history['val_acc'][-1]:.2f}%")
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, show_progress
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Validation
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, show_progress)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            print(f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
    
    return history


def get_predictions(
    model: nn.Module,
    images: torch.Tensor,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get model predictions for a batch of images.

    Runs inference on a batch of images and returns predicted class
    indices and softmax probabilities.

    Args:
        model: The neural network model for inference.
        images: Batch of images with shape ``(N, C, H, W)``.
        device: Device to use for computation.

    Returns:
        A tuple of:
            - predicted_classes: Tensor of shape ``(N,)`` with class indices.
            - probabilities: Tensor of shape ``(N, num_classes)`` with softmax probs.

    Example:
        >>> preds, probs = get_predictions(model, images, 'cuda')
        >>> print(f"Predicted: {preds[0]}, Confidence: {probs[0].max():.2%}")
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
    
    return predicted.cpu(), probs.cpu()
