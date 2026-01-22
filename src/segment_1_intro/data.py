"""Data loading utilities for the ImageNette dataset.

This module provides functions to download, preprocess, and load the ImageNette
dataset, which is a subset of ImageNet containing 10 easily classifiable classes.
The dataset is sourced from fast.ai's S3 bucket and uses torchvision's ImageFolder
for loading.

Example:
    Load the training set with default settings::

        from segment_1_intro import data

        train_loader = data.load_imagenette(split="train", batch_size=32)
        images, labels = data.get_sample_images(train_loader, num_samples=8)

    List available classes::

        classes = data.IMAGENETTE_CLASSES
        print(classes)  # ['tench', 'English springer', ...]

Attributes:
    IMAGENETTE_CLASSES (list): List of 10 class names in ImageNette.
    IMAGENETTE_VARIANTS (dict): Available dataset variants with URLs and metadata.
    DEFAULT_VARIANT (str): Default image size variant ('320px').

Note:
    The dataset is automatically downloaded on first use and cached locally.
"""
import os
import tarfile
import urllib.request

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import Tuple, Optional
from pathlib import Path

# ImageNette class labels (10 classes from ImageNet)
# Sorted by folder name (n0xxxxxxxx)
IMAGENETTE_CLASSES = [
    "tench",           # n01440764
    "English springer", # n02102040
    "cassette player",  # n02979186
    "chain saw",       # n03000684
    "church",          # n03028079
    "French horn",     # n03394916
    "garbage truck",   # n03417042
    "gas pump",        # n03425413
    "golf ball",       # n03445777
    "parachute",       # n03888257
]

# ImageNette URLs and folder names (from fast.ai)
IMAGENETTE_VARIANTS = {
    "160px": {
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
        "folder": "imagenette2-160",
        "description": "160px - Smallest (~25MB, fastest training)"
    },
    "320px": {
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        "folder": "imagenette2-320",
        "description": "320px - Medium (~100MB, balanced)"
    },
    "full": {
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        "folder": "imagenette2",
        "description": "Full size (~1.5GB, highest quality)"
    },
}

# Default variant
DEFAULT_VARIANT = "320px"

# Global data directory - always use project root
# This ensures data is stored at VisionInterpretability/data/ regardless of where code is run
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _download_and_extract(variant: str, data_dir: str) -> str:
    """Download and extract ImageNette dataset if not already present.

    Downloads the specified variant of the ImageNette dataset from the
    fast.ai S3 bucket and extracts it to the specified directory. If the
    dataset is already present, this function returns immediately.

    Args:
        variant: The image size variant to download. Must be one of
            '160px', '320px', or 'full'.
        data_dir: The root directory to store the downloaded and
            extracted dataset.

    Returns:
        The absolute path to the extracted dataset folder.

    Raises:
        ValueError: If an unknown variant is specified.

    Example:
        >>> path = _download_and_extract('320px', './data')
        >>> print(path)  # './data/imagenette2-320'
    """
    if variant not in IMAGENETTE_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(IMAGENETTE_VARIANTS.keys())}")
    
    info = IMAGENETTE_VARIANTS[variant]
    url = info["url"]
    folder_name = info["folder"]
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    extracted_path = data_path / folder_name
    if extracted_path.exists() and (extracted_path / "train").exists():
        return str(extracted_path)
    
    # Download
    archive_name = f"{folder_name}.tgz"
    archive_path = data_path / archive_name
    if not archive_path.exists():
        print(f"📥 Downloading ImageNette ({info['description']})...")
        print(f"   URL: {url}")
        urllib.request.urlretrieve(url, archive_path)
        print("✅ Download complete!")
    
    # Extract
    print("📦 Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_path)
    print("✅ Extraction complete!")
    
    return str(extracted_path)


def get_imagenette_transforms(
    image_size: int = 128,
    is_train: bool = True
) -> transforms.Compose:
    """Create torchvision transforms for ImageNette preprocessing.

    Generates a composition of image transforms suitable for training or
    evaluation. Training transforms include data augmentation (random crop,
    horizontal flip), while evaluation transforms use center crop only.

    Args:
        image_size: Target image size in pixels. Default is 128 for faster
            training. Use 224 for compatibility with pretrained models.
        is_train: If True, includes data augmentation transforms.
            If False, uses deterministic center crop.

    Returns:
        A ``torchvision.transforms.Compose`` object containing the
        preprocessing pipeline.

    Example:
        >>> train_transform = get_imagenette_transforms(128, is_train=True)
        >>> val_transform = get_imagenette_transforms(128, is_train=False)
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def load_imagenette(
    split: str = "train",
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    variant: str = "320px"
) -> DataLoader:
    """Load ImageNette dataset and return a PyTorch DataLoader.

    Downloads and prepares the ImageNette dataset automatically if not already
    present. Supports multiple image size variants and provides configurable
    data loading options.

    Data is always stored in the project root directory at:
    `VisionInterpretability/data/`

    Args:
        split: Dataset split to load. Use 'train' for training data or
            'validation'/'val' for validation data.
        image_size: Target image size in pixels after preprocessing.
            Default is 128 for faster training.
        batch_size: Number of samples per batch. Default is 32.
        num_workers: Number of subprocesses for data loading. Use 0 for
            main process only (safer for debugging). Default is 0.
        shuffle: Whether to shuffle the data. If None, defaults to True
            for training split and False for validation split.
        variant: Image size variant to download. Options are '160px' (~25MB),
            '320px' (~100MB), or 'full' (~1.5GB). Default is '320px'.

    Returns:
        A PyTorch DataLoader configured with the specified parameters.

    Example:
        >>> train_loader = load_imagenette(split='train', batch_size=64)
        >>> val_loader = load_imagenette(split='val', image_size=224)
        >>> for images, labels in train_loader:
        ...     print(images.shape)  # torch.Size([64, 3, 128, 128])
        ...     break
    """
    # Normalize split name
    if split in ("validation", "val", "test"):
        split = "val"
    
    is_train = (split == "train")
    
    if shuffle is None:
        shuffle = is_train
    
    # Download and extract if needed (uses global DATA_DIR)
    dataset_path = _download_and_extract(variant, str(DATA_DIR))
    split_path = os.path.join(dataset_path, split)
    
    # Get transforms
    transform = get_imagenette_transforms(image_size, is_train=is_train)
    
    # Create dataset using ImageFolder
    dataset = ImageFolder(split_path, transform=transform)
    
    print(f"✅ Loaded ImageNette {split} ({variant}): {len(dataset)} samples, {len(dataset.classes)} classes")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return dataloader


def get_sample_images(
    dataloader: DataLoader,
    num_samples: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract a batch of sample images from a DataLoader.

    Retrieves the first batch from the dataloader and returns a subset
    of images and labels for visualization or debugging purposes.

    Args:
        dataloader: A PyTorch DataLoader containing image-label pairs.
        num_samples: Number of samples to return. Must be less than or
            equal to the batch size.

    Returns:
        A tuple containing:
            - images (torch.Tensor): Tensor of shape ``(num_samples, C, H, W)``.
            - labels (torch.Tensor): Tensor of shape ``(num_samples,)``.

    Example:
        >>> loader = load_imagenette(split='train', batch_size=32)
        >>> images, labels = get_sample_images(loader, num_samples=8)
        >>> print(images.shape)  # torch.Size([8, 3, 128, 128])
    """
    images, labels = next(iter(dataloader))
    return images[:num_samples], labels[:num_samples]


def get_imagenette_classes() -> list:
    """Return the list of ImageNette class names.

    Returns:
        A list of 10 class name strings, ordered by their ImageNet
        synset IDs (folder names).

    Example:
        >>> classes = get_imagenette_classes()
        >>> print(classes[0])  # 'tench'
    """
    return IMAGENETTE_CLASSES


if __name__ == "__main__":
    print("Loading ImageNette train set...")
    train_loader = load_imagenette(split="train", batch_size=4)
    images, labels = get_sample_images(train_loader)
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels.tolist()}")
