"""
Activation Pipeline for InceptionV1 Mixed4a Layer.

This module provides tools to find diverse dataset samples showing the full
activation spectrum of neurons, inspired by Distill.pub's Feature Visualization.

Classes:
    ActivationExtractor: Extract activations from a layer via forward hooks.
    ActivationSpectrumTracker: Track min, slight-, slight+, max samples per neuron.
    ImageNetStreamer: Stream ImageNet-1k from HuggingFace.
    WANDBExperimentLogger: Log spectrum visualizations to Weights & Biases.
    FeatureOptimizer: Generate optimized examples via gradient ascent.

Example:
    >>> from segment_3_dataset_images.activation_pipeline import (
    ...     ActivationExtractor, ActivationSpectrumTracker
    ... )
    >>> extractor = ActivationExtractor(model, "mixed4a")
    >>> tracker = ActivationSpectrumTracker(num_neurons=10, samples_per_category=4)
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Optional imports with graceful fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from lucent.optvis import render
    from lucent.modelzoo import inceptionv1
    LUCENT_AVAILABLE = True
except ImportError:
    LUCENT_AVAILABLE = False


@dataclass
class SampleRecord:
    """Record of a sample and its activation value.
    
    Attributes:
        activation: The activation value for a specific neuron.
        image: PIL Image or tensor of the sample.
        image_id: Unique identifier or index of the sample.
        label: ImageNet class label (if available).
    """
    activation: float
    image: Image.Image | torch.Tensor
    image_id: int
    label: int | None = None
    
    def __lt__(self, other: SampleRecord) -> bool:
        """Enable sorting by activation value."""
        return self.activation < other.activation


class ActivationExtractor:
    """Extract activations from a specific layer using forward hooks.
    
    Uses PyTorch's register_forward_hook() to capture intermediate
    activations without modifying the model architecture.
    
    Args:
        model: PyTorch model to extract activations from.
        layer_name: Name of the layer to hook (e.g., "mixed4a").
        
    Example:
        >>> model = inceptionv1(pretrained=True)
        >>> extractor = ActivationExtractor(model, "mixed4a")
        >>> output = model(input_tensor)
        >>> activations = extractor.get_activations()  # (B, 512, H, W)
    """
    
    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.activations: torch.Tensor | None = None
        self._hook_handle = None
        self._register_hook()
    
    def _register_hook(self) -> None:
        """Register forward hook on the specified layer."""
        layer = self._get_layer(self.model, self.layer_name)
        if layer is None:
            raise ValueError(f"Layer '{self.layer_name}' not found in model")
        
        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            self.activations = output.detach()
        
        self._hook_handle = layer.register_forward_hook(hook_fn)
    
    def _get_layer(self, model: nn.Module, layer_name: str) -> nn.Module | None:
        """Get layer by name, supporting nested attributes."""
        parts = layer_name.split(".")
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current
    
    def get_activations(self) -> torch.Tensor | None:
        """Return the captured activations from the last forward pass."""
        return self.activations
    
    def get_max_activations_per_channel(self) -> torch.Tensor | None:
        """Return global max-pooled activations (B, C) for each channel."""
        if self.activations is None:
            return None
        # Global max pool over spatial dimensions
        return self.activations.amax(dim=(2, 3))
    
    def remove_hook(self) -> None:
        """Remove the registered hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class ActivationSpectrumTracker:
    """Track diverse samples across the activation spectrum per neuron.
    
    For each neuron, maintains samples in 4 categories:
    - Minimum: Most negative/suppressing activations
    - Slightly Negative: Around 25th percentile of negatives
    - Slightly Positive: Around 25th percentile of positives
    - Maximum: Most positive/activating samples
    
    Uses sorted insertion for efficient O(log k) updates.
    
    Args:
        num_neurons: Number of neuron channels to track.
        samples_per_category: Number of samples to keep per category.
        
    Example:
        >>> tracker = ActivationSpectrumTracker(num_neurons=10, samples_per_category=4)
        >>> tracker.update(activations, images, image_ids, labels)
        >>> spectrum = tracker.get_spectrum(neuron_idx=0)
    """
    
    def __init__(self, num_neurons: int = 10, samples_per_category: int = 4):
        self.num_neurons = num_neurons
        self.samples_per_category = samples_per_category
        
        # For each neuron, track all samples seen (sorted by activation)
        # We'll extract categories at the end for efficiency
        self._all_samples: list[list[SampleRecord]] = [[] for _ in range(num_neurons)]
        
        # Track min/max for efficient pruning
        self._min_samples: list[list[SampleRecord]] = [[] for _ in range(num_neurons)]
        self._max_samples: list[list[SampleRecord]] = [[] for _ in range(num_neurons)]
        
        # Statistics for percentile calculation
        self._activation_sum: list[float] = [0.0] * num_neurons
        self._activation_count: list[int] = [0] * num_neurons
        self._all_activations: list[list[float]] = [[] for _ in range(num_neurons)]
    
    def update(
        self,
        activations: torch.Tensor,
        images: list[Image.Image],
        image_ids: list[int],
        labels: list[int] | None = None,
    ) -> None:
        """Update tracker with a batch of samples.
        
        Args:
            activations: Tensor of shape (B, num_neurons) with activation values.
            images: List of PIL Images corresponding to each sample.
            image_ids: Unique identifiers for each sample.
            labels: Optional ImageNet class labels.
        """
        batch_size = activations.shape[0]
        if labels is None:
            labels = [None] * batch_size
        
        activations_np = activations.cpu().numpy()
        
        for b in range(batch_size):
            for n in range(self.num_neurons):
                act_val = float(activations_np[b, n])
                record = SampleRecord(
                    activation=act_val,
                    image=images[b],
                    image_id=image_ids[b],
                    label=labels[b],
                )
                
                # Track statistics
                self._activation_sum[n] += act_val
                self._activation_count[n] += 1
                self._all_activations[n].append(act_val)
                
                # Update min samples (keep lowest k)
                self._update_sorted_list(
                    self._min_samples[n], record, keep_lowest=True
                )
                
                # Update max samples (keep highest k)
                self._update_sorted_list(
                    self._max_samples[n], record, keep_lowest=False
                )
    
    def _update_sorted_list(
        self,
        samples: list[SampleRecord],
        record: SampleRecord,
        keep_lowest: bool,
    ) -> None:
        """Insert record maintaining sorted order, keeping only k samples."""
        k = self.samples_per_category
        
        # Binary search insertion
        idx = bisect.bisect_left(samples, record)
        samples.insert(idx, record)
        
        # Prune to keep only k samples
        if len(samples) > k:
            if keep_lowest:
                samples.pop()  # Remove highest
            else:
                samples.pop(0)  # Remove lowest
    
    def get_spectrum(self, neuron_idx: int) -> dict[str, list[SampleRecord]]:
        """Get the activation spectrum for a specific neuron.
        
        Returns:
            Dictionary with keys: 'minimum', 'slight_negative', 
            'slight_positive', 'maximum'
        """
        if neuron_idx >= self.num_neurons:
            raise ValueError(f"Neuron index {neuron_idx} out of range")
        
        all_acts = np.array(self._all_activations[neuron_idx])
        if len(all_acts) == 0:
            return {
                "minimum": [],
                "slight_negative": [],
                "slight_positive": [],
                "maximum": [],
            }
        
        # Calculate percentile thresholds
        negative_acts = all_acts[all_acts < 0]
        positive_acts = all_acts[all_acts >= 0]
        
        slight_neg_threshold = np.percentile(negative_acts, 75) if len(negative_acts) > 0 else 0
        slight_pos_threshold = np.percentile(positive_acts, 25) if len(positive_acts) > 0 else 0
        
        # Get samples near the thresholds (we need to track these differently)
        # For now, return min/max and indicate thresholds
        return {
            "minimum": self._min_samples[neuron_idx].copy(),
            "slight_negative": [],  # TODO: Track during streaming
            "slight_positive": [],  # TODO: Track during streaming  
            "maximum": self._max_samples[neuron_idx].copy(),
            "thresholds": {
                "slight_neg": slight_neg_threshold,
                "slight_pos": slight_pos_threshold,
            },
        }
    
    def get_all_spectrums(self) -> dict[int, dict[str, list[SampleRecord]]]:
        """Get activation spectrums for all tracked neurons."""
        return {n: self.get_spectrum(n) for n in range(self.num_neurons)}


class ActivationSpectrumTrackerV2:
    """Improved tracker that tracks all 4 categories during streaming.
    
    Maintains separate heaps for each category to enable efficient
    tracking of slight-negative and slight-positive samples.
    """
    
    def __init__(self, num_neurons: int = 10, samples_per_category: int = 4):
        self.num_neurons = num_neurons
        self.k = samples_per_category
        
        # For each neuron, maintain 4 sorted lists
        self.minimum: list[list[SampleRecord]] = [[] for _ in range(num_neurons)]
        self.slight_negative: list[list[SampleRecord]] = [[] for _ in range(num_neurons)]
        self.slight_positive: list[list[SampleRecord]] = [[] for _ in range(num_neurons)]
        self.maximum: list[list[SampleRecord]] = [[] for _ in range(num_neurons)]
        
        # Track running statistics for adaptive thresholds
        self._neg_count = [0] * num_neurons
        self._pos_count = [0] * num_neurons
    
    def update(
        self,
        activations: torch.Tensor,
        images: list[Image.Image],
        image_ids: list[int],
        labels: list[int] | None = None,
    ) -> None:
        """Update all categories with batch of samples."""
        batch_size = activations.shape[0]
        if labels is None:
            labels = [None] * batch_size
        
        activations_np = activations.cpu().numpy()
        
        for b in range(batch_size):
            for n in range(min(self.num_neurons, activations_np.shape[1])):
                act_val = float(activations_np[b, n])
                record = SampleRecord(
                    activation=act_val,
                    image=images[b],
                    image_id=image_ids[b],
                    label=labels[b],
                )
                
                # Always update min (lowest activations)
                self._insert_min(self.minimum[n], record)
                
                # Always update max (highest activations)
                self._insert_max(self.maximum[n], record)
                
                # Track slight categories based on sign
                if act_val < 0:
                    self._neg_count[n] += 1
                    # Slight negative: closest to 0 among negatives (highest negative)
                    self._insert_max(self.slight_negative[n], record)
                else:
                    self._pos_count[n] += 1
                    # Slight positive: closest to 0 among positives (lowest positive)
                    self._insert_min(self.slight_positive[n], record)
    
    def _insert_min(self, samples: list[SampleRecord], record: SampleRecord) -> None:
        """Keep k samples with lowest activation values."""
        idx = bisect.bisect_left(samples, record)
        if idx < self.k:
            samples.insert(idx, record)
            if len(samples) > self.k:
                samples.pop()
    
    def _insert_max(self, samples: list[SampleRecord], record: SampleRecord) -> None:
        """Keep k samples with highest activation values."""
        idx = bisect.bisect_left(samples, record)
        samples.insert(idx, record)
        if len(samples) > self.k:
            samples.pop(0)
    
    def get_spectrum(self, neuron_idx: int) -> dict[str, list[SampleRecord]]:
        """Get the full activation spectrum for a neuron."""
        return {
            "minimum": sorted(self.minimum[neuron_idx], key=lambda x: x.activation),
            "slight_negative": sorted(self.slight_negative[neuron_idx], key=lambda x: x.activation),
            "slight_positive": sorted(self.slight_positive[neuron_idx], key=lambda x: x.activation),
            "maximum": sorted(self.maximum[neuron_idx], key=lambda x: x.activation, reverse=True),
        }


class ImageNetStreamer:
    """Stream ImageNet-1k from HuggingFace in batches.
    
    Uses the datasets library with streaming=True to avoid downloading
    the full 150GB+ dataset.
    
    Args:
        batch_size: Number of samples per batch.
        transform: Optional torchvision transform to apply.
        max_samples: Maximum number of samples to stream (None for all).
        
    Example:
        >>> streamer = ImageNetStreamer(batch_size=64, max_samples=10000)
        >>> for images, labels, ids in streamer:
        ...     # Process batch
    """
    
    def __init__(
        self,
        batch_size: int = 64,
        transform: Callable | None = None,
        max_samples: int | None = None,
        split: str = "train",
    ):
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.split = split
        
        # Default transform for InceptionV1
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform
        
        self._dataset = None
    
    def _load_dataset(self):
        """Load the streaming dataset."""
        self._dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split=self.split,
            streaming=True,
        )
    
    def __iter__(self):
        """Iterate over batches of (images, labels, ids)."""
        if self._dataset is None:
            self._load_dataset()
        
        batch_images = []
        batch_tensors = []
        batch_labels = []
        batch_ids = []
        
        sample_count = 0
        
        for idx, sample in enumerate(self._dataset):
            if self.max_samples and sample_count >= self.max_samples:
                break
            
            try:
                image = sample["image"]
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                tensor = self.transform(image)
                label = sample.get("label", None)
                
                batch_images.append(image.copy())
                batch_tensors.append(tensor)
                batch_labels.append(label)
                batch_ids.append(idx)
                sample_count += 1
                
                if len(batch_tensors) >= self.batch_size:
                    yield (
                        torch.stack(batch_tensors),
                        batch_images.copy(),
                        batch_labels.copy(),
                        batch_ids.copy(),
                    )
                    batch_images.clear()
                    batch_tensors.clear()
                    batch_labels.clear()
                    batch_ids.clear()
                    
            except Exception as e:
                # Skip corrupted images
                continue
        
        # Yield remaining samples
        if batch_tensors:
            yield (
                torch.stack(batch_tensors),
                batch_images,
                batch_labels,
                batch_ids,
            )


class WANDBExperimentLogger:
    """Handle WANDB logging for activation spectrum experiments.
    
    Logs:
    - Training progress metrics
    - Activation spectrum grids per neuron
    - Summary tables with all top samples
    
    Args:
        project: WANDB project name.
        run_name: Name for this run.
        config: Experiment configuration dict.
        
    Example:
        >>> logger = WANDBExperimentLogger("vision-interp", "mixed4a-spectrum")
        >>> logger.log_progress(batch_idx=10, samples_processed=640)
        >>> logger.log_spectrum_grid(neuron_idx=0, spectrum_dict)
    """
    
    def __init__(
        self,
        project: str = "vision-interpretability",
        run_name: str | None = None,
        config: dict | None = None,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb required: pip install wandb")
        
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
        )
    
    def log_progress(self, batch_idx: int, samples_processed: int, **kwargs) -> None:
        """Log batch progress metrics."""
        wandb.log({
            "batch": batch_idx,
            "samples_processed": samples_processed,
            **kwargs,
        })
    
    def log_spectrum_grid(
        self,
        neuron_idx: int,
        spectrum: dict[str, list[SampleRecord]],
        optimized_image: Image.Image | None = None,
    ) -> None:
        """Log activation spectrum grid for a neuron."""
        images_dict = {}
        
        for category, samples in spectrum.items():
            if category == "thresholds":
                continue
            for i, sample in enumerate(samples[:4]):  # Max 4 per category
                if isinstance(sample.image, Image.Image):
                    images_dict[f"neuron_{neuron_idx}/{category}_{i}"] = wandb.Image(
                        sample.image,
                        caption=f"act={sample.activation:.3f}, label={sample.label}"
                    )
        
        if optimized_image is not None:
            images_dict[f"neuron_{neuron_idx}/optimized"] = wandb.Image(
                optimized_image,
                caption="Gradient ascent optimized"
            )
        
        wandb.log(images_dict)
    
    def log_summary_table(
        self,
        all_spectrums: dict[int, dict[str, list[SampleRecord]]],
    ) -> None:
        """Log summary table with all neurons and their top samples."""
        columns = ["Neuron", "Category", "Activation", "Label", "Image"]
        table = wandb.Table(columns=columns)
        
        for neuron_idx, spectrum in all_spectrums.items():
            for category, samples in spectrum.items():
                if category == "thresholds":
                    continue
                for sample in samples:
                    if isinstance(sample.image, Image.Image):
                        table.add_data(
                            neuron_idx,
                            category,
                            sample.activation,
                            sample.label,
                            wandb.Image(sample.image),
                        )
        
        wandb.log({"activation_spectrum_summary": table})
    
    def finish(self) -> None:
        """Finish the WANDB run."""
        wandb.finish()


class FeatureOptimizer:
    """Generate optimized examples via gradient ascent using torch-lucent.
    
    Creates synthetic images that maximally activate specific neurons,
    providing the "ideal" pattern each neuron seeks.
    
    Args:
        model: InceptionV1 model from lucent.modelzoo.
        device: Torch device for computation.
        
    Example:
        >>> optimizer = FeatureOptimizer(model)
        >>> optimized_img = optimizer.optimize_neuron("mixed4a", channel=0)
    """
    
    def __init__(self, model: nn.Module, device: torch.device | None = None):
        if not LUCENT_AVAILABLE:
            raise ImportError("torch-lucent required: pip install torch-lucent")
        
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
    
    def optimize_neuron(
        self,
        layer_name: str,
        channel: int,
        image_size: int = 224,
        steps: int = 512,
        show_progress: bool = False,
    ) -> Image.Image:
        """Generate optimized image for a specific neuron channel.
        
        Args:
            layer_name: Layer name (e.g., "mixed4a").
            channel: Channel index within the layer.
            image_size: Size of generated image.
            steps: Number of optimization steps.
            show_progress: Whether to show optimization progress.
            
        Returns:
            PIL Image of the optimized visualization.
        """
        objective = f"{layer_name}:{channel}"
        
        result = render.render_vis(
            self.model,
            objective,
            show_image=False,
            show_inline=False,
            thresholds=(steps,),
        )
        
        # Result is a list of numpy arrays
        if result and len(result) > 0:
            img_array = result[0][0]  # First threshold, first image
            # Convert from float [0,1] to uint8
            img_array = (img_array * 255).astype(np.uint8)
            return Image.fromarray(img_array)
        
        return None


def run_pipeline(
    num_neurons: int = 10,
    samples_per_category: int = 4,
    max_samples: int = 10000,
    batch_size: int = 64,
    wandb_project: str = "vision-interpretability",
    wandb_run_name: str = "mixed4a-spectrum",
    generate_optimized: bool = True,
    device: torch.device | None = None,
) -> dict:
    """Run the complete activation spectrum pipeline.
    
    Args:
        num_neurons: Number of neurons to track (first N channels).
        samples_per_category: Samples per category per neuron.
        max_samples: Maximum samples to process from ImageNet.
        batch_size: Batch size for streaming.
        wandb_project: WANDB project name.
        wandb_run_name: Name for this experiment run.
        generate_optimized: Whether to generate optimized examples.
        device: Torch device (defaults to CUDA if available).
        
    Returns:
        Dictionary containing the activation spectrums for all neurons.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Initialize WANDB
    config = {
        "num_neurons": num_neurons,
        "samples_per_category": samples_per_category,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "layer": "mixed4a",
        "model": "InceptionV1",
    }
    logger = WANDBExperimentLogger(wandb_project, wandb_run_name, config)
    
    # Load model
    print("Loading InceptionV1...")
    model = inceptionv1(pretrained=True)
    model.to(device).eval()
    
    # Setup activation extraction
    extractor = ActivationExtractor(model, "mixed4a")
    
    # Setup spectrum tracker
    tracker = ActivationSpectrumTrackerV2(num_neurons, samples_per_category)
    
    # Setup data streaming
    print(f"Streaming up to {max_samples} samples from ImageNet-1k...")
    streamer = ImageNetStreamer(batch_size=batch_size, max_samples=max_samples)
    
    # Process batches
    total_processed = 0
    for batch_idx, (tensors, images, labels, ids) in enumerate(tqdm(streamer)):
        tensors = tensors.to(device)
        
        with torch.no_grad():
            _ = model(tensors)
        
        # Get max activations per channel
        activations = extractor.get_max_activations_per_channel()
        
        # Update tracker (only first num_neurons channels)
        tracker.update(
            activations[:, :num_neurons],
            images,
            ids,
            labels,
        )
        
        total_processed += len(tensors)
        
        # Log progress every 10 batches
        if batch_idx % 10 == 0:
            logger.log_progress(batch_idx, total_processed)
    
    print(f"Processed {total_processed} samples")
    
    # Generate optimized examples
    optimized_images = {}
    if generate_optimized:
        print("Generating optimized examples...")
        optimizer = FeatureOptimizer(model, device)
        for n in tqdm(range(num_neurons)):
            try:
                optimized_images[n] = optimizer.optimize_neuron("mixed4a", n)
            except Exception as e:
                print(f"Failed to optimize neuron {n}: {e}")
                optimized_images[n] = None
    
    # Log results to WANDB
    print("Logging results to WANDB...")
    all_spectrums = {}
    for n in range(num_neurons):
        spectrum = tracker.get_spectrum(n)
        all_spectrums[n] = spectrum
        logger.log_spectrum_grid(n, spectrum, optimized_images.get(n))
    
    logger.log_summary_table(all_spectrums)
    
    # Cleanup
    extractor.remove_hook()
    logger.finish()
    
    print("Pipeline complete!")
    return all_spectrums
