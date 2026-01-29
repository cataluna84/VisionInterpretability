"""
Segment 3: Dataset Images for Vision Interpretability.

This package provides tools for finding diverse dataset samples that activate
specific neurons in neural networks, inspired by Distill.pub's Feature Visualization.

The activation spectrum approach helps understand what patterns each neuron
responds to by showing:
- Minimum (most suppressing) examples
- Slightly negative (near threshold) examples
- Slightly positive (barely activating) examples
- Maximum (most activating) examples
- Optimized (gradient ascent/descent) visualizations

Modules:
    activation_pipeline: Core classes for activation extraction and spectrum tracking.
    visualization: Distill.pub style plotting functions.

Example:
    >>> from segment_3_dataset_images import (
    ...     ActivationExtractor,
    ...     ActivationSpectrumTrackerV2,
    ...     FeatureOptimizer,
    ...     plot_neuron_spectrum_distill,
    ... )
    >>> extractor = ActivationExtractor(model, "mixed4a")
    >>> tracker = ActivationSpectrumTrackerV2(num_neurons=10)
"""

from segment_3_dataset_images.activation_pipeline import (
    ActivationExtractor,
    ActivationSpectrumTracker,
    ActivationSpectrumTrackerV2,
    ImageNetStreamer,
    WANDBExperimentLogger,
    FeatureOptimizer,
    SampleRecord,
    run_pipeline,
)

from segment_3_dataset_images.visualization import (
    plot_neuron_spectrum_distill,
)

__all__ = [
    # Activation Pipeline
    "ActivationExtractor",
    "ActivationSpectrumTracker",
    "ActivationSpectrumTrackerV2",
    "ImageNetStreamer",
    "WANDBExperimentLogger",
    "FeatureOptimizer",
    "SampleRecord",
    "run_pipeline",
    # Visualization
    "plot_neuron_spectrum_distill",
]
