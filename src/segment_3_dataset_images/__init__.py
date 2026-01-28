"""
Segment 3: Dataset Images for Vision Interpretability.

This package provides tools for finding diverse dataset samples that activate
specific neurons in neural networks, inspired by Distill.pub's Feature Visualization.

Modules:
    activation_pipeline: Core classes for activation extraction and spectrum tracking.
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

__all__ = [
    "ActivationExtractor",
    "ActivationSpectrumTracker",
    "ActivationSpectrumTrackerV2",
    "ImageNetStreamer",
    "WANDBExperimentLogger",
    "FeatureOptimizer",
    "SampleRecord",
    "run_pipeline",
]
