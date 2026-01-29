"""
Distill.pub style visualization for neuron activation spectra.

This module provides functions to create publication-quality visualizations
of neuron activation spectra, matching the Distill.pub Circuits thread style.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


def plot_neuron_spectrum_distill(
    neuron_idx: int,
    layer_name: str,
    spectrum: dict,
    optimized_img: Image.Image | None = None,
    negative_optimized_img: Image.Image | None = None,
    figsize: tuple[int, int] = (24, 6),
) -> plt.Figure:
    """
    Plot activation spectrum in Distill.pub style.
    
    Layout: Neg Optimized | Minimum | Slight-Neg | Slight-Pos | Maximum | Pos Optimized
    
    Args:
        neuron_idx: Neuron channel index.
        layer_name: Name of the layer (e.g., "mixed4a").
        spectrum: Dictionary with keys 'minimum', 'slight_negative', 
                  'slight_positive', 'maximum', each containing SampleRecord lists.
        optimized_img: Positive optimized image (gradient ascent).
        negative_optimized_img: Negative optimized image (gradient descent).
        figsize: Figure size as (width, height).
        
    Returns:
        Matplotlib Figure object.
    """
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 18, figure=fig, wspace=0.1, hspace=0.1)
    
    # Column spans: optimized images = 3 cols, grids = 3 cols each
    # Total: 3 + 3 + 3 + 3 + 3 + 3 = 18 columns
    
    # 1. Negative optimized (large, spans all 3 rows)
    ax_neg_opt = fig.add_subplot(gs[:, 0:3])
    ax_neg_opt.axis('off')
    if negative_optimized_img is not None:
        ax_neg_opt.imshow(negative_optimized_img)
    ax_neg_opt.set_title('Negative\noptimized', fontsize=9)
    
    # 2. Minimum examples (3x3 grid)
    samples_min = spectrum.get('minimum', [])
    for i in range(9):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, 3 + col])
        ax.axis('off')
        if i < len(samples_min):
            sample = samples_min[i]
            if hasattr(sample, 'image') and isinstance(sample.image, Image.Image):
                ax.imshow(sample.image)
    # Add title below grid
    fig.text(0.22, 0.02, 'Minimum activation\nexamples', ha='center', fontsize=9)
    
    # 3. Slightly negative (3x3 grid)
    samples_sneg = spectrum.get('slight_negative', [])
    for i in range(9):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, 6 + col])
        ax.axis('off')
        if i < len(samples_sneg):
            sample = samples_sneg[i]
            if hasattr(sample, 'image') and isinstance(sample.image, Image.Image):
                ax.imshow(sample.image)
    fig.text(0.39, 0.02, 'Slightly negative\nactivation examples', ha='center', fontsize=9)
    
    # 4. Slightly positive (3x3 grid)
    samples_spos = spectrum.get('slight_positive', [])
    for i in range(9):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, 9 + col])
        ax.axis('off')
        if i < len(samples_spos):
            sample = samples_spos[i]
            if hasattr(sample, 'image') and isinstance(sample.image, Image.Image):
                ax.imshow(sample.image)
    fig.text(0.56, 0.02, 'Slightly positive\nactivation examples', ha='center', fontsize=9)
    
    # 5. Maximum examples (3x3 grid)
    samples_max = spectrum.get('maximum', [])
    for i in range(9):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, 12 + col])
        ax.axis('off')
        if i < len(samples_max):
            sample = samples_max[i]
            if hasattr(sample, 'image') and isinstance(sample.image, Image.Image):
                ax.imshow(sample.image)
    fig.text(0.72, 0.02, 'Maximum activation\nexamples', ha='center', fontsize=9)
    
    # 6. Positive optimized (large, spans all 3 rows)
    ax_pos_opt = fig.add_subplot(gs[:, 15:18])
    ax_pos_opt.axis('off')
    if optimized_img is not None:
        ax_pos_opt.imshow(optimized_img)
    ax_pos_opt.set_title('Positive\noptimized', fontsize=9)
    
    # Add neuron label
    fig.suptitle(f"Layer {layer_name}, unit {neuron_idx}", fontsize=12, fontweight='bold', y=0.98)
    
    return fig
