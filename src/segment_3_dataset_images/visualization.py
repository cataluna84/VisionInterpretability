"""
Distill.pub style visualization for neuron activation spectra.

This module provides functions to create publication-quality visualizations
of neuron activation spectra, matching the Distill.pub Circuits thread style.

The main visualization shows 6 columns:
    1. Negative optimized - Pattern that maximally suppresses the neuron
    2. Minimum examples - Dataset images with lowest activations
    3. Slightly negative - Near-threshold negative activations
    4. Slightly positive - Near-threshold positive activations
    5. Maximum examples - Dataset images with highest activations
    6. Positive optimized - Pattern that maximally activates the neuron

Example:
    >>> from segment_3_dataset_images.visualization import plot_neuron_spectrum_distill
    >>> fig = plot_neuron_spectrum_distill(
    ...     neuron_idx=0,
    ...     layer_name="mixed4a",
    ...     spectrum=tracker.get_spectrum(0),
    ...     optimized_img=pos_img,
    ...     negative_optimized_img=neg_img,
    ... )
    >>> plt.show()

References:
    - Olah et al., "Feature Visualization", Distill, 2017
    - Olah et al., "The Building Blocks of Interpretability", Distill, 2018
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from PIL import Image


def plot_neuron_spectrum_distill(
    neuron_idx: int,
    layer_name: str,
    spectrum: dict,
    optimized_img: "Image.Image | None" = None,
    negative_optimized_img: "Image.Image | None" = None,
    figsize: tuple[int, int] = (24, 6),
) -> plt.Figure:
    """
    Plot activation spectrum in Distill.pub style.

    Creates a 6-column visualization showing the full activation spectrum
    of a single neuron, from maximally suppressing to maximally activating
    patterns.

    Layout:
        | Neg Opt | Min Grid | Slight-Neg | Slight-Pos | Max Grid | Pos Opt |

    Args:
        neuron_idx: The index of the neuron channel being visualized.
        layer_name: Name of the layer (e.g., "mixed4a", "mixed5a").
        spectrum: Dictionary containing sample lists for each category.
            Expected keys: 'minimum', 'slight_negative', 'slight_positive',
            'maximum'. Each value should be a list of SampleRecord objects.
        optimized_img: PIL Image of the positive optimized visualization
            (gradient ascent result). Displayed in rightmost column.
        negative_optimized_img: PIL Image of the negative optimized
            visualization (gradient descent result). Displayed in leftmost column.
        figsize: Figure size as (width, height) in inches. Default (24, 6)
            provides good visibility for the 6-column layout.

    Returns:
        matplotlib.figure.Figure: The generated figure object. Can be saved
            with fig.savefig() or displayed with plt.show().

    Raises:
        KeyError: If spectrum dict is missing required category keys.
        TypeError: If images are not PIL Image objects.

    Example:
        >>> from segment_3_dataset_images import (
        ...     ActivationSpectrumTrackerV2,
        ...     FeatureOptimizer,
        ...     plot_neuron_spectrum_distill,
        ... )
        >>> tracker = ActivationSpectrumTrackerV2(num_neurons=10)
        >>> # ... run activation tracking ...
        >>> optimizer = FeatureOptimizer(model)
        >>> pos_img = optimizer.optimize_neuron("mixed4a", 0)
        >>> neg_img = optimizer.optimize_neuron_negative("mixed4a", 0)
        >>> fig = plot_neuron_spectrum_distill(
        ...     neuron_idx=0,
        ...     layer_name="mixed4a",
        ...     spectrum=tracker.get_spectrum(0),
        ...     optimized_img=pos_img,
        ...     negative_optimized_img=neg_img,
        ... )
        >>> fig.savefig("neuron_0_spectrum.png", dpi=150, bbox_inches='tight')

    Note:
        The visualization expects 9 samples per category to fill the 3x3 grids.
        If fewer samples are available, empty grid cells will be displayed.
    """
    # Import PIL here to avoid circular imports
    from PIL import Image

    # Create figure with GridSpec including separator columns
    # Layout: NegOpt(3) | Sep | Min(3) | Sep | BelowMed(3) | Sep | AboveMed(3) | Sep | Max(3) | Sep | PosOpt(3)
    # Total: 6 regions × 3 cols + 5 separators = 23 columns
    # Rows: 4 (3 for images, 1 for bottom captions)
    fig = plt.figure(figsize=figsize)
    width_ratios = [1, 1, 1, 0.15,  # Neg Opt (0-2) + Sep1 (3)
                    1, 1, 1, 0.15,  # Minimum (4-6) + Sep2 (7)
                    1, 1, 1, 0.15,  # Below Median (8-10) + Sep3 (11)
                    1, 1, 1, 0.15,  # Above Median (12-14) + Sep4 (15)
                    1, 1, 1, 0.15,  # Maximum (16-18) + Sep5 (19)
                    1, 1, 1]        # Pos Opt (20-22)
    height_ratios = [1, 1, 1, 0.4]  # 3 image rows + 1 caption row
    gs = GridSpec(4, 23, figure=fig, wspace=0.05, hspace=0.1, 
                 width_ratios=width_ratios, height_ratios=height_ratios)

    # Create separator axes (gray vertical bars) - span all 4 rows
    sep_color = '#E0E0E0'
    for sep_col in [3, 7, 11, 15, 19]:
        ax_sep = fig.add_subplot(gs[:, sep_col])
        ax_sep.set_facecolor(sep_color)
        ax_sep.set_xticks([])
        ax_sep.set_yticks([])
        for spine in ax_sep.spines.values():
            spine.set_visible(False)

    # 1. Negative optimized (large, spans top 3 rows) - cols 0-2
    ax_neg_opt = fig.add_subplot(gs[0:3, 0:3])
    ax_neg_opt.axis('off')
    if negative_optimized_img is not None:
        ax_neg_opt.imshow(negative_optimized_img)
    ax_neg_opt.set_title('Negative\noptimized', fontsize=14)

    # 2. Minimum examples (3x3 grid) - cols 4-6
    samples_min = spectrum.get('minimum', [])
    for i in range(9):
        row, col = i // 3, i % 3
        # Use only top 3 rows for grid
        ax = fig.add_subplot(gs[row, 4 + col])
        ax.axis('off')
        if i < len(samples_min):
            sample = samples_min[i]
            if hasattr(sample, 'image') and isinstance(sample.image, Image.Image):
                ax.imshow(sample.image)
    # Caption in 4th row
    ax_cap = fig.add_subplot(gs[3, 4:7])
    ax_cap.axis('off')
    ax_cap.text(0.5, 0.8, 'Minimum activation\nexamples', 
                ha='center', va='top', fontsize=14)

    # 3. Below median (3x3 grid) - cols 8-10
    samples_sneg = spectrum.get('slight_negative', [])
    for i in range(9):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, 8 + col])
        ax.axis('off')
        if i < len(samples_sneg):
            sample = samples_sneg[i]
            if hasattr(sample, 'image') and isinstance(sample.image, Image.Image):
                ax.imshow(sample.image)
    # Caption in 4th row
    ax_cap = fig.add_subplot(gs[3, 8:11])
    ax_cap.axis('off')
    ax_cap.text(0.5, 0.8, 'Below median\n(near boundary)', 
                ha='center', va='top', fontsize=14)

    # 4. Above median (3x3 grid) - cols 12-14
    samples_spos = spectrum.get('slight_positive', [])
    for i in range(9):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, 12 + col])
        ax.axis('off')
        if i < len(samples_spos):
            sample = samples_spos[i]
            if hasattr(sample, 'image') and isinstance(sample.image, Image.Image):
                ax.imshow(sample.image)
    # Caption in 4th row
    ax_cap = fig.add_subplot(gs[3, 12:15])
    ax_cap.axis('off')
    ax_cap.text(0.5, 0.8, 'Above median\n(near boundary)', 
                ha='center', va='top', fontsize=14)

    # 5. Maximum examples (3x3 grid) - cols 16-18
    samples_max = spectrum.get('maximum', [])
    for i in range(9):
        row, col = i // 3, i % 3
        ax = fig.add_subplot(gs[row, 16 + col])
        ax.axis('off')
        if i < len(samples_max):
            sample = samples_max[i]
            if hasattr(sample, 'image') and isinstance(sample.image, Image.Image):
                ax.imshow(sample.image)
    # Caption in 4th row
    ax_cap = fig.add_subplot(gs[3, 16:19])
    ax_cap.axis('off')
    ax_cap.text(0.5, 0.8, 'Maximum activation\nexamples', 
                ha='center', va='top', fontsize=14)

    # 6. Positive optimized (large, spans top 3 rows) - cols 20-22
    ax_pos_opt = fig.add_subplot(gs[0:3, 20:23])
    ax_pos_opt.axis('off')
    if optimized_img is not None:
        ax_pos_opt.imshow(optimized_img)
    ax_pos_opt.set_title('Positive\noptimized', fontsize=14)

    # Add neuron label as figure title
    fig.suptitle(
        f"Layer {layer_name}, unit {neuron_idx}",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig
