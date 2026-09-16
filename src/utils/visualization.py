"""Visualization helpers for inspecting pipeline outputs (tiles, masks, crowns, clusters)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def show_image(image: np.ndarray, title: str = "", ax=None):
    """Render a single RGB image with matplotlib."""
    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    if owns_fig:
        plt.show()
    return ax


def overlay_crown_detections(image: np.ndarray, detections: list[dict], title: str = "Detections"):
    """Draw circles over detected tree-crown centers for quick visual QA."""
    _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    for det in detections:
        circ = Circle(
            (det["pixel_x"], det["pixel_y"]),
            radius=det.get("radius_px", 15),
            fill=False,
            linewidth=1.5,
            edgecolor="red",
        )
        ax.add_patch(circ)
    ax.set_title(title)
    ax.axis("off")
    plt.show()


def plot_embedding_clusters(coords_2d: np.ndarray, labels: np.ndarray, title: str = "Visual embedding clusters"):
    """Scatter-plot a 2D projection of visual embeddings, colored by cluster label."""
    _, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap="tab10", s=25)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    legend = ax.legend(*scatter.legend_elements(), title="cluster", loc="best", fontsize=8)
    ax.add_artist(legend)
    plt.show()
