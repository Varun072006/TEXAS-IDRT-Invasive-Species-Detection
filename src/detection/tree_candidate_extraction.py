"""
Tree candidate extraction.

Converts raw SAM/segmentation mask proposals into individual tree-crown
candidates: filtering by size/shape plausibility and computing a
per-candidate centroid, radius, and confidence proxy used by later
stages (crown separation, duplicate merging, clustering).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segmentation.sam_segmentation import SegmentationMask


@dataclass
class TreeCandidate:
    pixel_x: float
    pixel_y: float
    radius_px: float
    area_px: int
    mask: np.ndarray
    shape_score: float  # 0-1, how "blob-like" (circular) the region is


def _circularity(mask: np.ndarray) -> float:
    """Rough circularity proxy: filled area vs. bounding-circle area.
    1.0 = perfect disc, lower values indicate elongated/irregular shapes
    (roads, shadows, riverbanks) that are unlikely to be individual crowns.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    cx, cy = xs.mean(), ys.mean()
    radius = np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean()) * 1.4  # ~RMS radius correction
    circle_area = np.pi * radius**2
    return float(min(mask.sum() / circle_area, 1.0)) if circle_area > 0 else 0.0


def extract_tree_candidates(
    masks: list[SegmentationMask],
    min_radius_px: float = 4.0,
    max_radius_px: float = 60.0,
    min_shape_score: float = 0.35,
) -> list[TreeCandidate]:
    """Filter raw segmentation masks down to plausible individual tree-crown candidates."""
    candidates: list[TreeCandidate] = []

    for m in masks:
        ys, xs = np.nonzero(m.mask)
        if len(xs) == 0:
            continue

        cx, cy = float(xs.mean()), float(ys.mean())
        radius = float(np.sqrt(m.area_px / np.pi))

        if not (min_radius_px <= radius <= max_radius_px):
            continue

        score = _circularity(m.mask)
        if score < min_shape_score:
            continue

        candidates.append(
            TreeCandidate(
                pixel_x=cx,
                pixel_y=cy,
                radius_px=radius,
                area_px=m.area_px,
                mask=m.mask,
                shape_score=score,
            )
        )

    return candidates
