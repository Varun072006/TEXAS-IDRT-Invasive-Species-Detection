"""
Duplicate merging across tiles.

Because tiles overlap (see preprocessing/tiling.py), a single tree near a
tile boundary can be detected more than once, in neighboring tiles. This
module merges near-duplicate detections in full-scene coordinates using
a greedy non-maximum-suppression pass over circular crown footprints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SceneDetection:
    scene_x: float
    scene_y: float
    radius_px: float
    shape_score: float


def _circle_iou(a: SceneDetection, b: SceneDetection) -> float:
    """Approximate IoU between two circular crown footprints."""
    d = ((a.scene_x - b.scene_x) ** 2 + (a.scene_y - b.scene_y) ** 2) ** 0.5
    r1, r2 = a.radius_px, b.radius_px

    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        smaller_area = 3.14159265 * min(r1, r2) ** 2
        larger_area = 3.14159265 * max(r1, r2) ** 2
        return smaller_area / larger_area

    # Standard circle-circle intersection area formula.
    import math

    r1_sq, r2_sq = r1**2, r2**2
    a1 = math.acos(min(1.0, max(-1.0, (d**2 + r1_sq - r2_sq) / (2 * d * r1))))
    a2 = math.acos(min(1.0, max(-1.0, (d**2 + r2_sq - r1_sq) / (2 * d * r2))))
    inter = (
        r1_sq * a1
        + r2_sq * a2
        - 0.5 * math.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    )
    union = 3.14159265 * r1_sq + 3.14159265 * r2_sq - inter
    return inter / union if union > 0 else 0.0


def merge_duplicate_detections(
    detections: list[SceneDetection],
    iou_threshold: float = 0.35,
) -> list[SceneDetection]:
    """Greedy NMS: keep the highest-confidence (shape_score) detection in each
    cluster of overlapping circular footprints, dropping the rest as duplicates
    from overlapping tiles.
    """
    ordered = sorted(detections, key=lambda d: d.shape_score, reverse=True)
    kept: list[SceneDetection] = []

    for det in ordered:
        if all(_circle_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)

    return kept
