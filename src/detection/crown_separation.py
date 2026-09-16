"""
Crown separation.

Dense canopy causes adjacent tree crowns to merge into a single
segmentation mask. This module applies a watershed transform driven by
a distance-map peak search to split large, multi-peaked masks back into
individual crown instances before they're treated as one tree.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from detection.tree_candidate_extraction import TreeCandidate


def separate_merged_crowns(
    candidate: TreeCandidate,
    min_peak_distance_px: int = 8,
) -> list[TreeCandidate]:
    """Split a single (possibly multi-tree) candidate mask into sub-crowns via
    marker-controlled watershed on its distance transform.

    If only one local maximum is found, the original candidate is returned
    unchanged (no split needed).
    """
    mask = candidate.mask.astype(np.uint8)
    distance = ndi.distance_transform_edt(mask)

    peaks = peak_local_max(
        distance,
        min_distance=min_peak_distance_px,
        labels=mask,
    )

    if len(peaks) <= 1:
        return [candidate]

    markers = np.zeros_like(mask, dtype=np.int32)
    for i, (py, px) in enumerate(peaks, start=1):
        markers[py, px] = i

    labels = watershed(-distance, markers, mask=mask)

    sub_candidates: list[TreeCandidate] = []
    for label_id in range(1, labels.max() + 1):
        sub_mask = labels == label_id
        area = int(sub_mask.sum())
        if area == 0:
            continue

        ys, xs = np.nonzero(sub_mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        radius = float(np.sqrt(area / np.pi))

        sub_candidates.append(
            TreeCandidate(
                pixel_x=cx,
                pixel_y=cy,
                radius_px=radius,
                area_px=area,
                mask=sub_mask,
                shape_score=candidate.shape_score,
            )
        )

    return sub_candidates
