"""Merge tile detections into frame detections (class-aware NMS) and de-duplicate across frames."""
from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (n,4) and (m,4) boxes in x1,y1,x2,y2."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.clip(union, 1e-9, None), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Class-aware greedy NMS. Returns indices of kept detections, highest score first."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, float))
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    ious = box_iou(boxes, boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        same = np.asarray(classes) == classes[i]
        suppressed |= same & (ious[i] > iou_thr)
        suppressed[i] = True
    return np.asarray(keep, dtype=int)


def dedup_points(xy: np.ndarray, scores: np.ndarray, classes: np.ndarray, radius_m: float):
    """Greedy cross-frame de-duplication of detections in a metric (UTM) plane.

    Same-class detections closer than radius_m are one object. Returns
    (keep_index, n_views) where n_views[k] counts the detections merged into kept detection k
    and keep_index lists the highest-scoring member of each group.
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    scores = np.asarray(scores, float)
    classes = np.asarray(classes)
    order = np.argsort(-scores)
    assigned = np.zeros(len(xy), dtype=bool)
    keep: list[int] = []
    views: list[int] = []
    for i in order:
        if assigned[i]:
            continue
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        group = (~assigned) & (classes == classes[i]) & (d <= radius_m)
        group[i] = True
        assigned |= group
        keep.append(int(i))
        views.append(int(group.sum()))
    return np.asarray(keep, dtype=int), np.asarray(views, dtype=int)
