"""Detection metrics per class: precision/recall/F1 at an IoU threshold, AP50 and recall by box size."""
from __future__ import annotations

import numpy as np

from .merge import box_iou


def _match(gt: np.ndarray, pred: np.ndarray, scores: np.ndarray, iou_thr: float):
    """Greedy match predictions (descending score) to ground truth. Returns tp flags in score order."""
    order = np.argsort(-scores)
    ious = box_iou(pred[order], gt) if len(gt) else np.zeros((len(order), 0))
    used = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(order), dtype=bool)
    for r in range(len(order)):
        if ious.shape[1] == 0:
            break
        cand = np.where(~used & (ious[r] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[r, cand])]
            used[j] = True
            tp[r] = True
    return order, tp


def average_precision(tp: np.ndarray, n_gt: int) -> float:
    """VOC-style AP (all-points interpolation) from TP flags sorted by descending score."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(gt_by_image: dict, pred_by_image: dict, classes: list[str], iou_thr: float = 0.5,
             score_thr: float = 0.0) -> dict:
    """gt_by_image[img] = (boxes (n,4), class_idx (n,)); pred_by_image[img] = (boxes, scores, class_idx).

    Matching is done per image and class. Returns {class: {ap50, precision, recall, f1, n_gt, n_pred}}.
    """
    out = {}
    for ci, cname in enumerate(classes):
        all_scores, all_tp, n_gt = [], [], 0
        for img in set(gt_by_image) | set(pred_by_image):
            gb, gc = gt_by_image.get(img, (np.zeros((0, 4)), np.zeros(0, int)))
            pb, ps, pc = pred_by_image.get(img, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
            g = np.asarray(gb, float).reshape(-1, 4)[np.asarray(gc) == ci]
            sel = (np.asarray(pc) == ci) & (np.asarray(ps) >= score_thr)
            p = np.asarray(pb, float).reshape(-1, 4)[sel]
            s = np.asarray(ps, float)[sel]
            n_gt += len(g)
            if len(p):
                order, tp = _match(g, p, s, iou_thr)
                all_scores.append(s[order])
                all_tp.append(tp)
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
        tp = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
        o = np.argsort(-scores)
        tp_sorted = tp[o]
        n_tp = int(tp.sum())
        precision = n_tp / len(tp) if len(tp) else float("nan")
        recall = n_tp / n_gt if n_gt else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        out[cname] = dict(ap50=average_precision(tp_sorted, n_gt), precision=precision, recall=recall, f1=f1,
                          n_gt=n_gt, n_pred=len(tp))
    return out
