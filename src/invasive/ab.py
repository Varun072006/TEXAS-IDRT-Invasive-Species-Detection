"""Training/evaluation helpers for the YOLO-format dataset in data/images (AB_Oppenheimer flight).

Class order is fixed by data/images/classes.txt: 0 = macartney_rose, 1 = chinese_tallow.
Labels are normalised YOLO boxes in the RAW pixel frame (as stored in the JPG, NO EXIF rotation).
Verified empirically (scripts/ab_frame_probe.py): all 97 frames have EXIF orientation 3 (180 deg); against the
EXIF-upright image the boxes miss their targets (CLIP crop separability 47%), against the raw array it is 85%.
The provider's annotations.json and preview/ images wrongly assume the upright frame.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .tiling import Tile, clip_boxes_to_tile, extract_tile, make_grid

CLASS_NAMES = ["macartney_rose", "chinese_tallow"]  # must equal data/images/classes.txt
ROSE, TALLOW = 0, 1


def read_class_file(images_dir: Path) -> list[str]:
    return [ln.strip() for ln in (images_dir / "classes.txt").read_text().splitlines() if ln.strip()]


@dataclass
class Item:
    name: str            # file name, e.g. AB_..._00002.JPG
    path: Path
    seq: int
    w: int               # upright size
    h: int
    boxes: np.ndarray    # (n,4) x1,y1,x2,y2 raw-frame pixels
    cls: np.ndarray      # (n,) class ids

    @property
    def stem(self) -> str:
        return Path(self.name).stem


def load_items(images_dir: Path) -> list[Item]:
    images_dir = Path(images_dir)
    if read_class_file(images_dir) != CLASS_NAMES:
        raise ValueError(f"classes.txt {read_class_file(images_dir)} != {CLASS_NAMES}")
    items = []
    for p in sorted(images_dir.glob("*.JPG")):
        with Image.open(p) as im:
            w, h = im.size  # raw frame: labels are normalised against the stored pixel array
        lab = images_dir / "labels" / f"{p.stem}.txt"
        rows = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()] if lab.exists() else []
        arr = np.array([[float(v) for v in r] for r in rows]).reshape(-1, 5)
        cls = arr[:, 0].astype(int)
        cx, cy, bw, bh = arr[:, 1] * w, arr[:, 2] * h, arr[:, 3] * w, arr[:, 4] * h
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        seq = int(re.search(r"(\d+)$", p.stem).group(1))
        items.append(Item(p.name, p, seq, w, h, boxes, cls))
    return items


def read_frame(path) -> np.ndarray:
    """RGB array in the labelling (raw, un-rotated) frame."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def read_upright(path) -> np.ndarray:
    """EXIF-upright array. NOT the labelling frame for this dataset; kept for display only."""
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


# ------------------------------------------------------------------ folds
def make_folds(items: list[Item], n_folds: int = 5, block: int = 4) -> dict[str, int]:
    """Blocks of consecutive frames -> folds, balancing Tallow (rare), Rose and image counts."""
    order = sorted(items, key=lambda i: i.seq)
    blocks = [order[i:i + block] for i in range(0, len(order), block)]
    tot = np.array([sum(int((i.cls == TALLOW).sum()) for i in order), sum(int((i.cls == ROSE).sum()) for i in order),
                    len(order)], float)
    w = np.array([1.0, 0.5, 0.7])
    stats = [np.array([sum(int((i.cls == TALLOW).sum()) for i in b), sum(int((i.cls == ROSE).sum()) for i in b),
                       len(b)], float) / tot for b in blocks]
    load = np.zeros((n_folds, 3))
    fold_of_block = {}
    for k in sorted(range(len(blocks)), key=lambda k: -float(stats[k] @ w)):
        # place the block where the resulting worst-balanced fold is least bad
        f = min(range(n_folds), key=lambda f: float(((load[f] + stats[k]) @ w)))
        load[f] += stats[k]
        fold_of_block[k] = f
    return {i.name: fold_of_block[k] for k, b in enumerate(blocks) for i in b}


def buffer_images(items: list[Item], fold_of: dict[str, int], eval_folds: set[int], buffer: int = 1) -> set[str]:
    """Training frames within `buffer` frames of an evaluation frame (overlapping imagery -> leakage)."""
    order = sorted(items, key=lambda i: i.seq)
    out = set()
    for pos, it in enumerate(order):
        if fold_of[it.name] in eval_folds:
            continue
        for off in range(-buffer, buffer + 1):
            j = pos + off
            if 0 <= j < len(order) and fold_of[order[j].name] in eval_folds:
                out.add(it.name)
    return out


# ------------------------------------------------------------------ tile pool
def _lines(cls_ids, boxes, size):
    out = []
    for c, b in zip(cls_ids, boxes):
        x1, y1, x2, y2 = np.clip(b, 0, size)
        out.append(f"{int(c)} {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} "
                   f"{(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}")
    return out


def build_pool(items: list[Item], out_dir: Path, tile_px: int = 1024, scale: float = 0.5, overlap: float = 0.25,
               min_visible: float = 0.5, extra_per_tallow: int = 3, extra_per_rose: int = 0, seed: int = 42) -> pd.DataFrame:
    """Write all grid tiles + jittered box-centred extra tiles once; folds just pick subsets by name."""
    import cv2
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for it in items:
        img = read_frame(it.path)
        grid = make_grid(it.w, it.h, tile_px, scale, overlap)
        src = grid[0].src
        jit = 0.3 * src
        cands = [("g", f"{it.stem}_g{t.iy}{t.ix}", t) for t in grid]
        for k, (b, c) in enumerate(zip(it.boxes, it.cls)):
            n = extra_per_tallow if c == TALLOW else extra_per_rose
            for j in range(n):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                x = int(np.clip(cx - src / 2 + rng.uniform(-jit, jit), 0, max(0, it.w - src)))
                y = int(np.clip(cy - src / 2 + rng.uniform(-jit, jit), 0, max(0, it.h - src)))
                cands.append(("x", f"{it.stem}_x{k:03d}_{j}", Tile(x, y, src, tile_px, -1, -1)))
        for kind, name, tile in cands:
            tb, kept, partial = clip_boxes_to_tile(it.boxes, tile, it.w, it.h, min_visible)
            if len(kept) == 0 and len(partial) > 0:
                continue  # only a cut-off target here: neither a clean positive nor a clean negative
            cids = it.cls[kept]
            cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), cv2.cvtColor(extract_tile(img, tile), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_dir / "labels" / f"{name}.txt").write_text("".join(f"{ln}\n" for ln in _lines(cids, tb, tile_px)))
            rows.append(dict(tile=name, image=it.name, kind=kind, n_rose=int((cids == ROSE).sum()),
                             n_tallow=int((cids == TALLOW).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pool.csv", index=False)
    return df


def write_fold(pool_dir: Path, pool: pd.DataFrame, items: list[Item], fold_of: dict[str, int], val_fold: int,
               test_fold: int, run_dir: Path, buffer: int = 1, neg_ratio: float = 1.0, extra_per_tallow: int = 2,
               final: bool = False, seed: int = 42) -> dict:
    """Write train.txt / val.txt / dataset.yaml for one run. final=True trains on every non-test fold."""
    pool_dir, run_dir = Path(pool_dir), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    eval_folds = {test_fold} if final else {val_fold, test_fold}
    drop = buffer_images(items, fold_of, eval_folds, buffer)
    train_imgs = {i.name for i in items if fold_of[i.name] not in eval_folds and i.name not in drop}
    val_imgs = [i.name for i in items if fold_of[i.name] == (test_fold if final else val_fold)]
    tr = pool[pool.image.isin(train_imgs)].copy()
    # extras: first n per tallow box (names end _x<k>_<j>)
    ex = tr[tr.kind == "x"]
    ex = ex[ex.tile.str.extract(r"_x\d+_(\d+)$")[0].astype(int) < extra_per_tallow]
    pos = pd.concat([tr[(tr.kind == "g") & (tr.n_rose + tr.n_tallow > 0)], ex])
    neg = tr[(tr.kind == "g") & (tr.n_rose + tr.n_tallow == 0)]
    neg = neg.iloc[rng.permutation(len(neg))[:int(round(len(pos) * neg_ratio))]]
    train_tiles = pd.concat([pos, neg])
    val_tiles = pool[(pool.image.isin(val_imgs)) & (pool.kind == "g")]

    def lines(df):
        return "\n".join((pool_dir / "images" / f"{t}.jpg").as_posix() for t in df.tile) + "\n"

    (run_dir / "train.txt").write_text(lines(train_tiles))
    (run_dir / "val.txt").write_text(lines(val_tiles))
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (run_dir / "dataset.yaml").write_text(
        f"path: {run_dir.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n{names}\n")
    meta = dict(val_fold=val_fold, test_fold=test_fold, final=final, train_images=sorted(train_imgs),
                val_images=val_imgs, buffer_dropped=sorted(drop),
                train_tiles=len(train_tiles), val_tiles=len(val_tiles),
                train_boxes=dict(rose=int(train_tiles.n_rose.sum()), tallow=int(train_tiles.n_tallow.sum())),
                val_boxes=dict(rose=int(val_tiles.n_rose.sum()), tallow=int(val_tiles.n_tallow.sum())))
    (run_dir / "split.json").write_text(json.dumps(meta, indent=1))
    return meta


# ------------------------------------------------------------------ metrics (full-image level)
def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + ab[None, :] - inter, 1e-9, None)


def _ap_curve(tp: np.ndarray, conf: np.ndarray, n_gt: int):
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([]), np.array([])
    order = np.argsort(-conf)
    tp = tp[order]; fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.clip(ctp + cfp, 1e-9, None)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    x = np.linspace(0, 1, 101)
    ap = float(np.mean(np.interp(x, mrec, mpre)))  # 101-point, as COCO / Ultralytics
    return ap, rec, prec, conf[order]


def evaluate_detections(gt: dict[str, tuple[np.ndarray, np.ndarray]],
                        preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    """gt[img] = (boxes, cls); preds[img] = (boxes, scores, cls). Returns per-class and mean metrics.

    AP50 and AP50-95 per class; precision/recall/F1 at each class's best-F1 confidence (reported with
    that threshold). Matching is greedy by score, one prediction per ground-truth box.
    """
    res = {}
    for c, cname in enumerate(CLASS_NAMES):
        n_gt = sum(int((g[1] == c).sum()) for g in gt.values())
        aps = []
        for thr in np.arange(0.5, 0.96, 0.05):
            tps, confs = [], []
            for img, (gb, gc) in gt.items():
                pb, ps, pc = preds.get(img, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
                g = gb[gc == c]; m = pc == c
                b, s = pb[m], ps[m]
                o = np.argsort(-s); b, s = b[o], s[o]
                tp = np.zeros(len(b)); used = np.zeros(len(g), bool)
                if len(b) and len(g):
                    iou = _iou(b, g)
                    for i in range(len(b)):
                        j = np.argmax(np.where(used, -1, iou[i]))
                        if iou[i, j] >= thr and not used[j]:
                            tp[i] = 1; used[j] = True
                tps.append(tp); confs.append(s)
            tp = np.concatenate(tps) if tps else np.zeros(0); cf = np.concatenate(confs) if confs else np.zeros(0)
            ap, rec, prec, cs = _ap_curve(tp, cf, n_gt)
            aps.append(ap)
            if abs(thr - 0.5) < 1e-6:
                if len(cs):
                    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
                    k = int(np.argmax(f1))
                    p_, r_, t_ = float(prec[k]), float(rec[k]), float(cs[k])
                else:
                    p_ = r_ = t_ = 0.0
        res[cname] = dict(n_gt=n_gt, AP50=aps[0], AP50_95=float(np.nanmean(aps)), P=p_, R=r_, best_f1_conf=t_)
    res["mean"] = dict(AP50=float(np.nanmean([res[n]["AP50"] for n in CLASS_NAMES])),
                       AP50_95=float(np.nanmean([res[n]["AP50_95"] for n in CLASS_NAMES])),
                       P=float(np.mean([res[n]["P"] for n in CLASS_NAMES])),
                       R=float(np.mean([res[n]["R"] for n in CLASS_NAMES])),
                       min_class_AP50=float(np.nanmin([res[n]["AP50"] for n in CLASS_NAMES])))
    return res


# ------------------------------------------------------------------ SAHI inference
def sahi_predict(model_path: str, items: list[Item], scale: float = 0.5, tile_px: int = 1024, overlap: float = 0.25,
                 conf: float = 0.01, postprocess: str = "GREEDYNMM", match_metric: str = "IOS",
                 match_thr: float = 0.5, device: str = "cuda:0", imgsz: int | None = None):
    """Sliced prediction on full upright frames. Returns {image: (boxes, scores, cls)} in raw-frame pixels."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(model_path),
                                               confidence_threshold=conf, device=device, image_size=imgsz or tile_px)
    src = int(round(tile_px / scale))
    out = {}
    for it in items:
        img = read_frame(it.path)
        r = get_sliced_prediction(img, model, slice_height=src, slice_width=src, overlap_height_ratio=overlap,
                                  overlap_width_ratio=overlap, perform_standard_pred=False, postprocess_type=postprocess,
                                  postprocess_match_metric=match_metric, postprocess_match_threshold=match_thr,
                                  postprocess_class_agnostic=False, verbose=0)
        b, s, c = [], [], []
        for p in r.object_prediction_list:
            bb = p.bbox
            b.append([bb.minx, bb.miny, bb.maxx, bb.maxy]); s.append(p.score.value); c.append(p.category.id)
        out[it.name] = (np.array(b, float).reshape(-1, 4), np.array(s, float), np.array(c, int))
    return out


def gt_of(items: list[Item]) -> dict:
    return {i.name: (i.boxes, i.cls) for i in items}
