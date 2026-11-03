"""Manifest, leakage-safe splits, label QA and YOLO tile dataset creation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .flights import find_geotag_csv, frame_sequence, load_wingtra_geotags
from .io import canonical_name, check_image, is_az_temp, list_image_files, read_upright
from .tiling import Tile, clip_boxes_to_tile, extract_tile, make_grid

MANIFEST_COLUMNS = ["image_name", "flight_id", "path", "is_az_temp", "readable", "error", "width", "height",
                    "orientation", "sequence", "has_geotag", "lat", "lon", "alt", "omega", "phi", "kappa",
                    "h_acc"]


# ------------------------------------------------------------------ manifest
def build_manifest(data_root: str | Path, full_decode: bool = True) -> pd.DataFrame:
    """One row per image across all flights under data_root. Unreadable files are kept and flagged."""
    data_root = Path(data_root)
    rows: list[dict] = []
    for flight_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and (p / "IMAGES").is_dir()):
        csv_path = find_geotag_csv(flight_dir)
        tags = load_wingtra_geotags(csv_path) if csv_path else {}
        for path in list_image_files(flight_dir / "IMAGES"):
            name = canonical_name(path)
            chk = check_image(path, full_decode=full_decode)
            tag = tags.get(name)
            rows.append(dict(
                image_name=name, flight_id=flight_dir.name, path=str(path), is_az_temp=is_az_temp(path),
                readable=chk.readable, error=chk.error, width=chk.width, height=chk.height,
                orientation=chk.orientation, sequence=frame_sequence(name), has_geotag=tag is not None,
                lat=tag.lat if tag else np.nan, lon=tag.lon if tag else np.nan, alt=tag.alt if tag else np.nan,
                omega=tag.omega if tag else np.nan, phi=tag.phi if tag else np.nan,
                kappa=tag.kappa if tag else np.nan, h_acc=tag.h_acc if tag else np.nan))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


# ------------------------------------------------------------------ splits
def spatial_block_folds(manifest: pd.DataFrame, n_folds: int, block_size: int, seed: int = 42) -> pd.Series:
    """Assign each image a fold id using contiguous blocks of frames along each flight.

    Consecutive frames overlap heavily, so a random image split puts near-duplicate pixels in train
    and validation. Whole blocks of consecutive frames go to one fold instead. Returns a Series
    (index = manifest index) of fold ids; unreadable images get -1.
    """
    rng = np.random.default_rng(seed)
    folds = pd.Series(-1, index=manifest.index, dtype=int)
    block_id = 0
    block_of: dict[int, int] = {}
    for _, g in manifest[manifest.readable].groupby("flight_id"):
        order = g.sort_values("sequence").index.to_list()
        for i in range(0, len(order), block_size):
            for idx in order[i:i + block_size]:
                block_of[idx] = block_id
            block_id += 1
    blocks = np.arange(block_id)
    rng.shuffle(blocks)
    fold_of_block = {int(b): i % n_folds for i, b in enumerate(blocks)}
    for idx, b in block_of.items():
        folds[idx] = fold_of_block[b]
    return folds


def drop_buffer_frames(manifest: pd.DataFrame, folds: pd.Series, val_fold: int, buffer_frames: int) -> set[int]:
    """Train-image indices that sit within `buffer_frames` frames of a validation frame (to be excluded)."""
    excluded: set[int] = set()
    if buffer_frames <= 0:
        return excluded
    for _, g in manifest[manifest.readable].groupby("flight_id"):
        g = g.sort_values("sequence")
        idxs = g.index.to_list()
        for pos, idx in enumerate(idxs):
            if folds[idx] != val_fold:
                continue
            for off in range(-buffer_frames, buffer_frames + 1):
                j = pos + off
                if 0 <= j < len(idxs) and folds[idxs[j]] != val_fold:
                    excluded.add(idxs[j])
    return excluded


# ------------------------------------------------------------------ label QA
def label_qa(labels: pd.DataFrame, manifest: pd.DataFrame, min_side_px: float = 8, max_frac: float = 0.5) -> pd.DataFrame:
    """Per-box QA flags: tiny, huge, outside image, label for unknown/unreadable image, duplicate."""
    m = manifest.drop_duplicates("image_name").set_index("image_name")
    out = labels.copy()
    w = out.x2 - out.x1
    h = out.y2 - out.y1
    out["flag_tiny"] = (w < min_side_px) | (h < min_side_px)
    img_w = out.image_name.map(m.width)
    img_h = out.image_name.map(m.height)
    out["flag_huge"] = (w * h) > (max_frac * img_w * img_h)
    out["flag_outside"] = (out.x1 < -1) | (out.y1 < -1) | (out.x2 > img_w + 1) | (out.y2 > img_h + 1)
    out["flag_no_image"] = ~out.image_name.isin(m.index)
    readable = out.image_name.map(m.readable)
    out["flag_unreadable_image"] = readable.isna() | (readable == False)  # noqa: E712
    out["flag_duplicate"] = out.duplicated(["image_name", "class", "x1", "y1", "x2", "y2"], keep="first")
    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    out["any_flag"] = out[flag_cols].any(axis=1)
    return out


# ------------------------------------------------------------------ YOLO tile dataset
def _yolo_line(cls_id: int, box: np.ndarray, size: int) -> str:
    x1, y1, x2, y2 = np.clip(box, 0, size)
    return f"{cls_id} {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} {(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}"


def tiles_for_image(img: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray, cfg: dict, rng: np.random.Generator,
                    class_names: list[str]) -> tuple[list[tuple[Tile, np.ndarray, list[str]]], list[tuple[Tile, np.ndarray, list[str]]]]:
    """Positive and negative tiles for one image: lists of (tile, tile_image, yolo_lines).

    Tiles that only contain partially visible boxes are discarded (never labelled as full objects and
    never used as negatives, since they do contain the target).
    """
    t = cfg["tiling"]
    h, w = img.shape[:2]
    grid = make_grid(w, h, t["tile_px"], t["scale"], t["overlap"])
    src = grid[0].src
    extra = cfg["dataset"].get("extra_tiles_per_box", {})
    jitter = 0.3 * src

    candidates = list(grid)
    for box, cid in zip(boxes, class_ids):
        for _ in range(int(extra.get(class_names[int(cid)], 0))):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            x = int(np.clip(cx - src / 2 + rng.uniform(-jitter, jitter), 0, max(0, w - src)))
            y = int(np.clip(cy - src / 2 + rng.uniform(-jitter, jitter), 0, max(0, h - src)))
            candidates.append(Tile(x, y, src, t["tile_px"], -1, -1))

    pos, neg = [], []
    for tile in candidates:
        tb, kept, partial = clip_boxes_to_tile(boxes, tile, w, h, t["min_visible"])
        if len(kept) == 0 and len(partial) > 0:
            continue
        lines = [_yolo_line(int(class_ids[k]), b, tile.tile_px) for b, k in zip(tb, kept)]
        (pos if lines else neg).append((tile, None, lines))
    # materialise images lazily only for the tiles we keep
    return pos, neg


def write_yolo_dataset(manifest: pd.DataFrame, labels: pd.DataFrame, fold_of_image: dict[str, int], val_fold: int,
                       excluded: set[str], cfg: dict, out_dir: str | Path, class_names: list[str], seed: int = 42) -> Path:
    """Write images/labels for train/val and a dataset.yaml for Ultralytics. Returns the yaml path."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    cid_of = {c: i for i, c in enumerate(class_names)}
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    by_image = {k: g for k, g in labels.groupby("image_name")}
    neg_ratio = cfg["dataset"]["neg_tile_ratio"]
    usable = manifest[manifest.readable].drop_duplicates("image_name")
    for row in usable.itertuples():
        if row.image_name in excluded or row.image_name not in fold_of_image:
            continue
        split = "val" if fold_of_image[row.image_name] == val_fold else "train"
        g = by_image.get(row.image_name)
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(float) if g is not None else np.zeros((0, 4))
        cids = np.array([cid_of[c] for c in g["class"]], dtype=int) if g is not None else np.zeros(0, dtype=int)
        img = read_upright(row.path)
        pos, neg = tiles_for_image(img, boxes, cids, cfg, rng, class_names)
        n_neg = min(len(neg), int(round(len(pos) * neg_ratio))) if pos else min(len(neg), 2)
        neg = [neg[i] for i in rng.permutation(len(neg))[:n_neg]]
        stem = Path(row.image_name).stem
        for k, (tile, _, lines) in enumerate(pos + neg):
            name = f"{stem}_t{k:03d}"
            tile_img = extract_tile(img, tile)
            cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"), cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_dir / "labels" / split / f"{name}.txt").write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")

    yaml_path = out_dir / "dataset.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_path.write_text(f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
                         encoding="utf-8")
    return yaml_path


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
