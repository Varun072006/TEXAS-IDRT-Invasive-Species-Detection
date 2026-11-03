"""Single tiling implementation shared by training-set creation and inference.

A tile covers a square of `src` source pixels (src = tile_px / scale) and is resized to tile_px.
Box coordinates are converted between frame space and tile space here and nowhere else.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def axis_starts(length: int, size: int, stride: int) -> list[int]:
    """Start offsets along one axis, ending with an edge-aligned tile. Handles length < size."""
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts


@dataclass(frozen=True)
class Tile:
    x: int          # top-left in frame pixels
    y: int
    src: int        # side length in frame pixels
    tile_px: int    # side length after resizing (model input)
    ix: int = 0
    iy: int = 0

    @property
    def factor(self) -> float:
        """frame pixels per tile pixel"""
        return self.src / self.tile_px


def make_grid(width: int, height: int, tile_px: int, scale: float, overlap: float) -> list[Tile]:
    if not 0 < scale <= 4:
        raise ValueError("scale must be in (0, 4]")
    src = int(round(tile_px / scale))
    stride = max(1, int(round(src * (1.0 - overlap))))
    xs = axis_starts(width, src, stride)
    ys = axis_starts(height, src, stride)
    return [Tile(x, y, src, tile_px, ix, iy) for iy, y in enumerate(ys) for ix, x in enumerate(xs)]


def extract_tile(img: np.ndarray, tile: Tile) -> np.ndarray:
    """Crop (zero-padded where the tile exceeds the frame) and resize to tile_px x tile_px."""
    h, w = img.shape[:2]
    crop = img[tile.y:tile.y + tile.src, tile.x:tile.x + tile.src]
    if crop.shape[0] != tile.src or crop.shape[1] != tile.src:
        padded = np.zeros((tile.src, tile.src) + img.shape[2:], dtype=img.dtype)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    if tile.src != tile.tile_px:
        interp = cv2.INTER_AREA if tile.src > tile.tile_px else cv2.INTER_LINEAR
        crop = cv2.resize(crop, (tile.tile_px, tile.tile_px), interpolation=interp)
    return crop


def boxes_tile_to_frame(boxes: np.ndarray, tile: Tile) -> np.ndarray:
    """(n,4) x1,y1,x2,y2 in tile pixels -> frame pixels."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = boxes * tile.factor
    out[:, [0, 2]] += tile.x
    out[:, [1, 3]] += tile.y
    return out


def boxes_frame_to_tile(boxes: np.ndarray, tile: Tile) -> np.ndarray:
    """(n,4) frame pixels -> tile pixels (unclipped)."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    boxes[:, [0, 2]] -= tile.x
    boxes[:, [1, 3]] -= tile.y
    return boxes / tile.factor


def clip_boxes_to_tile(boxes: np.ndarray, tile: Tile, frame_w: int, frame_h: int,
                       min_visible: float = 0.5):
    """Clip frame-space boxes to a tile.

    Returns (kept_tile_boxes, kept_index, partial_index):
      kept_tile_boxes  boxes (tile pixels) whose visible fraction is >= min_visible
      kept_index       indices into `boxes` of the kept boxes
      partial_index    indices of boxes that touch the tile but fall below min_visible
    A box that is only partly visible is never used as if it were a full object.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    x1 = np.maximum(boxes[:, 0], tile.x)
    y1 = np.maximum(boxes[:, 1], tile.y)
    x2 = np.minimum(boxes[:, 2], min(tile.x + tile.src, frame_w))
    y2 = np.minimum(boxes[:, 3], min(tile.y + tile.src, frame_h))
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 1e-9, None) * np.clip(boxes[:, 3] - boxes[:, 1], 1e-9, None)
    visible = inter / area
    kept = np.where(visible >= min_visible)[0]
    partial = np.where((visible > 0) & (visible < min_visible))[0]
    clipped = np.stack([x1, y1, x2, y2], axis=1)[kept]
    return boxes_frame_to_tile(clipped, tile), kept, partial
