"""
High-resolution tiling.

Drone orthomosaics can span gigapixels and won't fit in GPU memory for
segmentation models like SAM. We split each orthomosaic into overlapping
tiles, run the pipeline per tile, then stitch/deduplicate detections back
into full-scene coordinates (see detection/duplicate_merging.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    image: np.ndarray
    x_offset: int  # top-left x of this tile in full-scene pixel coordinates
    y_offset: int  # top-left y of this tile in full-scene pixel coordinates


def tile_image(image: np.ndarray, tile_size: int = 1024, overlap: int = 128) -> list[Tile]:
    """Split a large image into overlapping square tiles.

    Overlap ensures tree crowns that straddle a tile boundary are fully
    captured in at least one tile; resulting duplicate detections are
    resolved later by duplicate_merging.merge_duplicate_detections().
    """
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")

    height, width = image.shape[:2]
    stride = tile_size - overlap

    tiles: list[Tile] = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile_img = image[y_start:y_end, x_start:x_end]
            tiles.append(Tile(image=tile_img, x_offset=x_start, y_offset=y_start))

            if x_end == width:
                break
        if y_end == height:
            break

    return tiles


def tile_to_scene_coords(tile: Tile, px: float, py: float) -> tuple[float, float]:
    """Map a point in tile-local pixel coordinates back to full-scene coordinates."""
    return tile.x_offset + px, tile.y_offset + py
