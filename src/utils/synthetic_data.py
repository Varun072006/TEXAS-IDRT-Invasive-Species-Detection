"""
Synthetic drone imagery generator.

This project's real training data comes from confidential drone survey
missions and cannot be published. This module generates procedurally
synthesized "orthomosaic-like" RGB imagery with randomly scattered
blob-shaped "tree crowns" so the rest of the pipeline (tiling,
segmentation, candidate extraction, clustering) can be demonstrated
end-to-end on data that is safe to share publicly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticCrown:
    """A single synthetic tree-crown blob placed on the mock orthomosaic."""

    center_xy: tuple[int, int]
    radius_px: int
    color_rgb: tuple[int, int, int]


@dataclass
class SyntheticScene:
    image: np.ndarray  # HxWx3 uint8
    crowns: list[SyntheticCrown] = field(default_factory=list)


def _random_canopy_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """Sample a plausible green/olive canopy color with mild variation."""
    green = int(rng.integers(90, 170))
    red = int(green * rng.uniform(0.45, 0.75))
    blue = int(green * rng.uniform(0.25, 0.55))
    return (red, green, blue)


def generate_synthetic_orthomosaic(
    width: int = 1024,
    height: int = 1024,
    num_crowns: int = 40,
    seed: int | None = 42,
) -> SyntheticScene:
    """Create a fake high-resolution "drone orthomosaic" with scattered tree crowns.

    Returns an HxWx3 uint8 array plus ground-truth crown metadata, used only
    for demo/testing purposes -- this is NOT real vegetation imagery.
    """
    rng = np.random.default_rng(seed)

    # Bare-earth / grassland background in a sandy-brown tone with light noise.
    base = rng.integers(120, 160, size=(height, width, 3), dtype=np.int16)
    base[..., 0] += 20  # push slightly warmer (more red) for a soil-like tone
    noise = rng.integers(-8, 8, size=(height, width, 3), dtype=np.int16)
    image = np.clip(base + noise, 0, 255).astype(np.uint8)

    yy, xx = np.mgrid[0:height, 0:width]
    crowns: list[SyntheticCrown] = []

    for _ in range(num_crowns):
        cx = int(rng.integers(40, width - 40))
        cy = int(rng.integers(40, height - 40))
        radius = int(rng.integers(12, 45))
        color = _random_canopy_color(rng)

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = dist <= radius
        # Soft-edge falloff to mimic canopy texture instead of a hard disc.
        falloff = np.clip(1.0 - (dist / max(radius, 1)), 0.0, 1.0) ** 0.5

        for c in range(3):
            image[..., c] = np.where(
                mask,
                (image[..., c] * (1 - falloff) + color[c] * falloff).astype(np.uint8),
                image[..., c],
            )

        crowns.append(SyntheticCrown(center_xy=(cx, cy), radius_px=radius, color_rgb=color))

    return SyntheticScene(image=image, crowns=crowns)
