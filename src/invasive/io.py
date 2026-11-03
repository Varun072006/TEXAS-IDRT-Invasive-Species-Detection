"""Image reading with one canonical orientation convention (UPRIGHT) and integrity checks.

Frames from the Wingtra/Sony camera carry EXIF orientation 3 (rotated 180 degrees). Labelbox shows
the upright image, so labels live in upright pixel space. Georeferencing angles refer to the RAW
sensor image, so this module also converts between the two spaces.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# azcopy leaves temporary names like ".azDownload-<uuid>-<real name>" for incomplete downloads.
_AZ_PREFIX = re.compile(r"^\.azDownload-[0-9a-fA-F-]{36}-")


def canonical_name(name: str | Path) -> str:
    """File name without the azcopy temporary prefix."""
    return _AZ_PREFIX.sub("", Path(name).name)


def is_az_temp(name: str | Path) -> bool:
    return bool(_AZ_PREFIX.match(Path(name).name))


def list_image_files(folder: str | Path) -> list[Path]:
    """All image files in a folder (non-recursive), including azcopy temp files."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


@dataclass(frozen=True)
class ImageCheck:
    readable: bool
    width: int = 0            # upright width
    height: int = 0           # upright height
    raw_width: int = 0
    raw_height: int = 0
    orientation: int = 1
    error: str = ""


def check_image(path: str | Path, full_decode: bool = True) -> ImageCheck:
    """Open an image and (optionally) fully decode it to detect truncated files."""
    try:
        with Image.open(path) as im:
            raw_w, raw_h = im.size
            orientation = int(im.getexif().get(274, 1) or 1)
            if full_decode:
                im.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure means unusable
        return ImageCheck(readable=False, error=f"{type(exc).__name__}: {str(exc)[:80]}")
    w, h = (raw_h, raw_w) if orientation in (5, 6, 7, 8) else (raw_w, raw_h)
    return ImageCheck(True, w, h, raw_w, raw_h, orientation)


def read_upright(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 array with EXIF orientation applied."""
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def read_raw(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 array exactly as stored (no EXIF rotation)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def upright_to_raw_xy(x, y, raw_w: int, raw_h: int, orientation: int):
    """Map upright pixel coordinates to raw sensor pixel coordinates.

    Supports the EXIF orientations produced by drone cameras (1, 3, 6, 8).
    Coordinates are continuous pixel positions in [0, W] x [0, H].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if orientation == 1:
        return x, y
    if orientation == 3:  # rotated 180
        return raw_w - x, raw_h - y
    if orientation == 6:  # needs 90 CW to display: upright (x, y) -> raw (y, raw_h - x)
        return y, raw_h - x
    if orientation == 8:  # needs 90 CCW to display
        return raw_w - y, x
    raise ValueError(f"unsupported EXIF orientation {orientation}")
