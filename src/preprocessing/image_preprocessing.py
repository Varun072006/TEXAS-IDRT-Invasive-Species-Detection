"""
Drone image preprocessing.

Raw drone orthomosaics vary in brightness, haze, and color balance
depending on flight conditions (sun angle, cloud cover, sensor gain).
Before any detection/segmentation step, we normalize contrast and
denoise so downstream models see consistent input statistics.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (contrast-limited adaptive histogram equalization) on the
    luminance channel to correct uneven sunlight/shadow across a large flight area.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def denoise(image: np.ndarray, strength: int = 5) -> np.ndarray:
    """Light bilateral denoising that preserves canopy edges better than Gaussian blur."""
    return cv2.bilateralFilter(image, d=strength, sigmaColor=50, sigmaSpace=50)


def preprocess_orthomosaic(image: np.ndarray) -> np.ndarray:
    """Full preprocessing chain applied to a raw drone orthomosaic tile."""
    image = denoise(image)
    image = normalize_illumination(image)
    return image
