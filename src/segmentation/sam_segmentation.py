"""
Segment Anything (SAM) wrapper for vegetation/tree-crown mask proposals.

In the full pipeline, the Segment Anything automatic mask generator
produces class-agnostic region proposals over each tile, which are then
filtered down to vegetation-like candidates (see detection/). Loading
real SAM checkpoints (hundreds of MB - GBs) is out of scope for this
public demo, so this module:

  1. Defines the real integration path (`SamCrownSegmenter`), used when
     the `segment-anything` package and a checkpoint are available.
  2. Falls back to a lightweight classical-CV mock segmenter
     (`MockCrownSegmenter`) based on color thresholding + connected
     components, so the demo notebook runs anywhere without GPU/weights.

Swap `USE_MOCK_SEGMENTER = False` and point `checkpoint_path` at a real
SAM checkpoint to run the production segmenter.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

USE_MOCK_SEGMENTER = True


@dataclass
class SegmentationMask:
    mask: np.ndarray  # boolean HxW mask
    area_px: int
    bbox_xywh: tuple[int, int, int, int]


class SamCrownSegmenter:
    """Thin wrapper around Meta's Segment Anything automatic mask generator.

    Requires `pip install segment-anything` and a downloaded checkpoint
    (e.g. sam_vit_b_01ec64.pth). Not bundled with this repo.
    """

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "cpu"):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=24,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.9,
            min_mask_region_area=150,
        )

    def segment(self, image: np.ndarray) -> list[SegmentationMask]:
        raw_masks = self.generator.generate(image)
        results = []
        for m in raw_masks:
            x, y, w, h = m["bbox"]
            results.append(
                SegmentationMask(
                    mask=m["segmentation"],
                    area_px=int(m["area"]),
                    bbox_xywh=(int(x), int(y), int(w), int(h)),
                )
            )
        return results


class MockCrownSegmenter:
    """Classical-CV stand-in for SAM, used for the public demo.

    Produces vegetation-region proposals via HSV color thresholding and
    connected-component analysis. This approximates what SAM's
    automatic mask generator + a vegetation filter would produce, at a
    fraction of the compute cost, so anyone can run the demo notebook
    without a GPU or model weights.
    """

    def __init__(self, min_area_px: int = 80):
        self.min_area_px = min_area_px

    def segment(self, image: np.ndarray) -> list[SegmentationMask]:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Broad green/olive band covering canopy hues.
        lower = np.array([25, 25, 25])
        upper = np.array([95, 255, 255])
        veg_mask = cv2.inRange(hsv, lower, upper)

        veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(veg_mask, connectivity=8)

        results = []
        for label_id in range(1, num_labels):  # skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < self.min_area_px:
                continue
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            results.append(
                SegmentationMask(
                    mask=(labels == label_id),
                    area_px=int(area),
                    bbox_xywh=(int(x), int(y), int(w), int(h)),
                )
            )
        return results


def get_default_segmenter():
    """Factory returning the mock segmenter for demo use, or a real SAM
    segmenter if `USE_MOCK_SEGMENTER` is disabled and a checkpoint is configured.
    """
    if USE_MOCK_SEGMENTER:
        return MockCrownSegmenter()
    raise RuntimeError(
        "Real SAM segmentation requires a local checkpoint path; "
        "instantiate SamCrownSegmenter(checkpoint_path=...) directly."
    )
