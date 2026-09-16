"""
Visual embeddings for extracted tree crowns.

Once individual crowns are isolated (detection/), we embed each crop
into a fixed-size feature vector so visually/spectrally similar crowns
can be grouped -- an early, exploratory step towards species
identification without requiring labeled training data up front.

Production path uses a self-supervised DINO-family vision transformer
(via torch.hub) to extract embeddings. Since downloading multi-hundred-MB
pretrained weights isn't practical for a public demo repo, a lightweight
hand-crafted color/texture descriptor (`MockEmbedder`) is provided as a
drop-in stand-in so the clustering step can still be demonstrated
end-to-end offline.
"""

from __future__ import annotations

import cv2
import numpy as np

USE_MOCK_EMBEDDER = True


class DinoEmbedder:
    """Wraps a self-supervised DINO-family ViT (loaded via torch.hub) as a
    fixed-size visual feature extractor for crown crops.

    Requires `torch` + `torchvision` and internet access to fetch pretrained
    weights on first use. Not bundled with this repo.
    """

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cpu"):
        import torch

        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(device)

    def embed(self, crop_rgb: np.ndarray) -> np.ndarray:
        import torch
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        tensor = preprocess(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze(0).cpu().numpy()


class MockEmbedder:
    """Lightweight hand-crafted descriptor used for the public demo in place
    of a pretrained DINO model: a concatenation of an HSV color histogram
    and a Local Binary Pattern-style texture histogram.

    This is intentionally simple -- it approximates "crowns that look
    similar cluster together" without needing GPU weights, so the
    clustering notebook cell runs instantly on any machine.
    """

    def __init__(self, color_bins: int = 16):
        self.color_bins = color_bins

    def embed(self, crop_rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        hist = []
        for channel in range(3):
            h, _ = np.histogram(
                hsv[..., channel], bins=self.color_bins, range=(0, 255), density=True
            )
            hist.append(h)

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        texture_hist, _ = np.histogram(
            np.sqrt(grad_x**2 + grad_y**2), bins=self.color_bins, range=(0, 255), density=True
        )

        return np.concatenate(hist + [texture_hist]).astype(np.float32)


def get_default_embedder():
    """Factory returning the mock embedder for demo use, or a real DINO
    embedder if `USE_MOCK_EMBEDDER` is disabled.
    """
    if USE_MOCK_EMBEDDER:
        return MockEmbedder()
    return DinoEmbedder()
