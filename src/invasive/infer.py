"""Inference: tile -> detect -> merge -> georeference -> tier -> cross-frame de-duplication."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .config import camera_for_flight
from .export import DETECTION_COLUMNS
from .flights import Geotag, estimate_ground_height
from .georef import CameraIntrinsics, FrameGeoreferencer
from .io import check_image, read_upright
from .merge import dedup_points, nms
from .tiling import boxes_tile_to_frame, extract_tile, make_grid


class Detector(Protocol):
    def predict(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """One (n,6) array per tile: x1,y1,x2,y2 (tile px), score, class index."""


class UltralyticsDetector:
    """Thin wrapper around an Ultralytics YOLO/RT-DETR checkpoint (note the AGPL licence)."""

    def __init__(self, weights: str | Path, conf: float = 0.05, imgsz: int = 1024, device: str | None = None,
                 augment: bool = False):
        from ultralytics import YOLO  # imported lazily so tests do not need the weights

        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.device, self.augment = conf, imgsz, device, augment

    def predict(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        results = self.model.predict(tiles, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                     augment=self.augment, verbose=False)
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6)))
                continue
            out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                       b.cls.cpu().numpy()[:, None]], axis=1))
        return out


def detect_frame(img: np.ndarray, detector: Detector, cfg: dict, batch: int = 4) -> np.ndarray:
    """Detections for one upright frame as an (n,6) array in FRAME pixels: x1,y1,x2,y2,score,class."""
    t = cfg["tiling"]
    h, w = img.shape[:2]
    grid = make_grid(w, h, t["tile_px"], t["scale"], t["overlap"])
    all_boxes, all_scores, all_cls = [], [], []
    for i in range(0, len(grid), batch):
        chunk = grid[i:i + batch]
        preds = detector.predict([extract_tile(img, tile) for tile in chunk])
        for tile, p in zip(chunk, preds):
            if len(p) == 0:
                continue
            fb = boxes_tile_to_frame(p[:, :4], tile)
            fb[:, [0, 2]] = np.clip(fb[:, [0, 2]], 0, w)
            fb[:, [1, 3]] = np.clip(fb[:, [1, 3]], 0, h)
            all_boxes.append(fb)
            all_scores.append(p[:, 4])
            all_cls.append(p[:, 5].astype(int))
    if not all_boxes:
        return np.zeros((0, 6))
    boxes, scores, cls = np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_cls)
    keep = nms(boxes, scores, cls, cfg["infer"]["merge_iou"])
    return np.concatenate([boxes[keep], scores[keep, None], cls[keep, None]], axis=1)


def assign_tier(cls_name: str, score: float, thresholds: dict) -> str | None:
    th = thresholds[cls_name]
    if score >= th["high"]:
        return "high"
    if score >= th["candidate"]:
        return "candidate"
    return None


def frame_detections_table(dets: np.ndarray, image_name: str, flight_id: str, cfg: dict, geotag: Geotag | None,
                           orientation: int, ground_height: float | None, model_version: str = "") -> pd.DataFrame:
    """Turn raw frame detections into rows of the detections table (with georeferencing when possible)."""
    classes = cfg["classes"]
    rows = []
    geo = None
    ground = None
    cam = camera_for_flight(cfg, flight_id)
    if geotag is not None and ground_height is not None and cam.get("focal_mm"):
        # image size the geotag angles refer to: upright size or raw sensor size (passed in cfg["_sizes"])
        up = bool(cam.get("angles_in_upright"))
        iw, ih = cfg["_sizes"]["upright" if up else "raw"]
        intr = CameraIntrinsics(iw, ih, cam["focal_mm"], cam["sensor_width_mm"])
        geo = FrameGeoreferencer(geotag, intr, ground_height, orientation, cfg["georef"]["rotation_variant"],
                                 cfg["georef"]["height_error_m"], angles_in_upright=up)
        if len(dets):
            ground = geo.boxes_to_ground(dets[:, :4])
    for k, d in enumerate(dets):
        cname = classes[int(d[5])]
        tier = assign_tier(cname, float(d[4]), cfg["infer"]["thresholds"])
        if tier is None:
            continue
        row = dict(det_id=f"{Path(image_name).stem}_{k:03d}", image_name=image_name, flight_id=flight_id,
                   **{"class": cname}, score=float(d[4]), tier=tier, x1=d[0], y1=d[1], x2=d[2], y2=d[3],
                   lon=np.nan, lat=np.nan, utm_x=np.nan, utm_y=np.nan, utm_crs="", box_area_m2=np.nan,
                   georef_method="none", georef_error_est_m=np.nan, n_views=1, status="candidate",
                   model_version=model_version)
        if ground is not None:
            row.update(lon=ground["centre_lonlat"][k, 0], lat=ground["centre_lonlat"][k, 1],
                       utm_x=ground["centre_utm"][k, 0], utm_y=ground["centre_utm"][k, 1], utm_crs=ground["crs"],
                       box_area_m2=float(ground["area_m2"][k]), georef_method=cfg["georef"]["method"],
                       georef_error_est_m=geo.error_estimate_m())
        rows.append(row)
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def dedup_across_frames(df: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    """Collapse the same object seen in several frames (same class, within radius_m). Needs georeferencing."""
    ok = df[df.utm_x.notna()]
    rest = df[df.utm_x.isna()]
    if ok.empty:
        return df
    kept = []
    for crs, g in ok.groupby("utm_crs"):
        keep, views = dedup_points(g[["utm_x", "utm_y"]].to_numpy(), g.score.to_numpy(), g["class"].to_numpy(),
                                   radius_m)
        sub = g.iloc[keep].copy()
        sub["n_views"] = views
        kept.append(sub)
    return pd.concat(kept + [rest], ignore_index=True)


def run_flight(manifest: pd.DataFrame, geotags: dict[str, Geotag], detector: Detector, cfg: dict,
               model_version: str = "", progress=None) -> pd.DataFrame:
    """Run detection + georeferencing on every readable image of one flight manifest."""
    rows = []
    ground = estimate_ground_height(geotags, cfg["camera"]["nominal_agl_m"]) if geotags else None
    it = manifest[manifest.readable].itertuples()
    for r in (progress(list(it)) if progress else it):
        img = read_upright(r.path)
        dets = detect_frame(img, detector, cfg, batch=cfg["train"].get("batch", 4))
        chk = check_image(r.path, full_decode=False)
        local_cfg = dict(cfg)
        local_cfg["_sizes"] = {"raw": (chk.raw_width, chk.raw_height), "upright": (chk.width, chk.height)}
        rows.append(frame_detections_table(dets, r.image_name, r.flight_id, local_cfg, geotags.get(r.image_name),
                                           r.orientation, ground, model_version))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=DETECTION_COLUMNS)
    return dedup_across_frames(df, cfg["georef"]["dedup_distance_m"])
