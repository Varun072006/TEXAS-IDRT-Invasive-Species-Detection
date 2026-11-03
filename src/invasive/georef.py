"""Pixel -> ground georeferencing from PPK geotags (position + omega/phi/kappa).

Model: pinhole camera, flat ground at a fixed ellipsoidal height. Angles follow the photogrammetric
(Pix4D-style) convention: R = Rx(omega) Ry(phi) Rz(kappa) maps the photogrammetric camera frame
(x right, y up, z backwards) to a local East-North-Up frame; with all angles zero the camera looks
straight down with image-right = East and image-up = North.

STATUS: the rotation convention and focal length are NOT confirmed. Use scripts/calibrate_georef.py
(overlap consistency between consecutive frames) before trusting absolute positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer

from .flights import Geotag
from .io import upright_to_raw_xy

# computer-vision camera (x right, y down, z forward) -> photogrammetric camera (x right, y up, z back)
_CV_TO_PHOTO = np.diag([1.0, -1.0, -1.0])

ROTATION_VARIANTS = ("pix4d", "transpose", "neg_omega", "neg_phi", "neg_kappa", "neg_omega_phi")


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int              # raw sensor image size in pixels
    height: int
    focal_mm: float
    sensor_width_mm: float

    @property
    def focal_px(self) -> float:
        return self.focal_mm / self.sensor_width_mm * self.width

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0


def rotation_matrix(omega: float, phi: float, kappa: float, variant: str = "pix4d") -> np.ndarray:
    """Camera-to-world rotation (degrees in). `variant` exists so calibration can test conventions."""
    if variant == "neg_omega":
        omega = -omega
    elif variant == "neg_phi":
        phi = -phi
    elif variant == "neg_kappa":
        kappa = -kappa
    elif variant == "neg_omega_phi":
        omega, phi = -omega, -phi
    elif variant not in ("pix4d", "transpose"):
        raise ValueError(f"unknown rotation variant {variant}")
    o, p, k = np.radians([omega, phi, kappa])
    rx = np.array([[1, 0, 0], [0, np.cos(o), -np.sin(o)], [0, np.sin(o), np.cos(o)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(k), -np.sin(k), 0], [np.sin(k), np.cos(k), 0], [0, 0, 1]])
    r = rx @ ry @ rz
    return r.T if variant == "transpose" else r


def utm_crs_for(lat: float, lon: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def pixel_to_enu_offset(u, v, intr: CameraIntrinsics, rot: np.ndarray, height_agl: float) -> np.ndarray:
    """Raw pixel coordinates -> (east, north) offsets in metres from the camera, flat ground.

    Returns an (n,2) array; rays that do not hit the ground (pointing up) are NaN.
    """
    u = np.atleast_1d(np.asarray(u, float))
    v = np.atleast_1d(np.asarray(v, float))
    d_cv = np.stack([(u - intr.cx) / intr.focal_px, (v - intr.cy) / intr.focal_px, np.ones_like(u)], axis=0)
    d_world = rot @ (_CV_TO_PHOTO @ d_cv)
    down = -d_world[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(down > 1e-9, height_agl / down, np.nan)
    return np.stack([d_world[0] * t, d_world[1] * t], axis=1)


class FrameGeoreferencer:
    """Georeference pixels of one frame (given in UPRIGHT or RAW coordinates)."""

    def __init__(self, geotag: Geotag, intr: CameraIntrinsics, ground_height: float, orientation: int = 1,
                 variant: str = "pix4d", height_error_m: float = 3.0, angles_in_upright: bool = False):
        self.geotag = geotag
        self.intr = intr
        self.orientation = orientation
        # When True the geotag angles refer to the displayed (upright) image, so `intr` must describe the
        # upright image and upright pixels are used directly. Observed for EXIF-orientation-3 (DPP) flights.
        self.angles_in_upright = angles_in_upright
        self.height_agl = geotag.alt - ground_height
        if self.height_agl <= 1.0:
            raise ValueError(f"{geotag.name}: non-positive height above ground ({self.height_agl:.1f} m)")
        self.rot = rotation_matrix(geotag.omega, geotag.phi, geotag.kappa, variant)
        self.crs = utm_crs_for(geotag.lat, geotag.lon)
        self._to_utm = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._to_wgs = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self.cam_e, self.cam_n = self._to_utm.transform(geotag.lon, geotag.lat)
        self.height_error_m = height_error_m

    def upright_to_utm(self, x, y) -> np.ndarray:
        """Upright pixel coordinates -> (n,2) UTM easting/northing."""
        if self.angles_in_upright:
            return self.raw_to_utm(x, y)
        rx, ry = upright_to_raw_xy(x, y, self.intr.width, self.intr.height, self.orientation)
        return self.raw_to_utm(rx, ry)

    def raw_to_utm(self, u, v) -> np.ndarray:
        off = pixel_to_enu_offset(u, v, self.intr, self.rot, self.height_agl)
        return off + np.array([self.cam_e, self.cam_n])

    def utm_to_lonlat(self, en: np.ndarray) -> np.ndarray:
        en = np.asarray(en, float).reshape(-1, 2)
        out = np.empty_like(en)
        for i, (e, n) in enumerate(en):           # per point: avoids pyproj's size-1 array deprecation path
            out[i] = self._to_wgs.transform(float(e), float(n))
        return out

    def boxes_to_ground(self, boxes_upright: np.ndarray) -> dict:
        """Project boxes (x1,y1,x2,y2 upright pixels) to ground centres, footprints and area."""
        b = np.asarray(boxes_upright, float).reshape(-1, 4)
        corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1)  # (n,4,2)
        n = len(b)
        flat = corners.reshape(-1, 2)
        utm = self.upright_to_utm(flat[:, 0], flat[:, 1]).reshape(n, 4, 2)
        centres = self.upright_to_utm((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2)
        x, y = utm[..., 0], utm[..., 1]
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))
        return dict(centre_utm=centres, corners_utm=utm, centre_lonlat=self.utm_to_lonlat(centres), area_m2=area,
                    crs=self.crs.to_string())

    def error_estimate_m(self, ray_off_nadir_deg: float = 0.0) -> float:
        """Heuristic horizontal error: geotag accuracy combined with a height error acting through tilt.

        Not a measured value; replace with residuals against field points when available.
        """
        tilt = np.radians(abs(self.geotag.omega) + abs(self.geotag.phi) + ray_off_nadir_deg)
        return float(np.hypot(self.geotag.h_acc, self.height_error_m * np.tan(tilt)))
