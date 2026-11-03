"""Wingtra flight metadata: PPK geotags (position + omega/phi/kappa) per image."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from .io import canonical_name

_GEOTAG_COLUMNS = ["name", "lat", "lon", "alt", "omega", "phi", "kappa", "h_acc", "v_acc"]


@dataclass(frozen=True)
class Geotag:
    name: str
    lat: float
    lon: float
    alt: float        # geodetic (ellipsoidal) height of the camera, metres
    omega: float      # degrees
    phi: float
    kappa: float
    h_acc: float      # horizontal accuracy, metres
    v_acc: float


def load_wingtra_geotags(csv_path: str | Path) -> dict[str, Geotag]:
    """Parse '<flight> geotags.csv' written by WingtraHub (PPK processed). Keys are canonical names."""
    df = pd.read_csv(csv_path, header=0, names=_GEOTAG_COLUMNS, encoding="utf-8")
    out: dict[str, Geotag] = {}
    for row in df.itertuples(index=False):
        name = canonical_name(row.name)
        out[name] = Geotag(name, float(row.lat), float(row.lon), float(row.alt), float(row.omega),
                           float(row.phi), float(row.kappa), float(row.h_acc), float(row.v_acc))
    return out


def find_geotag_csv(flight_dir: str | Path) -> Path | None:
    hits = sorted(Path(flight_dir).glob("OUTPUT/*geotags.csv"))
    return hits[0] if hits else None


def frame_sequence(name: str) -> int:
    """Trailing frame number of 'XXXX_00012.JPG' (or -1)."""
    m = re.search(r"_(\d+)\.[A-Za-z]+$", canonical_name(name))
    return int(m.group(1)) if m else -1


def consecutive_spacing_m(geotags: dict[str, Geotag]) -> np.ndarray:
    """Distance in metres between consecutive frames (sorted by frame number)."""
    ordered = sorted(geotags.values(), key=lambda g: frame_sequence(g.name))
    if len(ordered) < 2:
        return np.zeros(0)
    geod = Geod(ellps="WGS84")
    lons = [g.lon for g in ordered]
    lats = [g.lat for g in ordered]
    _, _, dist = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    return np.asarray(dist)


def estimate_ground_height(geotags: dict[str, Geotag], nominal_agl_m: float) -> float:
    """Ground height on the ellipsoid, assuming the flight was flown at nominal AGL.

    Rough (flat-terrain) estimate: median camera height minus the planned height above ground.
    Replace with a DSM lookup when one is available.
    """
    return float(np.median([g.alt for g in geotags.values()]) - nominal_agl_m)
