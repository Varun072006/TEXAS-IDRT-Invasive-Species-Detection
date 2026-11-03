"""End-to-end smoke test on the real flights (skipped when the Data folder is absent).

Labels here are synthetic: the test checks plumbing (orientation, tiling, georeferencing, export),
not model quality.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from invasive.config import load_config, resolve_path
from invasive.dataset import build_manifest, spatial_block_folds, write_yolo_dataset
from invasive.export import detections_to_gdf, export_all
from invasive.flights import find_geotag_csv, load_wingtra_geotags
from invasive.infer import run_flight
from invasive.labelbox import parse_export

CFG = load_config()
ROOT = resolve_path(CFG, "data_root")
HAS_DATA = ROOT.is_dir() and any((p / "IMAGES").is_dir() for p in ROOT.iterdir())
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="no flights under data/raw (the real-data smoke tests need them)")


@pytest.fixture(scope="module")
def manifest():
    return build_manifest(ROOT, full_decode=False)


def _pick(manifest, flight_key, n):
    m = manifest[manifest.flight_id.str.contains(flight_key) & manifest.readable].sort_values("sequence")
    return m.head(n)


def test_manifest_flags_incomplete_downloads(manifest):
    assert manifest[manifest.is_az_temp].readable.eq(False).all()      # incomplete downloads are never used
    readable = manifest[manifest.readable]
    assert readable.has_geotag.all()
    harc = readable[readable.flight_id.str.contains("HARC")]
    assert set(zip(harc.width, harc.height)) == {(9504, 6336)} and set(harc.orientation) == {1}
    dpp = readable[readable.flight_id.str.contains("DPP EO")]
    assert set(dpp.orientation) == {3}


def test_folds_have_no_cross_block_leak(manifest):
    f = spatial_block_folds(manifest, 5, 6, seed=42)
    assert (f[manifest.readable] >= 0).all() and (f[~manifest.readable] == -1).all()


def test_yolo_dataset_from_real_frames(manifest, tmp_path):
    m = pd.concat([_pick(manifest, "HARC", 2), _pick(manifest, "DPP EO", 2)])
    # one tallow and one rose box per image, expressed in upright pixels
    rows = []
    for r in m.itertuples():
        rows.append(dict(label_id=f"{r.image_name}#0", image_name=r.image_name, **{"class": "chinese_tallow"}, raw_class="x",
                         x1=1000.0, y1=800.0, x2=1300.0, y2=1100.0, annotator="t", workflow_status="DONE"))
        rows.append(dict(label_id=f"{r.image_name}#1", image_name=r.image_name, **{"class": "mccartney_rose"}, raw_class="x",
                         x1=2500.0, y1=1500.0, x2=2600.0, y2=1580.0, annotator="t", workflow_status="DONE"))
    labels = pd.DataFrame(rows)
    folds = {n: i % 2 for i, n in enumerate(m.image_name)}
    yaml_path = write_yolo_dataset(m, labels, folds, val_fold=1, excluded=set(), cfg=CFG, out_dir=tmp_path,
                                   class_names=CFG["classes"], seed=1)
    assert yaml_path.exists()
    lab_files = list((tmp_path / "labels").rglob("*.txt"))
    imgs = list((tmp_path / "images").rglob("*.jpg"))
    assert len(imgs) == len(lab_files) > 0
    n_boxes = 0
    for lf in lab_files:
        for line in lf.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            assert c in ("0", "1") and all(0 <= float(v) <= 1 for v in (cx, cy, w, h))
            n_boxes += 1
    assert n_boxes >= 4          # at least the tiles that contain each image's boxes
    assert {p.parent.name for p in imgs} == {"train", "val"}


class CentreDetector:
    """Returns one box at the centre of every tile; enough to exercise merge + georeferencing."""
    def predict(self, tiles):
        return [np.array([[462.0, 462.0, 562.0, 562.0, 0.9, 0.0]]) for _ in tiles]


@pytest.mark.parametrize("flight_key", ["HARC", "DPP EO"])
def test_inference_georeferences_near_geotag(manifest, flight_key, tmp_path):
    m = _pick(manifest, flight_key, 1)
    flight = m.iloc[0].flight_id
    tags = load_wingtra_geotags(find_geotag_csv(ROOT / flight))
    cfg = dict(CFG)
    cfg["tiling"] = dict(CFG["tiling"], scale=0.25)          # one tile covers 4096 px: few tiles, fast
    df = run_flight(m, tags, CentreDetector(), cfg, model_version="fake")
    assert len(df) >= 1 and df.georef_method.eq("wingtra_ppk").all()
    tag = tags[m.iloc[0].image_name]
    # every detection lies within the frame footprint plus tilt: < 150 m from the camera position
    d_deg = np.hypot((df.lat - tag.lat) * 111_000, (df.lon - tag.lon) * 111_000 * np.cos(np.radians(tag.lat)))
    assert (d_deg < 150).all()
    assert (df.box_area_m2 > 0).all() and (df.georef_error_est_m > 0).all()
    paths = export_all(detections_to_gdf(df), tmp_path, "det", ["gpkg", "geojson", "csv"])
    assert all(p.exists() for p in paths)
