import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from invasive.dataset import drop_buffer_frames, label_qa, spatial_block_folds
from invasive.export import DETECTION_COLUMNS, detections_to_gdf, export_all, summarise
from invasive.flights import Geotag, consecutive_spacing_m
from invasive.georef import CameraIntrinsics, FrameGeoreferencer, rotation_matrix
from invasive.infer import assign_tier, dedup_across_frames, detect_frame
from invasive.merge import box_iou, dedup_points, nms
from invasive.metrics import evaluate

INTR = CameraIntrinsics(6000, 4000, 20.0, 23.5)


def tag(**kw):
    base = dict(name="a.JPG", lat=29.7, lon=-95.4, alt=70.0, omega=0.0, phi=0.0, kappa=0.0, h_acc=0.1, v_acc=0.2)
    base.update(kw)
    return Geotag(**base)


# ---------------------------------------------------------------- georeferencing
def test_nadir_image_centre_is_below_camera():
    g = FrameGeoreferencer(tag(), INTR, ground_height=10.0)
    res = g.raw_to_utm([3000], [2000])
    assert np.allclose(res[0], [g.cam_e, g.cam_n], atol=1e-6)


def test_gsd_matches_pinhole_formula():
    g = FrameGeoreferencer(tag(), INTR, ground_height=10.0)          # 60 m above ground
    a, b = g.raw_to_utm([3000, 3100], [2000, 2000])
    gsd = np.hypot(*(b - a))
    assert gsd == pytest.approx(100 * 60.0 / INTR.focal_px, rel=1e-6)


def test_kappa_zero_means_image_up_is_north_and_right_is_east():
    g = FrameGeoreferencer(tag(), INTR, ground_height=10.0)
    c = g.raw_to_utm([3000], [2000])[0]
    up = g.raw_to_utm([3000], [1000])[0] - c
    right = g.raw_to_utm([4000], [2000])[0] - c
    assert up[1] > 0 and abs(up[0]) < 1e-6
    assert right[0] > 0 and abs(right[1]) < 1e-6


def test_kappa_180_flips_directions():
    g = FrameGeoreferencer(tag(kappa=180.0), INTR, ground_height=10.0)
    c = g.raw_to_utm([3000], [2000])[0]
    up = g.raw_to_utm([3000], [1000])[0] - c
    assert up[1] < 0


def test_tilt_shifts_footprint_by_height_times_tan():
    # a 12 degree tilt about the x axis moves the image centre by about 60 * tan(12 deg) = 12.8 m
    g = FrameGeoreferencer(tag(omega=12.0), INTR, ground_height=10.0)
    shift = g.raw_to_utm([3000], [2000])[0] - np.array([g.cam_e, g.cam_n])
    assert np.hypot(*shift) == pytest.approx(60.0 * np.tan(np.radians(12.0)), rel=1e-6)


def test_rotation_matrix_is_orthonormal_for_all_variants():
    for v in ("pix4d", "transpose", "neg_omega", "neg_phi", "neg_kappa", "neg_omega_phi"):
        r = rotation_matrix(-12.0, 1.8, -179.7, v)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_lonlat_roundtrip_distance_agrees_with_geod():
    g = FrameGeoreferencer(tag(), INTR, ground_height=10.0)
    out = g.boxes_to_ground(np.array([[2900, 1900, 3100, 2100.0]]))
    lon, lat = out["centre_lonlat"][0]
    _, _, d = Geod(ellps="WGS84").inv(-95.4, 29.7, lon, lat)
    assert d < 0.05                       # centre sits under the camera
    gsd = 60.0 / INTR.focal_px
    assert out["area_m2"][0] == pytest.approx(200 * gsd * 200 * gsd, rel=1e-3)


def test_orientation_3_inverts_pixel_mapping():
    up = FrameGeoreferencer(tag(), INTR, 10.0, orientation=3)
    raw = FrameGeoreferencer(tag(), INTR, 10.0, orientation=1)
    a = up.upright_to_utm([100.0], [50.0])
    b = raw.raw_to_utm([6000 - 100.0], [4000 - 50.0])
    assert np.allclose(a, b)


def test_ground_above_camera_raises():
    with pytest.raises(ValueError):
        FrameGeoreferencer(tag(alt=5.0), INTR, ground_height=10.0)


# ---------------------------------------------------------------- merge / dedup
def test_nms_is_class_aware():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]), 0.5)
    assert sorted(keep) == [0, 2]


def test_box_iou_values():
    assert box_iou(np.array([[0, 0, 10, 10.0]]), np.array([[0, 0, 10, 10.0]]))[0, 0] == pytest.approx(1.0)
    assert box_iou(np.array([[0, 0, 10, 10.0]]), np.array([[5, 0, 15, 10.0]]))[0, 0] == pytest.approx(1 / 3)


def test_dedup_points_counts_views_and_respects_class():
    xy = np.array([[0, 0], [0.5, 0], [10, 10], [0.2, 0.1]], float)
    keep, views = dedup_points(xy, np.array([0.5, 0.9, 0.7, 0.6]), np.array([0, 0, 0, 1]), 1.5)
    assert sorted(keep) == [1, 2, 3]
    assert views[list(keep).index(1)] == 2


def test_dedup_across_frames_keeps_ungeoreferenced_rows():
    df = pd.DataFrame([dict(zip(DETECTION_COLUMNS, [""] * len(DETECTION_COLUMNS))) for _ in range(2)])
    df["utm_x"], df["utm_y"], df["score"], df["class"] = np.nan, np.nan, [0.5, 0.6], "chinese_tallow"
    assert len(dedup_across_frames(df, 1.5)) == 2


# ---------------------------------------------------------------- detection on frames with a fake detector
class FakeDetector:
    """Reports one object at a fixed FRAME location, as seen from whichever tile contains it."""
    def __init__(self, frame_box, grid_fn):
        self.frame_box, self.grid_fn, self.calls = frame_box, grid_fn, []

    def predict(self, tiles):
        out = []
        for tile in tiles:
            out.append(np.zeros((0, 6)))
        return out


def test_detect_frame_empty_detector_returns_empty():
    from invasive.config import load_config
    cfg = load_config()
    img = np.zeros((1500, 2200, 3), np.uint8)
    dets = detect_frame(img, FakeDetector(None, None), cfg)
    assert dets.shape == (0, 6)


def test_detect_frame_merges_duplicate_from_overlapping_tiles():
    from invasive.config import load_config
    from invasive.tiling import boxes_frame_to_tile, make_grid
    cfg = load_config(overrides={"tiling": {"scale": 1.0, "overlap": 0.5}})
    img = np.zeros((1024, 1536, 3), np.uint8)
    grid = make_grid(1536, 1024, 1024, 1.0, 0.5)
    obj = np.array([[600.0, 300.0, 700.0, 400.0]])           # visible in both tiles

    class PerTile:
        def __init__(self): self.i = 0
        def predict(self, tiles):
            out = []
            for _ in tiles:
                tile = grid[self.i]; self.i += 1
                b = boxes_frame_to_tile(obj, tile)
                out.append(np.concatenate([b, [[0.8, 0]]], axis=1))
            return out

    dets = detect_frame(img, PerTile(), cfg, batch=1)
    assert len(dets) == 1 and np.allclose(dets[0, :4], obj[0], atol=1e-6)


def test_assign_tier():
    th = {"chinese_tallow": {"candidate": 0.25, "high": 0.6}}
    assert assign_tier("chinese_tallow", 0.7, th) == "high"
    assert assign_tier("chinese_tallow", 0.3, th) == "candidate"
    assert assign_tier("chinese_tallow", 0.1, th) is None


# ---------------------------------------------------------------- metrics
def test_metrics_perfect_and_missed():
    gt = {"a": (np.array([[0, 0, 10, 10.0], [50, 50, 60, 60.0]]), np.array([0, 0]))}
    pred = {"a": (np.array([[0, 0, 10, 10.0]]), np.array([0.9]), np.array([0]))}
    m = evaluate(gt, pred, ["chinese_tallow", "mccartney_rose"])
    assert m["chinese_tallow"]["precision"] == 1.0
    assert m["chinese_tallow"]["recall"] == 0.5
    assert m["chinese_tallow"]["ap50"] == pytest.approx(0.5)
    assert np.isnan(m["mccartney_rose"]["ap50"])


def test_metrics_false_positive_lowers_precision():
    gt = {"a": (np.array([[0, 0, 10, 10.0]]), np.array([0]))}
    pred = {"a": (np.array([[0, 0, 10, 10.0], [100, 100, 120, 120.0]]), np.array([0.9, 0.8]), np.array([0, 0]))}
    m = evaluate(gt, pred, ["chinese_tallow"])["chinese_tallow"]
    assert m["precision"] == 0.5 and m["recall"] == 1.0


# ---------------------------------------------------------------- splits and QA
def _manifest(n_per_flight=30):
    rows = []
    for f in ("A", "B"):
        for s in range(n_per_flight):
            rows.append(dict(image_name=f"{f}_{s:05d}.JPG", flight_id=f, sequence=s, readable=True, width=6000, height=4000))
    rows.append(dict(image_name="bad.JPG", flight_id="A", sequence=99, readable=False, width=0, height=0))
    return pd.DataFrame(rows)


def test_spatial_block_folds_keep_blocks_together_and_skip_unreadable():
    m = _manifest()
    folds = spatial_block_folds(m, n_folds=3, block_size=6, seed=1)
    assert folds[m.readable == False].eq(-1).all()
    for f, g in m[m.readable].groupby("flight_id"):
        g = g.sort_values("sequence")
        for i in range(0, len(g), 6):
            assert folds[g.index[i:i + 6]].nunique() == 1
    assert set(folds[m.readable]) == {0, 1, 2}


def test_buffer_frames_excludes_train_neighbours_of_validation():
    m = _manifest()
    folds = spatial_block_folds(m, 3, 6, seed=1)
    excl = drop_buffer_frames(m, folds, val_fold=0, buffer_frames=1)
    assert excl and all(folds[i] != 0 for i in excl)


def test_label_qa_flags():
    m = _manifest()
    labels = pd.DataFrame([
        dict(image_name="A_00000.JPG", **{"class": "chinese_tallow"}, x1=0, y1=0, x2=3, y2=3),          # tiny
        dict(image_name="A_00001.JPG", **{"class": "chinese_tallow"}, x1=0, y1=0, x2=5000, y2=3900),    # huge
        dict(image_name="A_00002.JPG", **{"class": "chinese_tallow"}, x1=10, y1=10, x2=60, y2=60),      # ok
        dict(image_name="A_00002.JPG", **{"class": "chinese_tallow"}, x1=10, y1=10, x2=60, y2=60),      # duplicate
        dict(image_name="nope.JPG", **{"class": "chinese_tallow"}, x1=10, y1=10, x2=60, y2=60),         # unknown image
    ])
    q = label_qa(labels, m)
    assert list(q.flag_tiny) == [True, False, False, False, False]
    assert list(q.flag_huge) == [False, True, False, False, False]
    assert list(q.flag_duplicate) == [False, False, False, True, False]
    assert list(q.flag_no_image) == [False, False, False, False, True]


# ---------------------------------------------------------------- flights / export
def test_consecutive_spacing_in_metres():
    tags = {f"x_{i:05d}.JPG": tag(name=f"x_{i:05d}.JPG", lat=29.7 + i * 0.00036) for i in range(3)}
    d = consecutive_spacing_m(tags)
    assert len(d) == 2 and d[0] == pytest.approx(40, abs=1.0)


def test_export_roundtrip(tmp_path):
    row = dict.fromkeys(DETECTION_COLUMNS, 0)
    row.update({"det_id": "a_000", "image_name": "a.JPG", "flight_id": "F", "class": "chinese_tallow", "score": 0.7,
                "tier": "high", "lon": -95.4, "lat": 29.7, "utm_crs": "EPSG:32615", "status": "candidate",
                "georef_method": "wingtra_ppk", "model_version": "v0", "box_area_m2": 12.5})
    df = pd.DataFrame([row])
    paths = export_all(detections_to_gdf(df), tmp_path, "det", ["gpkg", "geojson", "shp", "csv"])
    assert all(p.exists() for p in paths)
    import geopandas as gpd
    back = gpd.read_file(tmp_path / "det.gpkg")
    assert back.iloc[0]["class"] == "chinese_tallow" and back.crs.to_epsg() == 4326
    s = summarise(df)
    assert s.iloc[0].n_detections == 1


def test_angles_in_upright_uses_upright_pixels_directly():
    a = FrameGeoreferencer(tag(omega=10.0, kappa=170.0), INTR, 10.0, orientation=3, angles_in_upright=True)
    b = FrameGeoreferencer(tag(omega=10.0, kappa=170.0), INTR, 10.0, orientation=1, angles_in_upright=False)
    assert np.allclose(a.upright_to_utm([123.0], [456.0]), b.raw_to_utm([123.0], [456.0]))


def test_camera_for_flight_selects_per_flight_settings():
    from invasive.config import camera_for_flight, load_config
    cfg = load_config()
    assert camera_for_flight(cfg, "HARC Map61 60 m Flight 01")["focal_mm"] == 17.0
    dpp = camera_for_flight(cfg, "DPP EO 60 m Flight 01")
    assert dpp["angles_in_upright"] is True and dpp["sensor_width_mm"] == 23.5


def test_frame_detections_table_georeferences_when_camera_known():
    from invasive.config import load_config
    from invasive.infer import frame_detections_table
    cfg = load_config()
    cfg = dict(cfg)
    cfg["_sizes"] = {"raw": (6000, 4000), "upright": (6000, 4000)}
    dets = np.array([[2900, 1900, 3100, 2100, 0.8, 0], [100, 100, 200, 200, 0.1, 1]])   # 2nd below threshold
    df = frame_detections_table(dets, "DPP_EO_x.JPG", "DPP EO 60 m Flight 01", cfg, tag(), 3, 10.0, "t")
    assert len(df) == 1 and df.iloc[0].tier == "high"
    assert df.iloc[0].georef_method == "wingtra_ppk" and df.iloc[0].box_area_m2 > 0
    assert abs(df.iloc[0].lat - 29.7) < 1e-3
