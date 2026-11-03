import json

import numpy as np
from PIL import Image

from invasive.config import build_alias_map, canonical_class, load_config
from invasive.io import (canonical_name, check_image, is_az_temp, read_upright, upright_to_raw_xy)
from invasive.labelbox import parse_export


def test_canonical_name_strips_azcopy_prefix():
    n = ".azDownload-788786c5-49e0-3748-76bc-feab14012f64-DPP_EO_60_m_Flight_01_00003.JPG"
    assert canonical_name(n) == "DPP_EO_60_m_Flight_01_00003.JPG"
    assert is_az_temp(n) and not is_az_temp("DPP_EO_60_m_Flight_01_00003.JPG")


def test_alias_handles_macartney_spelling():
    cfg = load_config()
    amap = build_alias_map(cfg)
    assert canonical_class("Macartney Rose", amap) == "mccartney_rose"
    assert canonical_class("McCartney Rose", amap) == "mccartney_rose"
    assert canonical_class("Chinese Tallow", amap) == "chinese_tallow"
    assert canonical_class("Torpedo Grass", amap) is None


def test_check_image_detects_truncated_file(tmp_path):
    good = tmp_path / "good.jpg"
    Image.fromarray(np.random.randint(0, 255, (200, 300, 3), np.uint8)).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(good.read_bytes()[:500])
    assert check_image(good).readable and check_image(good).width == 300
    assert not check_image(bad).readable


def _save_with_orientation(path, orientation):
    arr = np.zeros((40, 60, 3), np.uint8)
    arr[5, 10] = (255, 0, 0)     # a single marker pixel in raw space
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[274] = orientation
    im.save(path, exif=exif, quality=100, subsampling=0)
    return arr


def test_upright_to_raw_matches_pillow_for_each_orientation(tmp_path):
    for orientation in (1, 3, 6, 8):
        p = tmp_path / f"o{orientation}.jpg"
        _save_with_orientation(p, orientation)
        up = read_upright(p)
        ys, xs = np.where(up[..., 0] > 128)
        ux, uy = xs[0] + 0.5, ys[0] + 0.5          # pixel centre in upright space
        rx, ry = upright_to_raw_xy(ux, uy, 60, 40, orientation)
        assert (int(rx), int(ry)) == (10, 5), orientation


def test_parse_current_export_with_skipped_and_unknown_class(tmp_path):
    cfg = load_config()
    rows = [
        {"data_row": {"id": "a", "external_id": "img1.JPG"}, "media_attributes": {"width": 6000, "height": 4000},
         "projects": {"p": {"project_details": {"workflow_status": "DONE"}, "labels": [{"label_details": {"created_by": "u"},
          "annotations": {"objects": [
              {"name": "Chinese Tallow", "feature_id": "f1", "bounding_box": {"top": 10, "left": 20, "height": 30, "width": 40}},
              {"name": "Macartney Rose", "feature_id": "f2", "bounding_box": {"top": 1, "left": 2, "height": 3, "width": 4}},
              {"name": "Torpedo Grass", "bounding_box": {"top": 1, "left": 2, "height": 3, "width": 4}}]}}]}}},
        {"data_row": {"id": "b", "external_id": "img2.JPG"}, "media_attributes": {},
         "projects": {"p": {"project_details": {"workflow_status": "SKIPPED"}, "labels": []}}},
    ]
    p = tmp_path / "export.ndjson"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    labels, images, warnings = parse_export(p, cfg)
    assert list(labels["class"]) == ["chinese_tallow", "mccartney_rose"]
    t = labels.iloc[0]
    assert (t.x1, t.y1, t.x2, t.y2) == (20, 10, 60, 40)
    assert images.set_index("image_name").loc["img2.JPG", "skipped"]
    assert any("Torpedo Grass" in w for w in warnings)


def test_parse_legacy_export(tmp_path):
    cfg = load_config()
    row = {"External ID": "x.JPG", "Skipped": False,
           "Label": {"objects": [{"title": "Chinese Tallow", "bbox": {"top": 0, "left": 0, "height": 10, "width": 10}}]}}
    p = tmp_path / "legacy.ndjson"
    p.write_text(json.dumps(row), encoding="utf-8")
    labels, images, _ = parse_export(p, cfg)
    assert len(labels) == 1 and images.iloc[0].n_objects == 1
