import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from invasive.ab import CLASS_NAMES, load_items, make_folds, read_class_file, buffer_images

IMAGES = Path(__file__).resolve().parents[1] / "data" / "images"


def test_class_order_matches_classes_txt():
    assert read_class_file(IMAGES) == CLASS_NAMES == ["macartney_rose", "chinese_tallow"]


def test_label_boxes_inside_image_and_counts():
    items = load_items(IMAGES)
    assert len(items) == 97
    assert sum(int((i.cls == 0).sum()) for i in items) == 1026
    assert sum(int((i.cls == 1).sum()) for i in items) == 141


def test_every_fold_has_both_classes_and_no_leakage():
    items = load_items(IMAGES)
    f = make_folds(items)
    for k in range(5):
        its = [i for i in items if f[i.name] == k]
        assert any((i.cls == 1).any() for i in its) and any((i.cls == 0).any() for i in its)
    drop = buffer_images(items, f, {0}, 1)
    assert all(f[n] != 0 for n in drop)


def test_labels_use_raw_frame():
    from invasive.ab import read_frame
    it = load_items(IMAGES)[0]
    assert read_frame(it.path).shape[:2] == (it.h, it.w)
