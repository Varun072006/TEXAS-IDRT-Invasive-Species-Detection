import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from PIL import Image, ImageOps

from invasive.exif_fix import read_exif_orientation, set_exif_orientation


def make_jpeg(orientation, size=(60, 40)):
    rng = np.random.default_rng(1)
    im = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    ex = Image.Exif(); ex[0x0112] = orientation
    buf = io.BytesIO(); im.save(buf, "JPEG", exif=ex.tobytes(), quality=90)
    return buf.getvalue()


@pytest.mark.parametrize("o", [1, 3, 6, 8])
def test_orientation_reset_is_lossless_and_minimal(o):
    data = make_jpeg(o)
    assert read_exif_orientation(data) == o
    new, old, pos = set_exif_orientation(data, 1)
    assert old == o and len(new) == len(data)
    diff = [i for i in range(len(data)) if data[i] != new[i]]
    assert set(diff) <= {pos, pos + 1}                       # only the two tag bytes may differ
    a, b = Image.open(io.BytesIO(data)), Image.open(io.BytesIO(new))
    assert a.size == b.size                                   # stored grid unchanged (no w/h swap)
    assert np.array_equal(np.asarray(a), np.asarray(b))       # pixels identical
    assert b.getexif().get(0x0112) == 1
    assert np.array_equal(np.asarray(ImageOps.exif_transpose(b)), np.asarray(a))  # no rotation applied any more


def test_no_exif_is_untouched():
    buf = io.BytesIO(); Image.new("RGB", (8, 8)).save(buf, "JPEG")
    new, old, pos = set_exif_orientation(buf.getvalue())
    assert old is None and pos is None and new == buf.getvalue()


def test_exif_transpose_rotates_orientation3_but_not_after_fix():
    data = make_jpeg(3)
    raw = np.asarray(Image.open(io.BytesIO(data)))
    rot = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))))
    assert np.array_equal(rot, raw[::-1, ::-1])               # 180 deg: what the old preview code did
    fixed = set_exif_orientation(data)[0]
    assert np.array_equal(np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(fixed)))), raw)
