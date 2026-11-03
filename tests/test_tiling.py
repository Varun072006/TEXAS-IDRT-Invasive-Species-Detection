import numpy as np
import pytest

from invasive.tiling import (axis_starts, boxes_frame_to_tile, boxes_tile_to_frame, clip_boxes_to_tile,
                             extract_tile, make_grid)


def test_axis_starts_ends_edge_aligned():
    s = axis_starts(6000, 2048, 1536)
    assert s[0] == 0 and s[-1] + 2048 == 6000
    assert all(b > a for a, b in zip(s, s[1:]))


def test_axis_starts_small_image_does_not_crash():
    assert axis_starts(500, 2048, 1536) == [0]


def test_grid_covers_whole_frame():
    w, h = 6000, 4000
    grid = make_grid(w, h, 1024, 0.5, 0.25)
    cover = np.zeros((h, w), bool)
    for t in grid:
        cover[t.y:t.y + t.src, t.x:t.x + t.src] = True
    assert cover.all()
    assert grid[0].src == 2048


def test_extract_tile_pads_small_frame():
    img = np.full((300, 400, 3), 255, np.uint8)
    tile = make_grid(400, 300, 1024, 1.0, 0.25)[0]
    out = extract_tile(img, tile)
    assert out.shape == (1024, 1024, 3)
    assert out[0, 0, 0] == 255 and out[-1, -1, 0] == 0


def test_box_roundtrip():
    tile = make_grid(6000, 4000, 1024, 0.5, 0.25)[3]
    b = np.array([[10.0, 20.0, 110.0, 220.0]])
    back = boxes_frame_to_tile(boxes_tile_to_frame(b, tile), tile)
    assert np.allclose(back, b)


def test_clip_keeps_mostly_visible_and_reports_partial():
    tile = make_grid(6000, 4000, 1024, 1.0, 0.0)[0]   # covers x,y in [0,1024)
    boxes = np.array([
        [100, 100, 200, 200],      # fully inside
        [1000, 100, 1100, 200],    # 24% visible -> partial
        [900, 100, 1100, 200],     # 62% visible -> kept, clipped
        [2000, 2000, 2100, 2100],  # outside
    ], float)
    tb, kept, partial = clip_boxes_to_tile(boxes, tile, 6000, 4000, min_visible=0.5)
    assert list(kept) == [0, 2]
    assert list(partial) == [1]
    assert tb[1, 2] == pytest.approx(1024)      # clipped to the tile edge


def test_clip_respects_frame_edge_for_visibility():
    # box extends past the frame edge; the tile is edge aligned, visibility is measured against the tile
    tile = make_grid(1500, 1500, 1024, 1.0, 0.0)[-1]
    tb, kept, _ = clip_boxes_to_tile(np.array([[1400, 1400, 1500, 1500.0]]), tile, 1500, 1500)
    assert len(kept) == 1 and (tb >= 0).all() and (tb <= 1024).all()
