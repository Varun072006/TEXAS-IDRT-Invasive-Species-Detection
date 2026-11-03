# invasive_pipeline

Drone-based detection of Chinese Tallow and McCartney Rose from Wingtra PPK flights, with georeferenced
candidate regions for field verification. Design: `docs/INVASIVE_PIPELINE_DESIGN.md`; diagram with build status: `docs/PIPELINE_DIAGRAM.md`. Why the old drainage pipeline was not reused: `docs/INVASIVE_SPECIES_ADAPTATION_REPORT.md`.

## Quick start

```
pip install -r requirements.txt
python -m pytest tests -q                       # 44 tests; the 5 real-data smoke tests skip if data/raw has no flights
python scripts/inventory.py                     # manifest + data report (no labels needed)
python scripts/stats.py --export export.ndjson  # label statistics, size/scale advice, QA flags
python scripts/build_dataset.py --export export.ndjson --val-fold 0 [--skipped negative]
python scripts/train.py --dataset work/datasets/fold0/dataset.yaml --name fold0
python scripts/infer.py --weights work/runs/fold0/weights/best.pt --flight HARC
python scripts/calibrate_georef.py --flight HARC            # georeferencing self-check from image overlap
```
Put images and labels under `data/` (see `data/README.md`). Paths and every tunable are in `configs/default.yaml`. Generated files go to `work/` (git-ignored).

## Status

| Part | State |
|---|---|
| Config, image IO (EXIF orientation, corrupt-file detection, azcopy names), flight geotags | implemented, tested |
| Tiling with box clipping (one implementation), YOLO tile dataset writer, spatial-block folds, label QA | implemented, tested (unit + real frames with synthetic labels) |
| Labelbox NDJSON parser (current and legacy formats, class aliases) | implemented, tested on **synthetic** exports only; check against the real export |
| Merge (class-aware NMS), cross-frame dedup, metrics (AP50/P/R/F1), export (GPKG/GeoJSON/SHP/CSV) | implemented, tested |
| Georeferencing from PPK geotags (omega/phi/kappa) | implemented; convention and focal lengths validated by image overlap (below), **not yet against field points** |
| `train.py`, `infer.py` with Ultralytics | written, **never run** (no labels, no weights yet) |
| Tallow/rose model accuracy | **unknown**: no real labels have been seen |

## What the real data showed (the 3 flights now in data/raw, checked 2026-11-03)

| Flight | Frames | Camera | Size / EXIF orientation | Notes |
|---|---|---|---|---|
| HARC Map61 60 m | 99 | Sony ILX-LR1 (61 MP), 17 mm lens (in EXIF) | 9504x6336, 1 | ~1.3 cm/px at 60 m; about 80% overlap |
| DPP EO 60 m | 187 | Sony ILCE-6100 (24 MP), focal length **not in EXIF** | 6000x4000, 3 (rotated 180) | **55 files are unreadable** (incomplete `.azDownload-` downloads): 132 usable |
| DPP oblique | 250 | Sony ILCE-6100 | 6000x4000 and 4000x6000, 3 and 6 | tilt up to 56 deg; georeferencing **not validated** for this flight |

- Every frame has a PPK geotag (position, omega/phi/kappa, 0.06-0.75 m horizontal accuracy) in
  `<flight>/OUTPUT/*geotags.csv`. The camera is tilted about 13 deg off nadir in cruise, so a flat
  nadir assumption would be 10+ m off; the pipeline uses the full rotation.
- `scripts/calibrate_georef.py` (consecutive-frame ground disagreement, lower is better):
  - HARC (known 17 mm): Pix4D-style rotation 0.77 m; the five alternatives 5.5-9.0 m.
  - DPP EO: angles refer to the UPRIGHT image (orientation 3); focal length 12.0 mm gives 0.49 m
    (11.5 mm: 1.2 m, 12.5 mm: 1.6 m, 14 mm: 5.2 m). **12 mm is inferred from overlap, not read from the lens**: confirm.
- DPP EXIF dates read 2019-01-01 (camera clock unset); use the flight JSON timestamps instead.
- The 61 MP size (9504x6336) matches the sample image size you mentioned, so the Labelbox project may be HARC.
  This is a guess until the export is available.

## Known limitations

- Ground is assumed flat at (median camera height - 60 m); trees add parallax of a few metres. A DSM would fix this.
- Box area is the footprint of the detection box, not canopy cover.
- `georef_error_est_m` is a heuristic until compared with field GPS points.
- Ultralytics is AGPL-3.0.
