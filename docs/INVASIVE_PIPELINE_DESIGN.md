# Invasive Species Detection Pipeline: Design (v0.1)

Phase 1 targets: **Chinese Tallow** (tree) and **McCartney Rose** (shrub), detected from drone RGB frames, delivered as georeferenced candidate regions for field verification.
Phase 2+ (extensible, not built now): Torpedo Grass, crown masks, multi-date monitoring.

Data has not arrived yet. Every value that depends on the data is marked **[DATA]** and is resolved by the first statistics run (section 4.1), not guessed now.

---

## 1. Scope and assumptions

| Item | Status |
|---|---|
| Inputs | Individual drone frames (.jpg/.png), about 9504x6336 px, flown at about 61 m |
| Sensor | Assumed RGB only. **Unconfirmed** (camera, lens, GSD from EXIF) |
| Labels | Labelbox bounding boxes. 97 images done, 1,167 boxes (rose 1,026, tallow 141), single annotator, no QA score |
| Output | Georeferenced candidate regions (GeoPackage / GeoJSON / Shapefile / CSV), area and coverage summary, verification status |
| Principle | Support treatment planning and field verification; never claim confirmed presence |

Non-goals for v1: species-level pixel masks, canopy cover % (boxes overstate area), real-time inference, Torpedo Grass.

### Key design decisions (with reasons)

1. **Object detection, not segmentation.** Labels are rectangles. This also matches the "candidate region" deliverable. Masks can be added later via box-prompted SAM, with human review.
2. **Detect on original frames, not on a stitched mosaic** (phase 1). Labels were drawn on frames, so training and inference see the same pixels. Georeferencing is handled as a separate, swappable stage (section 5).
3. **No water-proximity hard filter.** Tallow also grows in wetlands and riparian zones; McCartney Rose can occur along ditches. Context layers may only adjust confidence, never delete detections. (The old SAM 3 water stage is dropped for phase 1.)
4. **One shared package, config-driven.** Replaces the eight copies of model code in the drainage repo.
5. **Image- or site-level splits only.** No random tile splits.

---

## 2. Architecture

```
Labelbox export ──► [A] Dataset builder ──► manifest + labels ──► [B] Training ──► model + calibration
                                                                        │
New flights (frames) ─► [C] Ingest/QC ─► [D] Sliced inference ─► [E] Merge + calibrate
                                                                        │
                                          [F] Georeference + dedup ◄────┘
                                                    │
                          [G] Export (GPKG/GeoJSON/SHP/CSV + previews + summary)
                                                    │
                          [H] Verification (candidate/verified/rejected) ─► feedback to [A]
```

Cross-cutting: config file, logging, per-stage timing and VRAM (reuse the telemetry from `evaluate_pipeline.py`), tests, data versioning.

---

## 3. Data contracts

### 3.1 `manifest.csv` (one row per image)
`image_id, path, site, flight_id, capture_date, width, height, lat, lon, alt_m, yaw, gimbal_pitch, camera_model, focal_mm, gsd_cm, split, is_negative, skipped_reason`

`gsd_cm` is computed from EXIF (altitude, focal length, sensor size) or overridden in config.

### 3.2 `labels.parquet` (one row per box, pixel coordinates in the original frame)
`label_id, image_id, class, x1, y1, x2, y2, annotator, labelbox_status, qa_status`

Classes are an ordered list in config, so adding Torpedo Grass later means one config line plus new labels.

### 3.3 `detections` table (inference output, one row per detection)
`det_id, image_id, class, score, x1, y1, x2, y2, tile_id, lon, lat, utm_x, utm_y, box_area_m2, georef_method, georef_error_est_m, status, reviewer, reviewed_at, model_version`

`status` is one of `candidate | verified | rejected`. `georef_method` is one of `exif_flat | odm_camera | mosaic`.

---

## 4. Stage design

### 4.1 [A] Dataset builder (Labelbox to training set)
1. Pull the NDJSON export (or use the API) and convert to `labels.parquet`.
2. Read EXIF for every image and fill `manifest.csv`.
3. **Statistics report (runs first, drives all [DATA] choices):** boxes per image and class; box width/height in px and in metres; image size and GSD; list of images with zero boxes; the 34 skipped images and their reasons; open issues; duplicates and near-duplicates; overlap between consecutive frames.
4. **Negatives:** images skipped because there is no target are kept as `is_negative=1`. Skipped-because-unclear images are excluded. Add hard negatives later from reviewed false positives.
5. **Splits:** group by `site` (preferred) or `flight_id` or image. With about 97 images, use grouped 5-fold cross-validation for model selection, and keep a held-out test group if there are at least two sites **[DATA]**. Tiles from one image never cross a split.
6. **Label QA:** flag boxes that are tiny, huge, overlapping heavily across classes, or on image edges; sample 10–20% of tallow boxes for second-reviewer check.
7. Write versioned outputs (`dataset_v001/`) with a hash of the export.

### 4.2 [B] Training
- **Tiling for training:** square tiles of `tile_px` at a configured `scale` (a downscale factor applied to the frame before cutting). Both **[DATA]**: chosen so a typical target spans about 30–300 px in the model input.
  - Boxes clipped by a tile edge are kept if at least 50% of their area remains, otherwise the region is ignored (or the tile is dropped). Never train on a half-box as a full object.
  - Tiles with no boxes are sampled at a fixed ratio (default 1:1 to positive tiles) to teach negatives.
  - **Oversample tallow tiles** (141 boxes vs 1,026) with sampling weights, not duplication of the whole dataset.
- **Per-class scale (optional):** if tallow crowns and rose patches differ greatly in size, train and run two scale passes (for example tallow at 0.25x, rose at 0.5x) and merge. Decided from the statistics report.
- **Model:** pretrained detector fine-tuned on tiles. Candidates behind one interface: Ultralytics YOLO11 or RT-DETR (fast to iterate; note AGPL licence), RF-DETR / torchvision Faster R-CNN / MMDetection (permissive licences). Start with one baseline and compare after.
- **Augmentation:** flips, 90-degree rotations (nadir imagery has no orientation prior), scale jitter, colour/brightness/hue jitter, blur, copy-paste of tallow instances onto other tiles, mosaic. Keep hue jitter modest because colour is a key species cue.
- **Loss/imbalance:** class-weighted or focal classification loss plus tile oversampling.
- **Selection metric:** per-class AP50 and recall at a fixed precision on the held-out fold, not the mean over classes.
- **Calibration:** fit per-class score calibration (temperature or isotonic) on validation data, and choose two operating thresholds per class (section 4.5).
- **Tracking:** log config, data hash, git commit, metrics and sample predictions for each run (MLflow or Weights and Biases).

### 4.3 [C] Ingest and QC (new flights)
- Validate readable file, EXIF present, GPS present, orientation normalised once (read EXIF orientation; no 180-degree hacks).
- Compute GSD per frame; warn if it differs from the training range by more than a set factor, because scale shift degrades accuracy. **[DATA]**
- Flag blur, overexposure and heavy shadow frames; they are still processed but marked.

### 4.4 [D] Sliced inference
- Slice the frame with the same `scale` and `tile_px` as training, overlap about 25%, including the final edge-aligned tile (logic reused from the existing tilers; also handle images smaller than the tile).
- Batch inference in fp16, resumable per image (skip if output exists), run tiles in memory (no JPEG patches written to disk).
- Test-time augmentation (flip) as an option.

### 4.5 [E] Merge and thresholds
- Map tile boxes back to frame coordinates, merge duplicates from overlapping tiles with NMS or weighted box fusion, class-aware.
- Two operating points per class (calibrated on validation):
  - **candidate** threshold: high recall, goes to the review queue.
  - **high-confidence** threshold: high precision, prioritised for field crews.
  Exact values are set from field cost trade-offs, not fixed now.
- Optional soft context adjustment (for example a habitat or water layer) changes the score, never removes the detection.

### 4.6 [F] Georeferencing and cross-frame dedup
Behind one interface `Georeferencer`, three implementations:

| Method | Needs | Accuracy | When |
|---|---|---|---|
| `exif_flat` | EXIF GPS, altitude, yaw, camera intrinsics, nadir assumption, flat ground | roughly GPS error plus tilt and terrain error (metres) | Phase 1: fast candidate points |
| `odm_camera` | Photogrammetry output (WebODM camera models and DSM) | decimetre-level with good overlap | Phase 2: treatment planning |
| `mosaic` | Orthomosaic GeoTIFF | same as the mosaic | If you decide to infer on mosaics |

- Project box centre and corners to UTM (zone estimated automatically with pyproj), then to WGS84. Compute `box_area_m2` from projected corners.
- Store `georef_error_est_m` so users see how much to trust the position.
- **Cross-frame dedup:** the same tree appears in several overlapping frames. Cluster detections of the same class in world coordinates (distance or IoU of projected boxes), keep the best-scoring one and record how many frames supported it (`n_views`). Multiple views also raise confidence.
- Open requirement: confirm that flights have enough overlap (for photogrammetry about 70–75% forward and side) and whether GPS is RTK or standard.

### 4.7 [G] Outputs
Per flight or site:
- `detections.gpkg` (primary), `detections.geojson`, `detections.shp` (field names are truncated to 10 characters), `detections.csv`.
- Preview images: frame with boxes and labels, plus a map overview.
- `summary.csv`: per class, counts by threshold tier, total box area m² (labelled as candidate-region area, not canopy cover), frames processed, georef method, model version, thresholds, processing time.
- `run_manifest.json`: config, data hash, model version, git commit.
- CRS: store WGS84 for exchange, plus the local UTM coordinates for area.

### 4.8 [H] Verification and feedback loop
- Review in QGIS, QField (field phones) or back into Labelbox as pre-labels.
- Reviewers set `status` and optionally correct the box. Confirmed false positives become hard negatives, confirmed misses become new labels.
- Retrain periodically. Keep the original test group fixed so progress is comparable.
- Optional: box-prompted SAM 3 to propose crown masks for reviewers to accept or edit, giving true canopy area later.

---

## 5. Evaluation plan

| Level | Metric | Notes |
|---|---|---|
| Detector | AP50 and AP50:95 per class; precision, recall, F1 at both thresholds; PR curves; confusion between rose, tallow and background | On held-out images/sites only |
| Scale | Recall by box size bucket | Shows if small shrubs or big crowns are missed |
| Operational | Per-flight false positives per hectare; per-image recall | Closest to field workload |
| Geospatial | Distance from detection to field-verified point (m) | Needs GPS ground-truth points from the field |
| Label quality | Agreement between two reviewers on a sample | Establishes the noise ceiling |
| Efficiency | Seconds per frame, VRAM, frames per hour | Reuse existing benchmark harness; measure on at least 20 frames |

Acceptance gates (for example "tallow recall at least X at precision at least Y") are to be set with the field team after the baseline; no numbers are assumed here.

---

## 6. Configuration (example)

```yaml
project: invasive_v1
classes: [chinese_tallow, mccartney_rose]        # extend later: torpedo_grass
paths: {raw: data/raw, dataset: data/dataset_v001, runs: runs, out: outputs}
tiling: {tile_px: 1024, scale: 0.5, overlap: 0.25}            # [DATA] decided from stats
train:
  model: yolo11m            # swappable: rtdetr, rf-detr, faster_rcnn
  epochs: 100
  neg_tile_ratio: 1.0
  class_sampling_weights: {chinese_tallow: 3.0, mccartney_rose: 1.0}   # [DATA]
  split: {by: site, folds: 5}
infer:
  tta: false
  thresholds: {chinese_tallow: {candidate: 0.25, high: 0.6}, mccartney_rose: {candidate: 0.25, high: 0.6}}  # recalibrated
georef: {method: exif_flat, dedup_distance_m: 1.5}
export: {formats: [gpkg, geojson, shp, csv], crs_out: EPSG:4326}
```

---

## 7. Repository layout

```
invasive_pipeline/
  configs/default.yaml
  src/invasive/
    io.py            # image + EXIF reading, orientation, GSD
    labelbox.py      # export parsing, API pull
    dataset.py       # manifest, splits, tile sampling, label QA
    tiling.py        # one tiling implementation (+ box clipping rules)
    models.py        # detector wrappers behind one interface
    train.py  infer.py  merge.py  calibrate.py
    georef.py        # exif_flat, odm_camera, mosaic
    dedup.py  export.py  report.py  metrics.py
  scripts/  stats.py  build_dataset.py  train.py  infer.py  evaluate.py  benchmark.py
  tests/             # tiling edge cases, box clipping, georef math, dedup, schema
  docs/
```

### What is reused from the drainage repo
- Edge-aligned sliding-window logic (with a fix for small images), resumable processing, overlay previews, CSV/XLSX export, and the benchmarking and telemetry harness in `evaluate_time/evaluate_pipeline.py`.

### What is dropped
- U-Net and SAM 3 water stage, proximity filter, colour-threshold mask recovery from JPEG overlays, destructive delete scripts, the 180-degree rotation hack, hard-coded paths, duplicated model code.

---

## 8. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Data arrives | Export plus sample frames with EXIF available |
| M1 | Statistics and QA report | Box sizes, GSD, negatives, splits proposed; tile and scale decided |
| M2 | Dataset v001 | Manifest, labels, splits, label-QA flags, data hash |
| M3 | Baseline detector | Per-class metrics on held-out images; error analysis |
| M4 | Inference and export | Frames in, georeferenced GPKG/GeoJSON/CSV out with `exif_flat`; dedup |
| M5 | Review loop | Candidates reviewed in QGIS/Labelbox; hard negatives fed back; model v2 |
| M6 | Accurate georeferencing | `odm_camera` or mosaic path validated against field points |
| M7 | Extensions | Crown masks, Torpedo Grass, multi-date monitoring |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Only 141 tallow boxes | Fine-tune pretrained model, oversample, copy-paste augmentation, grouped CV; treat early results as feasibility; label more tallow first |
| Rose dominates (88%) | Class-weighted sampling and per-class metrics |
| Single-annotator labels | Second-reviewer sample, benchmark/consensus in Labelbox |
| Look-alike vegetation | Hard-negative mining, "other vegetation" negatives from field-reviewed false positives |
| Scale/season shift | GSD checks in ingest; hold out a flight or date; note season of labelled data **[DATA]** |
| Position error from flat-ground georef | Report `georef_error_est_m`; upgrade to ODM camera models for treatment planning |
| Box area overstates canopy | Label it "candidate region area"; add masks later |
| Detector licence (AGPL for Ultralytics) | Choose model family before any distribution; keep the model interface swappable |
| Overfitting to 97 images | Image/site-level splits, final test on unseen flight, report uncertainty |

---

## 10. Open items that only data or the team can settle

1. Sensor, lens and real GSD; RGB only or not. **[DATA]**
2. Box size statistics and therefore `tile_px` and `scale`, and whether two scale passes are needed. **[DATA]**
3. Why 34 images were skipped (negatives or unclear). **[DATA]**
4. Number of sites, flights and capture dates; season. **[DATA]**
5. Flight overlap and GPS grade (standard or RTK), which decides `exif_flat` vs `odm_camera`.
6. Who verifies in the field and with which tool (QField, Labelbox, QGIS).
7. Licence tolerance for the detector family.
8. Acceptance thresholds for recall and precision, agreed with the field team.

---

## 11. Addendum: what the real flight data changed (2026-11-03)

The code is in this package (see ../README.md for status). Findings from inspecting the flights (now in `data/raw/`):

1. **Georeferencing is much better than assumed.** Every frame has a PPK geotag with omega/phi/kappa and 0.06-0.75 m
   accuracy, so the phase-1 method is `wingtra_ppk`, not `exif_flat`. The camera is tilted about 13 deg off nadir in
   cruise, so a nadir assumption would be 10+ m off. The rotation convention was validated by image overlap.
2. **Two cameras, three flights.** HARC: Sony ILX-LR1 61 MP, 17 mm, 9504x6336, orientation 1. DPP: Sony ILCE-6100,
   6000x4000, orientation 3 (180 deg), focal length missing from EXIF (about 12 mm inferred). Camera settings are per flight.
3. **EXIF orientation explains the old 180 deg hack.** The pipeline now works in upright pixel space everywhere and
   converts to raw pixels only where needed (geotag angles refer to the upright image for the DPP flights).
4. **Data problem:** 55 of 187 DPP EO frames are incomplete downloads (`.azDownload-` temp files). They are flagged
   and skipped; re-download them.
5. **Spatial-block folds are needed:** consecutive frames overlap by 60-80%, so random image splits would leak.
6. **Tile scale is more likely 0.25-0.5 than 1.0:** at about 1.3-2 cm/px a 5-8 m shrub or tree crown is 250-600 px. This is
   still [DATA] and will be decided from `scripts/stats.py` on the real export.
7. No tallow/rose labels exist in the workspace yet; the pipeline has only been run on synthetic labels.
