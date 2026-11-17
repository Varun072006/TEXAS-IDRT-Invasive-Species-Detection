# Steps log: invasive species detection training (append-only)

Scope: only `invasive_pipeline/data/images` (97 images, 6000x4000). Goal for now: best detection accuracy for
**McCartney (Macartney) Rose** and **Chinese Tallow**. Geospatial mapping comes later.
Regenerate the numbers any time with `python scripts/ab_report.py` (uses the project venv `.venv`).
Files: `runs_summary.csv` (one row per run), `epochs_all_runs.csv` (every epoch of every run), `RUN_LOG.md` (readable summary).

## 0. Plan (2026-11-12)
Three approaches, one shared evaluation protocol:
1. **YOLO11 trained on tiles + SAHI inference** (core baseline).
2. **SAM3 prompted** (zero/few-shot) from Hugging Face. *Not started: needs `HF_TOKEN` and model id.*
3. **Hybrid**: YOLO proposals + SAM3 refinement + crop classifier. *Not started.*
Protocol: 5 spatial-block folds (4-frame blocks, 1-frame buffer), fold 4 held out as test, report per-class
precision / recall / AP50 / AP50-95 on **full frames after tile merge**; pick the best model by mean AP50, tie-break on the weaker class.

## 1. Data understanding
- 97 images; 1,167 boxes: **Rose 1,026 (class 0), Tallow 141 (class 1)**; 34 frames have no boxes. (`classes.txt`: 0 macartney_rose, 1 chinese_tallow.)
- User note: in `preview/` red = Rose, yellow = Tallow (matches class ids).
- `configs/default.yaml` lists classes in the opposite order (tallow=0). Not used by the new scripts; a test pins the order to `classes.txt`.
- Scope narrowed by user: `data/raw/*/IMAGES` ignored; no pseudo-labelling on unlabelled flights.

## 2. Environment
- Global Python had CPU-only torch 2.14 with a mismatched torchvision -> Ultralytics could not use the GPU.
- Created project venv `.venv` with torch 2.6.0+cu124, ultralytics 8.4.x, sahi 0.14, open_clip, scikit-learn. GPU: RTX 3050 6 GB.
- Global environment left untouched.

## 3. Code written
- `src/invasive/ab.py`: loading, stratified folds, tile pool, fold datasets, full-image metrics, SAHI inference.
- `scripts/ab_prepare.py` (folds + tile pool), `ab_train.py` (train + SAHI score), `ab_eval.py` (score any checkpoint),
  `ab_report.py` (this report), `ab_crop_probe.py` / `ab_frame_probe.py` (diagnostics), `ab_queue1.sh` / `ab_queue2.sh` (run queues).
- `tests/test_ab.py`: class order, counts, both classes per fold, no leakage, raw frame. 4 tests pass.

## 4. Folds and tiles
- First folds (6-frame blocks) were badly unbalanced (6 to 31 images). Rebalanced on Tallow, Rose and image count (4-frame blocks):
  17-20 images per fold, Tallow 25-32 boxes per fold.
- Tile pools: 1024 px tiles, 25% overlap. Scale 0.5 -> 1,118 grid tiles (425 positive); scale 0.75 -> 2,232 grid tiles (611 positive).
  Plus 3 jittered extra tiles per Tallow box (423), 2 used per training run. Negatives sampled 1:1 per fold.

## 5. Problems found and fixed (in order)
| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | First baseline: mAP50 about 0.0001 after 15 epochs, cls loss stuck | Ultralytics `nbs=64` with batch 6 -> gradient accumulation of 11, only ~8 optimiser updates per epoch | `nbs=12` |
| 2 | Still diverging (val cls loss 7,989, then NaN) and 3x slower | Old queue script survived the stop and started fold 1 in parallel on the same GPU; lr too high for fine-tuning | killed the stray run; AdamW `lr0=0.0005`, warmup 2 |
| 3 | After 1+2, still mAP50 about 0.0001 | **Labels are in the RAW pixel frame, not the EXIF-upright frame** (all 97 images have EXIF orientation 3). Provider's preview/annotations assume upright, so boxes missed the plants | training/eval now read the raw frame (`read_frame`) |
- Evidence for #3 (CLIP crop probe, linear classifier, rose/tallow/background): upright frame 47% accuracy; rotated 180 deg = raw frame 85%; flipX 48%; flipY 50%. Re-run on the fixed code: 85.7% (rose F1 0.89, tallow recall 0.94).
- Runs `trial_f0`, `trial2_f0`, and the first `A1_y11s_s050_f1` are **invalid** (pre-fix); kept only for the record.
- Harness notes: a broad process-kill and a glob `rm` were blocked by safety checks; only the specific runs I had started were stopped, and new run names were used instead of deleting folders.

## 6. Approach 1 runs (YOLO11 + SAHI), fold 0 validation, test fold 4 untouched
Settings common to all: 1024 px tiles, AdamW lr0 0.0005, nbs 12, flipud/fliplr 0.5, mosaic (off for last 10 epochs), 2 extra Tallow tiles per box.
| Run | Model | Scale | Epochs | Tile mAP50 (best) | Full-image mAP50 | Rose AP50 | Tallow AP50 |
|---|---|---|---|---|---|---|---|
| A1_y11s_s050_f0 | yolo11s | 0.5 | 40/50 (early stop) | 0.282 (ep 21) | 0.209 | 0.052 | 0.367 |
| B1_y11s_s075_f0 | yolo11s | 0.75 | 60/60 | 0.343 (ep 20) | **0.299** | 0.080 (P 0.13, R 0.23) | 0.517 (P 0.55, R 0.57) |
| B2_y11m_s050_f0 | yolo11m | 0.5 | cancelled before epoch 1 (resolution matters more, see below) | - | - | - | - |
| B3_y11s_s100_f0 | yolo11s | 1.0 | 21/50 (interrupted) | 0.357 (ep 13) | 0.259 | 0.069 (P 0.13, R 0.17) | 0.449 (P 0.48, R 0.54) |
| B4_y11m_s075_f0 | yolo11m | 0.75 | 43/50 (interrupted) | 0.347 (ep 31) | 0.292 | 0.081 (P 0.15, R 0.22) | 0.504 (P 0.65, R 0.46) |
Observations: validation mAP is very noisy epoch to epoch (single fold, 17 frames); training loss keeps falling after epoch ~20 (overfitting).
Rose is the weak class: many small, heavily overlapping boxes (255 Rose boxes in 17 frames), AP50 0.05 at scale 0.5.
Higher resolution helps a lot: scale 0.5 -> 0.75 raised full-image mAP50 from 0.209 to 0.299 (Tallow 0.37 -> 0.52, Rose 0.05 -> 0.08). So B2 (bigger model at 0.5) was cancelled and replaced by B3 (scale 1.0, native resolution) and B4 (yolo11m at 0.75). Scale 1.0 tile pool: 3,656 grid tiles (752 positive). **Update this table from `RUN_LOG.md` as runs finish.**

## 7. Still to do
- Finish B1/B2, compare on full frames; confirm the winner on a second fold.
- Tune merge (NMS vs GREEDYNMM) and per-class confidence thresholds.
- Final model: train on all non-test folds, score once on fold 4 (test); report precision / recall / AP50 per class.
- Approaches 2 and 3 once `HF_TOKEN` and the SAM3 model id are available (SAM3 weights on Hugging Face are gated).
- Georeferencing stage later: pixel coordinates are in the raw frame (rotate 180 deg for the upright camera model).

## 8. Orientation / annotation alignment fix (user request, 2026-11-12)
- Reported: previews are rotated while boxes stay put. Traced to `export_annotations.py` (preview used `exif_transpose`; assumed
  Labelbox annotates the upright image). Labelbox boxes are in the stored pixel grid; all 97 files have EXIF orientation 3.
  `download_images.py` is a byte copy and not at fault; `labels/*.txt` equal the Labelbox export for all 97 images.
- Added `src/invasive/exif_fix.py` (lossless 2-byte orientation reset), `scripts/fix_orientation.py`, `tests/test_exif_fix.py` (6 tests).
- Output: `data/images/corrected/` (images with orientation 1, labels, previews, comparison sheets, mapping.csv, annotations.json,
  VERIFICATION_REPORT.md). 97/97 verified. Originals, old preview/, ndjson untouched.
- Corrected the wrong statements in `export_annotations.py` (not re-run) and `configs/default.yaml`.
- Training is unaffected: it already reads the stored grid (this is the same finding as step 5 #3, now fixed at the source).

- Deleted the old misaligned `data/images/preview/` folder (97 JPGs, 79 MB) at the user's request. `corrected/preview/` replaces it. `data/images/Labelbox Annotated Images.zip` still contains the same old previews (not deleted). `fix_orientation.py` now skips BEFORE/AFTER sheets when the old folder is absent.
