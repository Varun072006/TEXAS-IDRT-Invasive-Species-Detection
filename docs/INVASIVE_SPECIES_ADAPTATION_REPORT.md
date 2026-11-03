# Adapting the NKT / Golden Pipeline from Drainage Infrastructure to Drone-Based Invasive Species Detection

Scope: every file under `nkt_pipeline/` (29 files: 15 Python scripts, 2 notebooks, 2 stats markdown files, 1 shell script, 6 CSV/XLSX benchmark outputs). No image data, model weights or COCO files are in the folder. They live on the original machine at `/home/ameen/texas_drone/...`.

---

## 1. Executive summary

**What the pipeline is today**

1. SAM 3, prompted with the text "water", segments water on each drone frame.
2. The frame is tiled into 1024x1024 patches (stride 819, 20% overlap).
3. A hand-written U-Net with a ResNet-50 encoder predicts 3 classes per pixel: background, vegetation overgrowth, bank erosion.
4. A geometric filter keeps a detection only if it lies within 32 px (vegetation) or 60 px (erosion) of the SAM water mask.
5. The patches are stitched back and rotated 180°. Connected components are counted and Excel/CSV statistics are exported.

**What that means for your goal**

- The domain logic (water proximity, 2 classes, "defect" counts) is specific to drainage. The scaffolding is reusable: tiling, batched inference, stitching, stats export, benchmarking.
- The pipeline has structural weaknesses that matter more for invasive species than they did for drainage.
  - It discards all negative patches in training.
  - It trains on its own pseudo-labels.
  - It uses pixel units instead of metres.
  - It has no georeferencing.
  - It has inconsistent input normalisation.
  - It makes a patch-level split that leaks.
- The recorded accuracy is lower than the docs claim. The golden notebook's own output ends at **val mIoU ≈ 0.66** (epoch 50), while `dataset_statistics.md` lists **0.8465**. That figure is labelled a "target" and the NKT notebook has no outputs. Treat 0.66 as your real baseline on drainage.

**Recommendation in one paragraph**

Keep the architecture of the pipeline (tile → segment → context filter → stitch → report). Rebuild around it.

1. Make the data and label schema species-aware and keep negatives.
2. Make everything georeferenced and metric.
3. Turn the "SAM water proximity" stage into a pluggable *context prior* that is only used when the target species is riparian or aquatic.
4. Replace the training loop with a proper one: ImageNet normalisation, site-level splits, class weighting, proper augmentation, per-class precision/recall.
5. Consolidate the 8 copies of the model code into one package with a config file.

Details and a phased plan are below.

---

## 2. Complete file inventory and what each file does

### 2.1 Root `nkt_pipeline/`: dataset creation from COCO annotations (stage 1, training data)

| File | Purpose | Drainage-specific? | Verdict for invasive species |
|---|---|---|---|
| `process_nkt_pipeline.py` | Reads Label Studio COCO export; keeps only `vegetation_overgrowth` and `bank_erosion`; overlays polygons on the original 1024-ish patches; rebuilds parent frames from `patches_info.json` offsets; rotates 180°. | Yes (category names, 180° rotation hack) | **Rewrite.** Keep the idea (COCO → per-parent masks) but drive it from a config class map. |
| `slice_annotated_1024.py` | Slices reconstructed parents to 1024 tiles, stride 819. | No | **Reuse** (replace with the shared tiler). |
| `filter_annotated_1024.py` | Deletes tiles that don't intersect any annotation polygon (destructive `unlink`). | Partly | **Drop or invert.** Deleting negatives is harmful (see 4.1). |
| `generate_nkt_unet_dataset.py` | Rasterises COCO polygons to a parent-size mask (0/1/2), slices, **keeps only tiles with class 1 or 2**, writes `input_patches/` and `masks/` and the stats md. Polygon coordinates are flipped to the "180°-rotated upright" space (`generate_nkt_unet_dataset.py:112-114`). | Yes | **Rewrite.** Same concept; N classes, keep negatives, record provenance. |
| `generate_annotated_overlays.py` | Alpha-blends masks onto tiles for visual QA. | Colours only | **Reuse** with a palette from config. |
| `create_golden_pipeline.py` | Builds the "golden" dataset: 574 human-labelled tiles + 1,269 *model-predicted, water-filtered* tiles → 1,843. Copies model predictions in as ground-truth masks (`create_golden_pipeline.py:90-92`). | Yes | **Do not copy this pattern blindly.** It is self-training without any human verification step (see 4.2). |
| `run_nkt_inference_pipeline.py` | Original end-to-end inference on 110 images: SAM3 water → tile → U-Net → vegetation-only proximity filter (32 px) → overlay → stitch. | Yes | Superseded by the golden version. |
| `train_unet_1024_resnet50.ipynb` | Training notebook (NKT data). No saved outputs. | No | **Reuse as a starting point**, with the fixes in section 5.4. |
| `dataset_statistics.md` | Dataset and hyper-parameter summary. Contains a metrics table (mIoU 0.8465, etc.) labelled "target objective". | n/a | Metrics are unverified. Don't carry them over. |

### 2.2 `golden_pipeline/`: production-style inference + data curation

| File | Purpose | Notes |
|---|---|---|
| `run_full_dataset_pipeline.py` | **Main production pipeline.** Symlinks 5,757 frames from `full_dataset/*/IMAGES/`, SAM3 water (frame downscaled to max 1024 px), tiles to 40 patches/frame, batch U-Net (batch 2, fp16 autocast), proximity filter (veg 32 px, erosion 60 px), overlay, stitch, rotate 180°, resumable (skips finished frames). | Core of what you will reuse. Writes ~230k patch JPEGs to disk. |
| `rerun_patch_inference_and_export_excel.py` | Re-runs U-Net on all frames reusing cached SAM masks; writes per-patch masks, overlays, and a CSV/XLSX of **pixel counts** per patch; batch 4. | Example of the "re-run without SAM" optimisation: reuse. |
| `count_instance_clusters_from_masks.py` | Walks 230,280 patch masks; `cv2.connectedComponents` (8-conn.) per class → "instances" CSV/XLSX. | Instance = connected blob in a *single patch*. Double counts across overlaps. |
| `organize_annotated_parents.py` | Moves stitched images into per-flight subfolders; **deletes** frames with no detections by detecting overlay colours in JPEGs. | Colour-sniffing on lossy JPEGs. Fragile and destructive. |
| `filter_and_count_annotated.py` | Same idea: `abs(output - input).mean() > 0.5` to decide if a frame has any detection; **deletes** the rest. Prints a per-flight table. | Same caveats. |
| `extract_selected_dataset.py` | Picks 10 frames per flight (every 5th); reconstructs the parent mask **from overlay JPEG colours**; saves orig/mask/overlay. | Contains a tile-position bug (4.5). |
| `populate_selected_images_pipeline.py` | Same selection, but copies 6 assets per frame (parent, water mask, input/output patches, mask, overlay). | Same tile-position bug. |
| `generate_exact_selected_masks.py` | Re-runs U-Net + water proximity at full parent size, writes exact masks and overlays. | Uses **ImageNet normalisation**, unlike training and the other scripts (4.3). |
| `run_sample_inference.py` | Smoke test on 2 frames; no SAM, no filter. | Fine for debugging. |
| `sample_check_verification.py` | Re-stitches output patches for 2 random frames per flight to visually verify alignment. | A useful QA idea; keep the concept. |
| `train_unet_1024_resnet50.ipynb` | Training on the 1,843-patch golden set (1,475 train / 368 val). **Has outputs**: 50 epochs, final val loss 0.275, val mIoU 0.664 (epoch 1 was 0.582). Only the 3 path cells differ from the NKT notebook. | Real baseline numbers live here. |
| `dataset_statistics.md` | Golden dataset summary. 88% of patches contain vegetation, 54% erosion. Has no metrics table. | |
| `evaluate_time/evaluate_pipeline.py` | 5-phase benchmark harness with hardware telemetry (GPU/CPU memory, per-phase time) and instance statistics; appends to CSV/XLSX. | **Best reusable asset**: keep and extend. |
| `evaluate_time/run_evaluation.sh` | Launcher with a hard-coded Linux venv path. | Replace with a cross-platform entry point. |
| `evaluate_time/statistics/*.csv/.xlsx` | Results for **2 images only**: 5.28 s/frame on an RTX 3060 (SAM3 1.59 s = 30%, tiling 0.18 s, U-Net 2.94 s = 56%, filter 0.38 s, stitch 0.18 s), peak VRAM 7.1 GB. | Sample is too small to be a benchmark. |

### 2.3 The model (copy-pasted into 8 files)

`UNetWithResNet50` is a custom U-Net: ResNet-50 encoder (ImageNet weights), 5 decoder blocks with bilinear upsampling and skip concatenation, 1x1 classifier. It is duplicated in `run_nkt_inference_pipeline.py`, `run_full_dataset_pipeline.py`, `evaluate_pipeline.py`, `generate_exact_selected_masks.py`, `rerun_patch_inference_and_export_excel.py`, `run_sample_inference.py` and both notebooks. `extract_polygons` and `filter_by_water_proximity` are likewise duplicated in 4 files. Any change must currently be made in many places.

---

## 3. Conceptual differences: drainage vs invasive species

| Aspect | Drainage pipeline (today) | Invasive-species problem |
|---|---|---|
| Target | Two coarse, visually distinct "defect" classes tied to a water channel | One or more *species*, often visually close to native vegetation (confusable background) |
| Spatial prior | Strong: defects are near water → SAM water + 32/60 px filter | Depends on species. Riparian/aquatic (water hyacinth, giant reed, hydrilla, salvinia): water-adjacent helps. Terrestrial (kudzu, cogongrass, buffelgrass, tree-of-heaven, Chinese tallow, Johnson grass): no water prior |
| Discriminating signal | Shape/texture/colour at moderate scale | Texture, canopy structure, **phenology** (flowering, senescence, leaf-on/off), often NIR/red-edge. RGB alone may not be enough |
| Object size | Large blobs (tens of thousands of px) | Anything from single plants (small) to monocultures (huge) |
| Class balance | Vegetation in 88% of patches | Usually rare (<5% of pixels). Extreme imbalance |
| Ground truth | Human polygons in Label Studio | Needs field-validated labels (botanist, GPS plots), or at least expert review |
| Output | Count of defect "instances" per frame | Cover %, area in m², location (lat/lon), density, change over time, priority for treatment |
| Georeferencing | None (pixels only, files are JPEG frames) | Essential: managers act on map coordinates |
| Error cost | Missed erosion/overgrowth = inspection miss | False negative = infestation spreads; false positive = wasted herbicide/crew time and loss of trust |

---

## 4. Problems found in the current code (they affect any re-use)

These are ordered by how much they will hurt invasive-species work.

### 4.1 Negatives are thrown away in training data
`generate_nkt_unet_dataset.py:159-160` ("Filter out unannotated patches") and `filter_annotated_1024.py` keep only tiles with a class-1/2 pixel. Of 1,760 NKT tiles, 1,186 (67%) were discarded. The model never sees hard negatives such as native plants, shadows, bare ground or roofs. For drainage this was tolerable because the SAM water proximity filter removed many false positives afterwards. For invasive species, **look-alike native vegetation is the main error source**, so negatives are required, and the proximity safety net may not exist.

### 4.2 Self-training on unverified pseudo-labels
`create_golden_pipeline.py` adds 1,269 model-predicted (and water-filtered) tiles as ground truth. That is 69% of the 1,843 "golden" tiles. Consequences:
- The model learns its own mistakes (confirmation bias).
- The validation split mixes pseudo-labels in, so validation measures agreement with itself, not correctness.
- Your real accuracy on human-labelled data is unmeasured.

Pseudo-labelling can work, but only with a human review step on a sample, a held-out *human-only* test set, and confidence thresholds.

### 4.3 Inconsistent preprocessing
- Training (notebook cell 3) and most inference scripts feed `RGB/255` with **no mean/std normalisation**, although the encoder is ImageNet-pretrained.
- `generate_exact_selected_masks.py:250` applies ImageNet mean/std. Its masks come from different input statistics than the weights were trained with, so they are likely degraded. The two kinds of output are not comparable.

Pick one convention (ImageNet mean/std is the right one for torchvision ResNet) and retrain, since the current weights were trained without it.

### 4.4 Leaky train/validation split
Notebook cell 6 shuffles patches randomly and splits 80/20. Tiles overlap 20% and many come from the same frame and flight, so near-duplicate pixels are in both train and val. This inflates val mIoU. For invasive species, split by **site / flight / date**, and keep a final test set from a different site.

### 4.5 Stitching and tile-index bugs
- Overlap regions are **overwritten**, not blended: `run_full_dataset_pipeline.py:394`, `evaluate_pipeline.py:578`. The log text in `evaluate_pipeline.py:565` says "Blend Overlapping Borders", but no blending exists. Seams and double edge artefacts follow.
- `extract_selected_dataset.py:39-40` and `populate_selected_images_pipeline.py:54-55` compute tile position as `j*STRIDE`. The real grid appends a final edge-aligned tile at `w-1024` or `h-1024`, so the last column and row of tiles are placed wrongly. Masks reconstructed this way are misaligned at the right and bottom edges. `sample_check_verification.py` has the correct logic.
- `x_starts[-1]` raises `IndexError` if an image is smaller than 1024 px on a side (all tilers).

### 4.6 Masks recovered from lossy colour overlays
`extract_selected_dataset.py:48-50`, `populate_selected_images_pipeline.py:62-63`, `organize_annotated_parents.py:45-47` and `filter_and_count_annotated.py:53` recover "masks" or "has detection" by thresholding overlay colours in 90%-quality JPEGs. The thresholds differ in each file (35/25/70, 40/30/80, 50/40/90). This is lossy, picks up natural green or brown pixels, and can disagree between scripts. The real mask is already available at inference time (`validated_mask`) and should be saved directly.

### 4.7 Geometric filter side effects
- `extract_polygons` uses `RETR_EXTERNAL` and then `drawContours(..., FILLED)`. Holes are filled, so the validated mask is not the raw prediction.
- The water mask is clipped per patch, so a detection near water that lies just across a patch boundary can be rejected.
- `min_area=30` px drops small detections. At a fine GSD this would delete real small invasive plants.
- `cnt.squeeze()` breaks for contours with a single point (guarded only by the shape check).
- Complexity is O(detections x water polygons) with Shapely calls. Fine for 40 patches/frame; slow for large orthomosaics.

### 4.8 Units are pixels, not metres
32 px and 60 px mean different real distances depending on altitude and sensor. All area numbers (`vegetation_area_px`, `water_area_px`) are pixels. Nothing in the repo stores GSD, altitude, camera, GPS or CRS.

### 4.9 SAM 3 is run on a heavily downscaled frame
6000x4000 frames are shrunk to max 1024 px (`run_full_dataset_pipeline.py:256-263`), about 5.9x, then masks are upsampled with nearest neighbour. Fine for water bodies; unsuitable for anything small. (The original `run_nkt_inference_pipeline.py` ran SAM at full size with a `> 300` px component filter; the golden version uses `> 100` on the shrunken mask, so the two differ.)

### 4.10 Instance counting is not instance counting
`cv2.connectedComponents` on each patch counts blobs, so the same object in the 20% overlap zone is counted in two patches, and touching plants merge into one. For a frame-level count, the code does run it on the merged mask (`evaluate_pipeline.py:593-607`), which is better. The patch-level CSVs (`count_instance_clusters_from_masks.py`, 230k rows) are not additive.

### 4.11 Metrics and training loop
- `calculate_iou` is computed per batch of 2 and averaged. It is not computed over the whole validation set; NaN classes are skipped per batch. Use a global confusion matrix.
- The best checkpoint is chosen by validation loss, not by per-class IoU or F1.
- Unweighted `CrossEntropyLoss` plus soft Dice averaged across classes (background included) will favour the majority class when positives are sparse.
- Augmentation: flips and 90° rotations only. No colour/brightness jitter, scale, blur or haze, which matters for varying illumination, season and sensors.
- The NKT notebook was saved without outputs, so `0.8465` has no run behind it.

### 4.12 Engineering issues
- Hard-coded `/home/ameen/texas_drone/...` paths in all 17 scripts and notebooks; `run_evaluation.sh` points to a Linux venv.
- The HF token is parsed from a `.env` by string splitting. (`HF_TOKEN=` content split, no dotenv.)
- `torch.load` without `weights_only=True`.
- Destructive operations without dry-run: `unlink()` in `filter_annotated_1024.py:114,146`, `filter_and_count_annotated.py:90`, `organize_annotated_parents.py:105`, `shutil.rmtree` in `create_golden_pipeline.py:50` and `populate_selected_images_pipeline.py:91`.
- 180° rotation hack in 8 places. It exists because annotations were made on unrotated Label Studio patches while the "upright" frames are EXIF-transposed; each script re-implements the coordinate flip `(W-1-x, H-1-y)`. This is a fragile contract; a new dataset should be stored in one canonical orientation.
- ~230k JPEG patches written to disk for 5,757 frames. Slow and lossy; patches can be produced on the fly.
- The code calls full frames "orthophotos". They are individual nadir camera frames, not orthorectified mosaics.
- No tests, no config, no requirements file.

---

## 5. Recommended approach

### 5.1 First, pin down the problem (decisions that change the design)

These answers determine most of the architecture. I'd settle them before writing code:

1. **Which species?** One species vs a list. Riparian/aquatic or terrestrial?
2. **What sensor?** RGB only, or RGB+NIR / multispectral (MicaSense, Parrot Sequoia)? Thermal?
3. **Flight altitude / GSD?** Target cm/px. The number of pixels a plant occupies drives patch size and model choice.
4. **Frames or orthomosaics?** Individual JPEGs (as now) or stitched GeoTIFFs (WebODM, Pix4D, Metashape)?
5. **What labels exist or can be produced?** Polygons, points, boxes, field plots; who verifies them (botanist)?
6. **What deliverable?** Map layer (GeoTIFF/GeoPackage), cover % per site, alert list, time series?
7. **How many sites/dates?** Determines whether you can test generalisation and use phenology.

### 5.2 Target architecture

```
configs/*.yaml  (species, class map, GSD, paths, thresholds)
        │
 [A] Ingest & metadata  ── EXIF/GeoTIFF → GSD, CRS, date, site; canonical orientation
        │
 [B] Context layers (optional, pluggable)  ── water (SAM3/NDWI), habitat/land-cover, DEM, roads/trails
        │
 [C] Tiling (windowed reads, 20-50% overlap, no disk patches)
        │
 [D] Segmentation model  ── N+1 classes, ImageNet-norm, (optional extra bands)
        │
 [E] Logit/probability stitching (Hann-window blending, TTA)
        │
 [F] Post-processing  ── threshold per class, morphological clean-up, min area in m², optional context prior
        │
 [G] Outputs  ── class GeoTIFF/COG, polygons (GeoPackage/GeoJSON), confidence map, per-site CSV (area m², cover %)
        │
 [H] Evaluation & benchmarking  ── per-class P/R/F1/IoU, site-level split, timing/VRAM (reuse evaluate_pipeline.py)
```

### 5.3 What to keep, change, replace

**Keep (works, reusable idea)**
- Overall staged flow and `evaluate_pipeline.py` benchmarking and telemetry harness.
- Sliding-window tiling logic including the final edge-aligned tile (but fix the small-image case).
- Resume-friendly "skip if output exists" behaviour.
- Overlay visualisation and the sample-check idea.
- Batched fp16 inference.
- CSV/XLSX statistics export.

**Change**
- **Class schema**: from `{0 bg, 1 veg-overgrowth, 2 erosion}` to `{0 background, 1..N species}`. Consider an explicit extra class for *"other vegetation / native look-alike"* so the model learns the distinction instead of lumping it into background. Drive the class list and palette from config, not constants in each script.
- **Water proximity stage** (`filter_by_water_proximity`, SAM 3 "water"): make it optional and configurable per species.
  - Riparian/aquatic target: keep it, but express thresholds in metres (`distance_m / GSD`), compute water at the frame or mosaic level rather than per patch, and consider a spectral index (NDWI) as a cheaper alternative if you have NIR.
  - Terrestrial target: remove the stage. Don't force a water prior where none exists.
  - In both cases generalise the function to "keep detection if a context-layer condition holds" so that other priors (habitat polygons, proximity to roads, DEM) can be plugged in.
- **Units**: pixels → metres using GSD from the sensor and altitude or from the GeoTIFF transform. Report area in m² and hectares, and cover %.
- **Instance logic**: report area and cover % as the primary metric. If instance counts are needed, count on the merged mosaic-level mask and consider watershed or per-plant detection when plants are separable.
- **Data pipeline**: store `masks` straight from `fillPoly`/model output, never recover them from overlays; save as PNG (lossless) or a single GeoTIFF; save negatives.
- **Training**: see 5.4.
- **Outputs**: add georeferenced vector polygons and a confidence layer.

**Replace / remove**
- The eight copies of the model and helper functions → one `invasive_pipeline/` package (`model.py`, `tiling.py`, `stitch.py`, `context.py`, `io.py`, `metrics.py`).
- `create_golden_pipeline.py` as it is (unreviewed pseudo-labels merged into ground truth).
- All colour-threshold mask recovery in `extract_selected_dataset.py`, `populate_selected_images_pipeline.py`, `organize_annotated_parents.py`, `filter_and_count_annotated.py`.
- The 180° rotation hack: normalise orientation once at ingest.
- Hard-coded paths and `run_evaluation.sh`.
- The shipped metrics claims in `dataset_statistics.md`.

### 5.4 Model and training changes

1. **Normalise inputs** with ImageNet mean/std (or the stats of your foundation backbone) consistently at train and inference.
2. **Keep negatives**: sample background tiles at a ratio (for example 1:1 to 3:1 against positive tiles), and add *hard-negative mining*: run the model, collect false positives, add them back and retrain.
3. **Split by site/flight/date** with a separate test set. Avoid random patch-level splits. If data is scarce, use k-fold by site.
4. **Loss**: class-weighted or focal CE plus Dice/Tversky/Lovász computed only over foreground classes, so rare species are not drowned by background.
5. **Augmentation** (use `albumentations`): colour/brightness/contrast/hue jitter, scale and random-resized crop, blur, noise, haze, rotation, flips. Simulate seasonal and illumination change since species appearance shifts with phenology.
6. **Backbone options** (in increasing effort):
   - Keep the U-Net, but switch to `segmentation_models_pytorch` so you get tested decoders, multiple encoders and an easy `in_channels` change.
   - Try a stronger pretrained encoder: ConvNeXt/Swin, or a SegFormer / UperNet / Mask2Former head. DINOv2-style self-supervised features are strong with few labels.
   - Remote-sensing-pretrained backbones can help with limited data.
7. **Extra bands**: if you have NIR / red edge, extend the first conv to 4–5 channels (copy and average the RGB weights for new channels) and add indices such as NDVI and NDRE as inputs.
8. **Patch size / scale**: a 1024 tile at about 1–2 cm GSD covers 10–20 m. Pick tile size and downscaling so a typical target plant spans roughly 20–200 px. Consider multi-scale inference when infestations vary from single plants to large stands.
9. **Metrics**: global confusion matrix per class (IoU, precision, recall, F1), PR curves, and a *detection-level* metric (is each infestation found?) plus *area error* (m²). Select checkpoints on foreground-class F1 or IoU, not total loss.
10. **Uncertainty / review loop**: save per-pixel confidence; route low-confidence tiles to human review in Label Studio; feed corrections back (active learning). This replaces the unreviewed pseudo-labelling in `create_golden_pipeline.py` with a controlled version.
11. **SAM 3 as an annotation accelerator**, not a classifier: use text or point prompts to propose polygons that a botanist accepts or corrects. This could multiply your labelling speed. Do not rely on SAM text prompts to decide the species.

### 5.5 Inference and post-processing changes

- Read windows straight from the source raster (`rasterio` windowed reads) rather than writing 40 JPEG patches per image. This is faster and avoids a second JPEG compression.
- Accumulate **softmax probabilities** with a Hann/cosine window across overlapping tiles and take the argmax at the end. Optionally add flip/rot TTA (typically +1–3 IoU).
- Store the probability map so thresholds can be tuned per class without re-running the model.
- Post-process: remove blobs smaller than `min_area_m2`, close small gaps, optionally require N overlapping frames to agree (for frame sequences with GPS/overlap).
- Georeference outputs: write class raster as a GeoTIFF/COG with the source CRS, and polygons as GeoPackage/GeoJSON with attributes (species, area_m2, mean_confidence, date, site, flight id).
- For frame sequences without orthorectification, either build an orthomosaic first (WebODM, free) or use EXIF GPS plus camera pose to project detections; orthomosaics are far simpler for mapping.
- Multi-date monitoring: add a change-detection stage (new infestation, spread, treatment effect) on co-registered rasters.

### 5.6 Reporting changes

Replace `vegetation_instances / bank_erosion_instances / total_defect_instances` with, per species and per site/flight:
`area_m2`, `area_ha`, `cover_pct`, `n_patches/blobs`, `mean_confidence`, `largest_patch_m2`, `nearest_water_m` (if relevant), plus processing metadata (GSD, date, model version, thresholds). Keep the timing/VRAM telemetry from `evaluate_pipeline.py`, but benchmark on a representative set (the current numbers come from 2 frames).

### 5.7 Engineering and repository structure

```
invasive_pipeline/
  configs/default.yaml        # paths, classes, palette, thresholds in metres, tile size, batch
  src/
    io.py                     # raster/EXIF reading, GSD, CRS, orientation normalisation
    tiling.py                 # window grid (one implementation)
    model.py                  # smp/SegFormer builder + checkpoint load (weights_only=True)
    context.py                # water (SAM3/NDWI), habitat masks; pluggable filters
    stitch.py                 # probability blending
    postprocess.py            # min area, morphology, vectorise
    metrics.py                # confusion matrix, per-class IoU/P/R/F1, area error
    report.py                 # CSV/XLSX/GeoPackage export
  scripts/ train.py, infer.py, evaluate.py, benchmark.py   # benchmark = evolved evaluate_pipeline.py
  data_tools/ coco_to_masks.py, make_tiles.py (negatives kept), qa_overlays.py
  tests/                      # tiling edge cases, orientation, class map, stitch
```
Also: `requirements.txt`, no hard-coded paths, a `--dry-run` for any command that deletes, experiment tracking (MLflow or W&B), versioned data manifests (a CSV listing image, site, date, GSD, split).

---

## 6. Suggested phased plan

| Phase | Work | Output |
|---|---|---|
| 0. Define | Answer the 7 questions in 5.1; choose species, classes, deliverables | One-page spec |
| 1. Data | Collect/ingest imagery with metadata; label with botanist review; include negatives and look-alikes; site/date-based splits and a held-out test site | Versioned dataset + manifest |
| 2. Refactor | Create the shared package; unify tiling, model, normalisation, orientation; config-driven classes | Single code path, tests passing |
| 3. Baseline | Train on the new dataset with the existing U-Net (fixed normalisation, class weights, new augmentations); evaluate per class on the held-out site | Honest baseline numbers |
| 4. Improve | Better backbone (SegFormer/UperNet/DINOv2), extra bands if available, hard-negative mining, TTA | Model v2 |
| 5. Context + post-processing | Add the optional context prior (water etc.), probability stitching, metric min-area, vectorisation | Georeferenced outputs |
| 6. Review loop | Confidence-based review in Label Studio, re-train | Model v3, active-learning cycle |
| 7. Operationalise | Benchmark on representative sets, batch processing, reports, change detection | Production pipeline |

---

## 7. Quick wins you could do first (low effort, high value)

1. Fix preprocessing consistency (ImageNet mean/std) and retrain. It is a one-line change in the dataset class plus inference scripts.
2. Stop deleting unannotated tiles; sample negatives into the training set.
3. Replace the random patch split with a split by frame/flight (even before you have new data, it will show how optimistic 0.66 may be).
4. Report global per-class IoU/precision/recall instead of the per-batch mean.
5. Move the model and helper functions into one module and import it everywhere.
6. Save the validated mask at inference time instead of reconstructing it from JPEG colours.

---

## 8. Risks and things to watch

- **Spectral confusion**: RGB may not separate the invasive from native plants of similar colour. Evaluate early with a small labelled pilot; add NIR or use seasonal flights if needed.
- **Domain shift**: sensors, altitude, lighting, season and site change the appearance. Hold out a whole site for testing, and expect to fine-tune per region.
- **Label quality**: species-level labels are expensive. Plan the botanist time first; model work comes second.
- **False-positive cost**: managers act on the output. Report precision and recall separately, show confidence, and provide a review workflow.
- **Reuse of the drainage model**: the current weights encode drainage classes and were trained without normalisation. They are not useful as a starting checkpoint except, at best, as a generic vegetation-texture initialiser. Start from ImageNet (or better) weights.
- **Licence / access**: `facebook/sam3` is a gated Hugging Face model needing a token; confirm the licence terms for your use.

---

## 9. Concrete questions for you

1. Which invasive species (and how many) are you targeting, and are they aquatic/riparian or terrestrial?
2. What camera/sensor and altitude do you fly (RGB only? NIR?). Do you know the GSD?
3. Will inputs be individual JPEG frames, as in the drainage data, or orthomosaic GeoTIFFs?
4. Do you already have labelled polygons or field survey points, and who can verify them?
5. What does the end user need: a map, an area/cover report, alerts, or a time series?
6. Do you want to keep the water-proximity logic (riparian species), or drop it?

With those answers I can turn section 5 into a concrete refactor and start with the quick wins in section 7.
