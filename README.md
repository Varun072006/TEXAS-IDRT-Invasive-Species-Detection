# 🌿 AI-Driven Drone Detection & Mapping of Invasive Species
### Texas IDRT — Automated Aerial Monitoring for Ecological Conservation & Infrastructure Protection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF?logo=yolo&logoColor=black)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Tests Passing](https://img.shields.io/badge/tests-44%20passed-brightgreen?logo=pytest&logoColor=white)](./tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

---

## 📖 Table of Contents
- [Executive Overview](#-executive-overview)
- [Target Species](#-target-species)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
  - [1. Installation & Environment](#1-installation--environment)
  - [2. Run the Verification Suite](#2-run-the-verification-suite)
  - [3. End-to-End Workflow](#3-end-to-end-workflow)
- [Pipeline Deep-Dive](#-pipeline-deep-dive)
  - [A. Flight Ingestion & Orientation Handling](#a-flight-ingestion--orientation-handling)
  - [B. Sliding Window Tiling & Bounding Box Clipping](#b-sliding-window-tiling--bounding-box-clipping)
  - [C. Leak-Free Spatial-Block Dataset Splitting](#c-leak-free-spatial-block-dataset-splitting)
  - [D. Metric Georeferencing & 3D Camera Projection](#d-metric-georeferencing--3d-camera-projection)
  - [E. Cross-Frame Deduplication & GIS Deliverables](#e-cross-frame-deduplication--gis-deliverables)
- [Configuration Reference (`configs/default.yaml`)](#-configuration-reference-configsdefaultyaml)
- [Sensors & Photogrammetric Validation](#-sensors--photogrammetric-validation)
- [Confidentiality & Data Protection Guardrails](#-confidentiality--data-protection-guardrails)
- [Contributing & License](#-contributing--license)

---

## 🔭 Executive Overview

Invasive plant species present a severe ecological and operational threat across Texas wetlands, coastal plains, and drainage networks. Species such as **Chinese Tallow** (*Triadica sebifera*) and **Macartney Rose** (*Rosa bracteata*) rapidly outcompete native vegetation, restrict water runoff, destabilize soil banks, and escalate flooding hazards.

Traditional field surveys are labor-intensive, logistically demanding, and hazardous across dense brush or wetland terrains. Aerial surveys using autonomous fixed-wing drones (e.g., **Wingtra PPK**) enable rapid capture of ultra-high-resolution aerial imagery (1.0–1.5 cm/px Ground Sampling Distance at 60 m flight height). However, processing this data introduces critical engineering challenges:

1. **Extreme Resolution**: Aerial frames exceed 61 Megapixels ($9504 \times 6336$ px), requiring memory-efficient sliced inference.
2. **Flight Dynamics & Camera Pitch**: Fixed-wing drones tilt 10°–15° off-nadir in cruise, invalidating standard flat-nadir assumptions and causing 5–10 meter ground localization errors without full rotation matrices.
3. **Data Overlap**: Consecutive drone frames overlap by 60%–80%, leading to duplicate detections and training data leakage if frames are split naively.
4. **Actionable Deliverables**: Conservation teams require **georeferenced candidate regions** (GeoPackage / GeoJSON / Shapefile) directly loadable into mobile field tools (QGIS / QField) for ground verification and targeted eradication.

This repository implements an end-to-end, config-driven computer vision and geospatial pipeline engineered specifically to address these challenges.

---

## 🎯 Target Species

| Species | Form | Ecological & Infrastructure Impact | Target Deliverable |
| :--- | :--- | :--- | :--- |
| **Chinese Tallow**<br/>(*Triadica sebifera*) | Deciduous tree | Aggressive colonizer; transforms coastal prairies into monoculture woodlands; alters soil chemistry and blocks drainage corridors. | Individual tree crown candidate polygons with GPS centroid & estimated footprint. |
| **Macartney Rose**<br/>(*Rosa bracteata*) | Evergreen thorny shrub | Forms impenetrable, massive clumps along drainage channels and rangelands, restricting wildlife movement and flow capacity. | Shrub boundary clusters with georeferenced bounding boxes and field verification tags. |

---

## 🏗️ System Architecture

![Pipeline Architecture](./Invasive%20Species%20Detection%20Pipeline%20Architecture%20diagram.png)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                   │
│   • Drone RGB Frames (Sony ILX-LR1 61MP / ILCE-6100 24MP)                   │
│   • Wingtra PPK Geotags (Lat, Lon, Altitude, Omega, Phi, Kappa)             │
│   • Labelbox Annotations (NDJSON bounding boxes)                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 1: DATASET BUILDER                           │
│   • EXIF Orientation Normalization (Upright Coordinate Space)               │
│   • Class-Balanced Tiling (1024x1024 with 25% overlap & Box Clipping)       │
│   • Spatial-Block Cross-Validation Folds (Buffer Frames to Prevent Leakage) │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 2: MODEL TRAINING                            │
│   • Fine-tuning Ultralytics Object Detectors (YOLO11)                       │
│   • Class-Weighted Loss (Oversampling minority Chinese Tallow)              │
│   • Per-class Precision / Recall / AP50 Evaluation                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 3: SLICED INFERENCE                             │
│   • Batched FP16 Sliced Tiling Inference on New Drone Flights               │
│   • Local Tile Box Remapping to Full-Frame Upright Pixels                   │
│   • Intra-Frame Non-Maximum Suppression (NMS) Merge                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 4: 3D METRIC GEOREFERENCING                         │
│   • Photogrammetric Projection (PPK position + Camera Rotation Matrix)      │
│   • Cross-Frame Spatial Deduplication (Ground Distance Clustering)          │
│   • Confidence Calibration & Heuristic Uncertainty Estimation               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 5: FIELD DELIVERABLES                             │
│   • QGIS / QField Ready: GeoPackage (.gpkg), GeoJSON, Shapefile (.shp), CSV │
│   • Species Count, Canopy Footprint Area, and Verification Status           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
invasive_pipeline/
├── configs/
│   └── default.yaml         # Central configuration: classes, sensors, tiling, thresholds
├── data/
│   ├── README.md            # Directory contract explaining data placement
│   ├── raw/                 # [GIT-IGNORED] Survey flight folders (IMAGES, OUTPUT, geotags)
│   ├── labels/              # [GIT-IGNORED] Labelbox export NDJSON files
│   └── dsm/                 # [GIT-IGNORED] Digital Surface Model rasters (optional)
├── docs/
│   ├── INVASIVE_PIPELINE_DESIGN.md       # Full mathematical & architectural specification
│   ├── INVASIVE_SPECIES_ADAPTATION_REPORT.md  # Engineering audit from legacy drainage pipeline
│   └── PIPELINE_DIAGRAM.md               # Detailed mermaid architecture diagram & status
├── scripts/
│   ├── _common.py           # Shared path helpers & root resolution
│   ├── inventory.py         # Ingests flights, validates EXIF/geotags, creates manifest.csv
│   ├── stats.py             # Computes label geometry distributions, GSD, and class balance
│   ├── build_dataset.py     # Generates YOLO tile datasets with spatial-block folds
│   ├── train.py             # Trains YOLO detector with configured hyperparameters
│   ├── infer.py             # Runs sliced inference on survey flights and exports GIS files
│   └── calibrate_georef.py  # Empirically validates camera focal length & rotation convention
├── src/
│   └── invasive/            # Reusable core Python package
│       ├── __init__.py      # Package entry point
│       ├── config.py        # YAML configuration loader & validation schema
│       ├── io.py            # EXIF orientation handling, safe image loading, azcopy cleanup
│       ├── flights.py       # Wingtra PPK geotag reader & flight metadata extraction
│       ├── tiling.py        # Sliding window tiling with fractional box visibility clipping
│       ├── dataset.py       # Manifest builder, spatial splits, and YOLO annotation writer
│       ├── labelbox.py      # NDJSON parser supporting both modern & legacy Labelbox schemas
│       ├── georef.py        # 3D pinhole camera projection (omega/phi/kappa -> WGS84)
│       ├── infer.py         # Sliced frame inference orchestrator
│       ├── merge.py         # Class-aware NMS & intra-frame duplicate resolution
│       ├── metrics.py       # Ground-truth evaluation metrics (AP50, Precision, Recall, F1)
│       └── export.py        # Geospatial exporter (GeoPackage, GeoJSON, Shapefile, CSV)
├── tests/
│   ├── conftest.py          # Pytest fixtures and mock survey frames
│   ├── test_tiling.py       # Unit tests for patch slicing and bounding box clipping
│   ├── test_io_labelbox.py  # Tests for image IO, EXIF rotation, and NDJSON parsing
│   ├── test_georef_merge_metrics.py  # Tests for PPK projection, NMS, and evaluation
│   └── test_real_data_smoke.py       # Non-destructive smoke tests on real survey data
├── requirements.txt         # Pinned production dependencies
├── .gitignore               # Comprehensive confidentiality guardrails
└── README.md                # Project documentation
```

---

## ⚡ Quick Start Guide

### 1. Installation & Environment

Clone the repository and set up a Python 3.10+ virtual environment:

```bash
# Clone the repository
git clone https://github.com/Varun072006/TEXAS-IDRT-Invasive-Species-Detection.git
cd TEXAS-IDRT-Invasive-Species-Detection

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Verification Suite

Run all 44 unit and smoke tests to verify mathematical correctness, IO routines, and tiling logic:

```bash
python -m pytest tests -q
```
*(Expected output: `44 passed in ~28s`)*

---

### 3. End-to-End Workflow

#### Step 1: Ingest Flights & Generate Inventory Manifest
Scans `data/raw/` for survey flights, verifies image integrity (excluding partial `.azDownload` files), normalizes EXIF orientations, matches PPK geotags, and calculates Ground Sampling Distance (GSD):

```bash
python scripts/inventory.py
```
*Outputs: `work/manifest.csv` and a detailed console summary of all available survey flights.*

#### Step 2: Annotation Quality Assurance & Scale Analysis
Analyze your Labelbox annotation export to inspect bounding box sizes, label distribution, and class balance:

```bash
python scripts/stats.py --export data/labels/export.ndjson
```
*Recommends optimal tile downscale factors (e.g. 0.5x vs 1.0x) so target species match YOLO's receptive field.*

#### Step 3: Build Tiled Dataset with Spatial-Block Splits
Generates 1024x1024 image tiles with 25% overlap, clips bounding boxes, oversamples the minority class (Chinese Tallow), balances negative tiles, and creates leak-free spatial validation folds:

```bash
python scripts/build_dataset.py --export data/labels/export.ndjson --val-fold 0
```
*Outputs: `work/datasets/fold0/dataset.yaml` ready for YOLO training.*

#### Step 4: Train the Detection Model
Fine-tunes the object detector on the prepared dataset:

```bash
python scripts/train.py --dataset work/datasets/fold0/dataset.yaml --name fold0
```

#### Step 5: Sliced Inference on New Survey Flights
Executes sliced inference across ultra-high-resolution flight frames, merges overlapping tiles via NMS, projects candidate bounding boxes onto ground coordinates, deduplicates across overlapping flight passes, and exports GIS layers:

```bash
python scripts/infer.py --weights work/runs/fold0/weights/best.pt --flight "HARC Map61 60 m Flight 01"
```
*Outputs: `work/outputs/HARC_Map61_60_m_Flight_01_candidates.gpkg` (and GeoJSON/CSV).*

#### Step 6: Empirical Georeferencing Calibration
If a drone's exact camera focal length or lens calibration is unconfirmed in EXIF metadata, calibrate it against consecutive overlapping image pairs:

```bash
python scripts/calibrate_georef.py --flight "HARC Map61 60 m Flight 01"
```

---

## 🔍 Pipeline Deep-Dive

### A. Flight Ingestion & Orientation Handling
Consumer and industrial drone cameras frequently encode physical orientation flags in EXIF metadata (e.g. `Orientation 3` = rotated 180°). Annotators in web tools like Labelbox annotate the **displayed upright image**. 

The pipeline guarantees that:
- Image loading automatically resolves EXIF orientation tags into upright pixel coordinates.
- PPK flight angles ($\omega, \phi, \kappa$) are mapped consistently to the upright image frame.
- Corrupted downloads (such as Azure CLI temporary files `*.azDownload-*`) are cleanly quarantined.

### B. Sliding Window Tiling & Bounding Box Clipping
Directly downsampling a $9504 \times 6336$ px image to $640 \times 640$ destroys subtle vegetative textures needed to distinguish invasive flora from native shrubs. Instead:
- Frames are tiled into $1024 \times 1024$ patches with a 25% sliding stride.
- Ground truth boxes bisected by tile boundaries are clipped; boxes with $< 50\%$ residual area are dropped to prevent training on truncated fragments.
- Negative tiles (containing background foliage and no invasives) are kept at a configurable ratio (default 1.0) to suppress false positives.

### C. Leak-Free Spatial-Block Dataset Splitting
In fixed-wing drone mapping, consecutive frames overlap by up to 80%. Splitting individual tiles randomly between train and validation causes severe data leakage, resulting in artificially inflated test metrics that collapse in the field.

`src/invasive/dataset.py` groups consecutive frames along the flight path into **spatial blocks** (default 6 frames per block) separated by **buffer exclusion frames**, ensuring that no tree seen in the training fold appears in the validation fold.

### D. Metric Georeferencing & 3D Camera Projection
Standard GIS tools assume nadir (perpendicular) camera views. However, during automated survey flights, aircraft pitch and roll tilt the camera by 10°–15°. At 60 m flight height, a 10° pitch error offsets ground locations by over **10.5 meters**.

The georeferencing engine calculates the exact 3D camera rotation matrix $\mathbf{R} = \mathbf{R}_z(\kappa) \mathbf{R}_y(\phi) \mathbf{R}_x(\omega)$ and projects upright 2D pixel coordinates $(u, v)$ through the calibrated pinhole camera model onto the WGS84 ground plane, delivering sub-meter geospatial accuracy.

### E. Cross-Frame Deduplication & GIS Deliverables
Because each tree appears in 3 to 6 overlapping survey frames, raw frame detections produce duplicate targets. The pipeline groups ground coordinates within a metric threshold (default 1.5 m) and merges them into a single high-confidence candidate polygon with an estimated canopy footprint area.

Deliverables are exported directly into:
- **GeoPackage (`.gpkg`)**: Standard GIS format for direct import into QGIS and ArcGIS.
- **GeoJSON (`.geojson`)**: Web-ready format for web maps and dashboards.
- **Shapefile (`.shp`)**: Traditional CAD and GIS compatibility.
- **Summary CSV (`.csv`)**: Tabular report of species counts, coordinates, and confidence levels.

---

## ⚙️ Configuration Reference (`configs/default.yaml`)

All pipeline behaviors and hyperparameters are governed by a central YAML file:

```yaml
project: invasive_v1

# Target species classes (ordered by model class ID)
classes: 
  - chinese_tallow
  - mccartney_rose

# Labelbox annotation aliases (handles naming variations)
class_aliases:
  chinese_tallow: ["Chinese Tallow", "Chinese Tallow Tree", "chinese_tallow"]
  mccartney_rose: ["Macartney Rose", "McCartney Rose", "Mccartney rose"]

# Tiling configuration
tiling:
  tile_px: 1024        # Patch size fed to detector
  scale: 0.5           # Downscale ratio (0.5x = ~2.6 cm/px)
  overlap: 0.25        # 25% overlap between adjacent patches
  min_visible: 0.5     # Discard clipped boxes with < 50% visibility

# Cross-validation & sampling strategy
dataset:
  neg_tile_ratio: 1.0                       # 1 negative patch per positive patch
  extra_tiles_per_box: {chinese_tallow: 2}  # Oversample minority class
  split:
    strategy: spatial_blocks
    folds: 5
    block_size: 6      # 6 consecutive frames per block
    buffer_frames: 1   # Exclusion frame between blocks

# Camera sensor profiles (focal length & sensor dimensions)
camera:
  nominal_agl_m: 60.0  # Above Ground Level (AGL) height in meters
  per_flight:
    HARC: {sensor_width_mm: 35.7, focal_mm: 17.0, angles_in_upright: false}
    DPP:  {sensor_width_mm: 23.5, focal_mm: 12.0, angles_in_upright: true}

# Georeferencing & deduplication
georef:
  method: wingtra_ppk
  rotation_variant: pix4d
  dedup_distance_m: 1.5 # Merge targets within 1.5 meters on the ground
```

---

## 🛰️ Sensors & Photogrammetric Validation

The pipeline includes built-in sensor profiles for industrial aerial mapping hardware:

| Flight / Platform | Sensor | Resolution | Lens | Ground Resolution (GSD) | Georef Residual Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HARC Map61 Flight** | Sony ILX-LR1 Full Frame | $9504 \times 6336$ (61 MP) | 17 mm | ~1.3 cm/pixel at 60 m | **0.77 m** (Pix4D rotation) |
| **DPP EO Flight** | Sony ILCE-6100 APS-C | $6000 \times 4000$ (24 MP) | 12 mm | ~1.8 cm/pixel at 60 m | **0.49 m** (Upright calibration) |

*Georeferencing residuals are calculated by `scripts/calibrate_georef.py` measuring the ground agreement of identical landscape points across consecutive overlapping frames.*

---

## 🛡️ Confidentiality & Data Protection Guardrails

To protect institutional partner agreements and private landholder data:

> [!IMPORTANT]
> **No raw drone imagery, confidential flight logs, ground survey coordinates, or proprietary client datasets are tracked or hosted on this repository.**

- **Strict Exclusion**: All `data/raw/*`, `data/labels/*`, `*.jpg`, `*.tif`, `*.ulg`, `*.26o`, and `*.json` survey files are permanently excluded via [`.gitignore`](file:///C:/Projects/Invasive%20Species%20Detection/invasive_pipeline/.gitignore).
- **Public Portfolio Demonstration**: This repository contains the **complete algorithmic implementation, unit test suite, and configuration scaffolding**. Users can run the pipeline on synthetic mock frames or supply their own survey flights following [`data/README.md`](file:///C:/Projects/Invasive%20Species%20Detection/invasive_pipeline/data/README.md).

---

## 📜 Contributing & License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

Developed as part of the **Texas IDRT Invasive Species Detection & Aerial Monitoring Initiative**. For inquiries regarding methodology, dataset collection protocols, or field verification integrations, please open an issue in this repository.
