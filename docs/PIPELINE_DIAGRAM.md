# Pipeline diagram (matches the code as of 2026-11-03)

Colour = implementation status. Green: built and tested. Yellow: partly built. Red dashed: planned, not built.
"Built and tested" means unit tests and real-frame smoke tests with synthetic labels; no model has been trained yet.
Preview in VS Code with a Mermaid extension, or paste into https://mermaid.live.

```mermaid
flowchart TD
    subgraph S1["1. DATA SOURCES"]
        LB["Labelbox export<br/>NDJSON bounding boxes"]
        IMG["Drone RGB frames<br/>JPG + EXIF"]
        META["Wingtra flight metadata<br/>PPK geotags CSV / JSON<br/>position + omega/phi/kappa"]
    end

    subgraph S2["2. MODEL DEVELOPMENT"]
        A["A. Dataset builder<br/>labels + manifest + upright orientation<br/>per-flight camera settings"]
        QA["Statistics and label QA<br/>box sizes, GSD, negatives, QA flags"]
        VER["Dataset versioning<br/>export hash + dataset meta"]
        SPL["Grouped splitting<br/>spatial blocks of consecutive frames + buffer"]
        TRN["B. Tiled training<br/>pretrained detector, negative tiles,<br/>tallow oversampling"]
        EVAL["Model evaluation<br/>AP50 / precision / recall per class"]
        CAL["Score calibration<br/>per-class thresholds"]
        MODEL["Trained model + model version"]
        A --> QA --> VER --> SPL --> TRN --> EVAL --> CAL --> MODEL
    end

    subgraph S3["3. DETECTION PIPELINE (new flights)"]
        C["C. Ingestion and QC<br/>readable file, EXIF orientation,<br/>geotag present, GSD, blur"]
        D["D. Tiling<br/>1024 px model input, scale and overlap from config"]
        INF["Sliced model inference<br/>batched, FP16"]
        MAP["E. Coordinate mapping<br/>tile boxes to frame boxes"]
        NMS["Merge duplicates<br/>NMS, weighted box fusion"]
        TIER["Per-class tiers<br/>candidate / high confidence"]
        C --> D --> INF --> MAP --> NMS --> TIER
    end

    subgraph S4["4. GEOSPATIAL PROCESSING"]
        F["F. Georeferencer"]
        PPK["PPK geotag projection<br/>omega/phi/kappa + camera model"]
        ODM["ODM camera model"]
        MOS["Orthomosaic georeferencing"]
        VAL["Georeference validation<br/>consecutive-frame overlap check"]
        DEDUP["Cross-frame deduplication<br/>world coordinates + number of views"]
        F --> PPK
        F --> ODM
        F --> MOS
        PPK --> DEDUP
        ODM --> DEDUP
        MOS --> DEDUP
        VAL -.checks.-> PPK
    end

    subgraph S5["5. OUTPUT AND FIELD VERIFICATION"]
        G["G. Geospatial exports<br/>GeoPackage / GeoJSON / SHP / CSV"]
        SUM["Summary CSV<br/>counts + candidate-region area"]
        PREV["Preview images + map"]
        H["H. Human verification<br/>QGIS / QField / Labelbox"]
        IMP["Import reviewed status"]
        DEC{"Review status"}
        VERD["Verified<br/>new positives"]
        REJ["Rejected<br/>hard negatives"]
        COR["Corrected labels<br/>fixed boxes, missed objects"]
        G --> SUM
        G --> PREV
        G --> H --> IMP --> DEC
        DEC --> VERD
        DEC --> REJ
        DEC --> COR
    end

    LB --> A
    IMG --> A
    META --> A
    IMG ==>|new flights| C
    META --> C
    MODEL ==>|weights| INF
    TIER --> F
    META --> F
    DEDUP --> G
    VERD -.feedback.-> A
    REJ -.feedback.-> A
    COR -.feedback.-> A

    classDef built fill:#d4edda,stroke:#2e7d32,color:#000;
    classDef partial fill:#fff3cd,stroke:#b28704,color:#000;
    classDef planned fill:#f8d7da,stroke:#b71c1c,stroke-dasharray:5 3,color:#000;
    classDef source fill:#e3f2fd,stroke:#1565c0,color:#000;

    class LB,IMG,META source;
    class A,QA,SPL,TRN,D,MAP,TIER,F,PPK,VAL,DEDUP,G,SUM built;
    class VER,EVAL,MODEL,C,INF,NMS partial;
    class CAL,ODM,MOS,PREV,IMP planned;
    class H,DEC,VERD,REJ,COR partial;
```

## Status notes

| Box | State | Detail |
|---|---|---|
| A, QA, SPL | built | `build_dataset.py`, `stats.py`; spatial-block folds exist because consecutive frames overlap 60-80% |
| TRN | built, never run | `train.py` wraps Ultralytics (AGPL licence); no labels or weights exist yet |
| VER | partial | hash of the export and a `dataset_meta.json`; no manifest hash or dataset registry |
| EVAL | partial | `metrics.py` (AP50, precision, recall, F1) exists; no evaluation script wires it to a held-out fold |
| CAL | planned | thresholds are fixed values in `configs/default.yaml` until calibrated on validation data |
| MODEL | partial | the weights path is stored as the model version; no registry |
| C | partial | corrupt/incomplete file detection, EXIF orientation, geotag presence done; blur check not done; GSD is computed in `stats.py` only |
| D | built | tile covers `tile_px / scale` source pixels; the scale is not decided yet (needs real box sizes) |
| INF | partial | batched inference works through the detector wrapper; FP16 is not enabled explicitly |
| NMS | partial | class-aware NMS only; weighted box fusion not implemented |
| TIER | built | thresholds are uncalibrated placeholders |
| PPK | built | validated by overlap: HARC 0.77 m, DPP EO 0.49 m median disagreement; not checked against field points |
| ODM, MOS | planned | need photogrammetry output or an orthomosaic |
| DEDUP, G, SUM | built | area is the footprint of detection boxes, not canopy cover |
| PREV | planned | preview images and map |
| H, IMP, DEC | external / planned | reviewers work in QGIS, QField or Labelbox; the `status` column exists, but no code reads reviewed results back |
| Feedback loop | planned | verified becomes positives, rejected becomes hard negatives, corrected becomes fixed labels, all re-entering at A |

## Changes from the earlier diagram

1. Labelbox feeds only the dataset builder (it was wired into inference).
2. New flights enter at ingestion; the trained model feeds inference as weights.
3. Added the third input, Wingtra PPK geotags. EXIF has no GPS, so "EXIF flat projection" is replaced by PPK geotag projection.
4. Added the georeference validation step.
5. All three review outcomes feed back; "candidate" is the initial status before review.
6. Removed the unexplained outer loop line.
7. Boxes are marked built, partial or planned instead of implying everything exists.
