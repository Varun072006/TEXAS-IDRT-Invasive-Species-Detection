# Pipeline Workflow

Step-by-step walkthrough of how a drone orthomosaic flows through the
system, matching the stages implemented under [`src/`](../src/) and
demonstrated in [`notebooks/demo_pipeline.ipynb`](../notebooks/demo_pipeline.ipynb).

## Step 1 — Ingest
A raw drone orthomosaic (or, for this public demo, a synthetically
generated stand-in from `src/utils/synthetic_data.py`) enters the pipeline.
In production this file is pulled from Azure Blob Storage.

## Step 2 — Preprocess
`src/preprocessing/image_preprocessing.py` denoises the image and applies
CLAHE illumination correction so brightness/contrast is consistent
regardless of flight conditions.

## Step 3 — Tile
`src/preprocessing/tiling.py` splits the (potentially gigapixel) image into
overlapping square tiles. Overlap guarantees a tree near a tile boundary is
fully visible in at least one tile.

## Step 4 — Segment
Each tile is passed to `src/segmentation/sam_segmentation.py`. In
production this runs Meta's Segment Anything Model automatic mask
generator; the demo uses a classical color-thresholding fallback so no
model weights need to be downloaded.

## Step 5 — Extract candidates
`src/detection/tree_candidate_extraction.py` filters raw masks down to
regions with a plausible tree-crown size and circularity, discarding
shadows, roads, and other non-crown regions.

## Step 6 — Separate merged crowns
`src/detection/crown_separation.py` applies a watershed transform to split
masks that actually contain multiple touching/overlapping crowns (common
in dense canopy) into individual candidates.

## Step 7 — Merge duplicates
`src/detection/duplicate_merging.py` converts tile-local candidates into
full-scene coordinates and runs a greedy IoU-based NMS pass to collapse
duplicate detections that appear in more than one overlapping tile.

## Step 8 — Embed
`src/clustering/visual_embeddings.py` crops each final detection and embeds
it into a fixed-size feature vector. Production uses DINO-family
self-supervised embeddings; the demo uses a lightweight color/texture
descriptor.

## Step 9 — Cluster
`src/clustering/cluster_analysis.py` groups embeddings with KMeans or
HDBSCAN, surfacing visually distinct groups of crowns for expert review —
an exploratory precursor to supervised species classification.

## Step 10 — Export
`src/utils/geo_utils.py` converts final detections into a GeoDataFrame and
exports GeoJSON for use in GIS tools.

## Planned next steps
- Supervised species classification on top of clustered/labeled crowns.
- Full Azure ML deployment for scalable batch inference.
- Odoo-based reporting/dashboarding for field operations and stakeholders.

See [architecture.md](architecture.md) for the system diagram and design
rationale, and the README's **Project Status & Roadmap** section for what
is implemented today versus planned.
