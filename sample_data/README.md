# Sample Data

This directory intentionally ships **without any imagery** — no real drone
captures, no partner/institution datasets, no derived crops.

## Why

The original project processes confidential drone survey imagery collected
for a specific field engagement. That imagery, along with any associated
GPS metadata, flight logs, or site identifiers, is not published here.

## What to use instead

Every module and the demo notebook in this repository runs entirely on
**synthetic, procedurally generated imagery** produced at runtime by
[`src/utils/synthetic_data.py`](../src/utils/synthetic_data.py). It creates
a fake "orthomosaic" with randomly scattered blob-shaped tree-crown regions
— sufficient to exercise every pipeline stage (tiling, segmentation,
candidate extraction, crown separation, duplicate merging, embeddings,
clustering) without needing any external download.

To try the pipeline on your own (non-confidential) imagery instead, drop
`.tif` / `.jpg` / `.png` orthomosaic files into this folder locally (it is
excluded from version control beyond this README) and point
[`notebooks/demo_pipeline.ipynb`](../notebooks/demo_pipeline.ipynb) at them.
