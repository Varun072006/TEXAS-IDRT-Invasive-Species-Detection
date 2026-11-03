# data/ : where the images and labels go

Nothing in here is tracked by git (see ../.gitignore). Drop new material in as follows.

```
data/
  raw/
    <flight name>/                      e.g. "HARC Map61 60 m Flight 01"
      IMAGES/                           the drone frames (.JPG)
      OUTPUT/<flight name> geotags.csv  WingtraHub PPK geotags (position + omega/phi/kappa)
  labels/
    export.ndjson                       Labelbox export (bounding boxes)
  dsm/                                  optional: elevation rasters, for later georeferencing improvements
```

After adding data:

```
python scripts/inventory.py                      # checks every image (corrupt/incomplete files are listed)
python scripts/stats.py --export data/labels/export.ndjson
python scripts/build_dataset.py --export data/labels/export.ndjson --val-fold 0
```

Rules:
- One folder per flight, named like the Wingtra flight (the name selects the per-flight camera settings in
  `configs/default.yaml`: a new camera/flight type needs a `camera.per_flight` entry).
- Do not copy `.azDownload-*` files: they are incomplete downloads. Re-download them.
- Keep the original file names; Labelbox external IDs are matched on them.
