"""Run detection + georeferencing on one flight and export GeoPackage/GeoJSON/Shapefile/CSV.

Usage: python scripts/infer.py --weights runs/fold0/weights/best.pt --flight HARC [--limit 5]
Needs scripts/inventory.py to have written the manifest.
"""
import argparse

import _common  # noqa: F401
import pandas as pd
from tqdm import tqdm

from invasive.config import load_config, resolve_path
from invasive.export import detections_to_gdf, export_all, summarise
from invasive.flights import find_geotag_csv, load_wingtra_geotags
from invasive.infer import UltralyticsDetector, run_flight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--flight", required=True, help="substring of the flight folder name")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N readable images (testing)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = resolve_path(cfg, "data_root")
    work = resolve_path(cfg, "workdir")
    manifest = pd.read_csv(work / "manifest.csv")
    flights = sorted(f for f in manifest.flight_id.unique() if args.flight in f)
    if len(flights) != 1:
        raise SystemExit(f"--flight must match exactly one flight, got {flights}")
    flight = flights[0]
    m = manifest[(manifest.flight_id == flight) & manifest.readable].sort_values("sequence")
    if args.limit:
        m = m.head(args.limit)
    tags = load_wingtra_geotags(find_geotag_csv(root / flight))

    detector = UltralyticsDetector(args.weights, conf=cfg["infer"]["conf_floor"], imgsz=cfg["tiling"]["tile_px"],
                                   augment=cfg["infer"]["tta"])
    df = run_flight(m, tags, detector, cfg, model_version=str(args.weights), progress=lambda x: tqdm(x, desc=flight))

    out = work / "outputs" / flight.replace(" ", "_")
    paths = export_all(detections_to_gdf(df), out, "detections", cfg["export"]["formats"])
    summ = summarise(df)
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.to_string(index=False))
    print("written:", *[str(p) for p in paths], sep="\n  ")


if __name__ == "__main__":
    main()
