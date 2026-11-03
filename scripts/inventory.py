"""Inventory of the raw flights: readability, sizes, orientation, geotag coverage, frame spacing.

Writes <workdir>/manifest.csv and prints a report. Needs no labels.
Usage: python scripts/inventory.py [--config cfg.yaml] [--no-decode]
"""
import argparse

import _common  # noqa: F401
import numpy as np

from invasive.config import load_config, resolve_path
from invasive.dataset import build_manifest
from invasive.flights import consecutive_spacing_m, find_geotag_csv, load_wingtra_geotags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-decode", action="store_true", help="skip full decode (faster, misses truncated files)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = resolve_path(cfg, "data_root")
    work = resolve_path(cfg, "workdir")
    work.mkdir(parents=True, exist_ok=True)

    if not root.is_dir() or not any((p / "IMAGES").is_dir() for p in root.iterdir()):
        raise SystemExit(f"No flights found under {root}. Expected <flight name>/IMAGES/*.JPG and "
                         f"<flight name>/OUTPUT/<flight name> geotags.csv (see data/README.md).")
    m = build_manifest(root, full_decode=not args.no_decode)
    m.to_csv(work / "manifest.csv", index=False)

    print(f"data root: {root}")
    print(f"images found: {len(m)}  readable: {int(m.readable.sum())}  unreadable: {int((~m.readable).sum())}")
    for flight, g in m.groupby("flight_id"):
        ok = g[g.readable]
        print(f"\n== {flight}")
        print(f"   files {len(g)} | readable {len(ok)} | azcopy temp names {int(g.is_az_temp.sum())} | "
              f"unreadable {int((~g.readable).sum())}")
        if len(ok):
            print(f"   upright size(s): {sorted(set(zip(ok.width, ok.height)))} | EXIF orientation(s): {sorted(set(ok.orientation))}")
        print(f"   with PPK geotag: {int(g.has_geotag.sum())}/{len(g)}")
        csv = find_geotag_csv(root / flight)
        if csv:
            tags = load_wingtra_geotags(csv)
            sp = consecutive_spacing_m(tags)
            alt = np.array([t.alt for t in tags.values()])
            tilt = np.array([max(abs(t.omega), abs(t.phi)) for t in tags.values()])
            hacc = np.array([t.h_acc for t in tags.values()])
            print(f"   geotags {len(tags)} | spacing between frames: median {np.median(sp):.1f} m (min {sp.min():.1f}, max {sp.max():.1f})")
            print(f"   camera ellipsoidal height: {alt.min():.1f}..{alt.max():.1f} m | max(|omega|,|phi|): median {np.median(tilt):.1f} deg, max {tilt.max():.1f}")
            print(f"   PPK horizontal accuracy: median {np.median(hacc):.2f} m, max {hacc.max():.2f} m")
        missing = g[~g.has_geotag & g.readable]
        if len(missing):
            print(f"   WARNING: {len(missing)} readable images without a geotag")
    bad = m[~m.readable]
    if len(bad):
        print("\nUnreadable files (re-download these):")
        for flight, g in bad.groupby("flight_id"):
            names = ", ".join(n.split("_")[-1].split(".")[0] for n in g.image_name)
            print(f"   {flight}: {len(g)} files (frame numbers: {names})")
            print(f"      cause: {g.error.str.split(':').str[0].value_counts().to_dict()}; "
                  f"{int(g.is_az_temp.sum())} are azcopy '.azDownload-' temp files (incomplete download)")
    print(f"\nmanifest written to {work / 'manifest.csv'}")


if __name__ == "__main__":
    main()
