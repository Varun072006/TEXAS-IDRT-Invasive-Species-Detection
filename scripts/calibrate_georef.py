"""Check the georeferencing model against the images themselves (no field points needed).

For consecutive frames, SIFT matches (RANSAC-filtered) give pixel pairs that show the same ground
point. Projecting both pixels to the ground with the PPK geotags must give (nearly) the same place.
The median distance between the two projections is the residual. We scan rotation conventions and,
if requested, focal lengths, and report which gives the smallest residual.

Residuals include real parallax from trees/buildings (a few metres), so compare candidates against
each other, not against zero. Usage:
  python scripts/calibrate_georef.py --flight HARC --pairs 8
  python scripts/calibrate_georef.py --flight "DPP EO" --scan-focal 12 35 1
"""
import argparse

import _common  # noqa: F401
import cv2
import numpy as np

from invasive.config import camera_for_flight, load_config, resolve_path
from invasive.flights import estimate_ground_height, find_geotag_csv, frame_sequence, load_wingtra_geotags
from invasive.georef import (ROTATION_VARIANTS, CameraIntrinsics, pixel_to_enu_offset, rotation_matrix,
                             utm_crs_for)
from pyproj import Transformer


def load_gray_raw(path, target_w=1800):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    s = target_w / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s, img.shape[1], img.shape[0]


def match_pair(path_a, path_b):
    ga, s, w, h = load_gray_raw(path_a)
    gb, _, _, _ = load_gray_raw(path_b)
    sift = cv2.SIFT_create(nfeatures=4000)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    knn = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    if mask is None:
        return None
    inl = mask.ravel().astype(bool)
    return pa[inl] / s, pb[inl] / s, w, h       # raw pixel coordinates in full resolution


def residual(pa, pb, ga, gb, intr, ground, variant, to_utm):
    ea, na = to_utm.transform(ga.lon, ga.lat)
    eb, nb = to_utm.transform(gb.lon, gb.lat)
    ra = rotation_matrix(ga.omega, ga.phi, ga.kappa, variant)
    rb = rotation_matrix(gb.omega, gb.phi, gb.kappa, variant)
    qa = pixel_to_enu_offset(pa[:, 0], pa[:, 1], intr, ra, ga.alt - ground) + [ea, na]
    qb = pixel_to_enu_offset(pb[:, 0], pb[:, 1], intr, rb, gb.alt - ground) + [eb, nb]
    d = np.hypot(*(qa - qb).T)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--flight", required=True, help="substring of the flight folder name")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--angles-in-upright-image", action="store_true",
                    help="treat omega/phi/kappa as referring to the EXIF-upright image (flip raw pixels 180 deg first)")
    ap.add_argument("--scan-focal", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"), default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = resolve_path(cfg, "data_root")
    flight_dir = next(p for p in sorted(root.iterdir()) if args.flight in p.name and (p / "IMAGES").is_dir())
    tags = load_wingtra_geotags(find_geotag_csv(flight_dir))
    ground = estimate_ground_height(tags, cfg["camera"]["nominal_agl_m"])
    cam = camera_for_flight(cfg, flight_dir.name)

    names = sorted(tags, key=frame_sequence)
    files = {}
    for p in (flight_dir / "IMAGES").iterdir():
        from invasive.io import canonical_name, check_image
        files[canonical_name(p)] = p
    usable = [n for n in names if n in files and check_image(files[n], full_decode=False).readable]
    pairs = [(a, b) for a, b in zip(usable, usable[1:]) if frame_sequence(b) - frame_sequence(a) == 1]
    step = max(1, len(pairs) // args.pairs)
    pairs = pairs[::step][:args.pairs]
    print(f"flight {flight_dir.name}: {len(pairs)} consecutive pairs, ground height {ground:.1f} m (ellipsoidal), "
          f"camera {cam}")

    matches = []
    for a, b in pairs:
        r = match_pair(files[a], files[b])
        if r is not None:
            matches.append((a, b, *r))
            print(f"  {a} -> {b}: {len(r[0])} inlier matches")
    if not matches:
        raise SystemExit("no usable matches")

    w, h = matches[0][4], matches[0][5]
    if args.angles_in_upright_image:
        matches = [(a, b, np.column_stack([w - pa[:, 0], h - pa[:, 1]]), np.column_stack([w - pb[:, 0], h - pb[:, 1]]), w_, h_)
                   for a, b, pa, pb, w_, h_ in matches]
        print("pixel coordinates flipped 180 deg (angles assumed to refer to the upright image)")
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs_for(tags[usable[0]].lat, tags[usable[0]].lon), always_xy=True)
    sensor = cam["sensor_width_mm"]
    if args.scan_focal:
        focals = np.arange(args.scan_focal[0], args.scan_focal[1] + 1e-9, args.scan_focal[2])
    else:
        if not cam.get("focal_mm"):
            raise SystemExit("focal_mm unknown for this flight: pass --scan-focal MIN MAX STEP")
        focals = np.array([cam["focal_mm"]])

    print("\nmedian ground disagreement between consecutive frames (metres); lower is better")
    print(f"{'focal_mm':>9} | " + " | ".join(f"{v:>13}" for v in ROTATION_VARIANTS))
    best = (1e9, None, None)
    for f in focals:
        intr = CameraIntrinsics(w, h, float(f), sensor)
        row = []
        for v in ROTATION_VARIANTS:
            vals = [residual(pa, pb, tags[a], tags[b], intr, ground, v, to_utm) for a, b, pa, pb, _, _ in matches]
            m = float(np.nanmedian(vals))
            row.append(m)
            if m < best[0]:
                best = (m, f, v)
        print(f"{f:9.1f} | " + " | ".join(f"{x:13.2f}" for x in row))
    print(f"\nbest: variant={best[2]} focal={best[1]:.1f} mm, median residual {best[0]:.2f} m")


if __name__ == "__main__":
    main()
