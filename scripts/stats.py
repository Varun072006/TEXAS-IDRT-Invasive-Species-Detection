"""Label statistics report: the first thing to run when the Labelbox export arrives.

Answers the [DATA] questions of the design: box counts per class/image, box sizes in pixels and metres,
skipped images, label/image mismatches, QA flags, and a suggested tiling scale.
Usage: python scripts/stats.py --export path/to/export.ndjson   (run scripts/inventory.py first)
"""
import argparse

import _common  # noqa: F401
import numpy as np
import pandas as pd

from invasive.config import camera_for_flight, load_config, resolve_path
from invasive.dataset import label_qa
from invasive.labelbox import parse_export


def gsd_cm(cfg, flight_id, width):
    cam = camera_for_flight(cfg, flight_id)
    if not cam.get("focal_mm"):
        return np.nan
    focal_px = cam["focal_mm"] / cam["sensor_width_mm"] * width
    return 100.0 * cfg["camera"]["nominal_agl_m"] / focal_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--target-box-px", type=float, default=128.0,
                    help="desired median box side in model-input pixels, used to suggest a scale")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = resolve_path(cfg, "workdir")
    manifest = pd.read_csv(work / "manifest.csv")

    labels, images, warnings = parse_export(args.export, cfg)
    qa = label_qa(labels, manifest)
    m = manifest.drop_duplicates("image_name").set_index("image_name")

    lines = ["# Label statistics", ""]
    lines.append(f"- export rows (images): {len(images)} | skipped: {int(images.skipped.sum())} | "
                 f"images with >=1 box: {int((images.n_objects > 0).sum())}")
    lines.append(f"- boxes: {len(labels)}")
    lines.append("\n## Boxes per class\n")
    lines.append(labels["class"].value_counts().rename_axis("class").reset_index(name="boxes").to_markdown(index=False))

    per_img = labels.groupby(["image_name", "class"]).size().unstack(fill_value=0)
    lines.append("\n## Boxes per image (images that have boxes)\n")
    lines.append(per_img.describe().round(2).to_markdown())

    labels = labels.assign(w=labels.x2 - labels.x1, h=labels.y2 - labels.y1)
    labels["side"] = np.sqrt(labels.w * labels.h)
    labels["flight_id"] = labels.image_name.map(m.flight_id)
    labels["img_w"] = labels.image_name.map(m.width)
    labels["gsd_cm"] = [gsd_cm(cfg, f, w) if isinstance(f, str) else np.nan
                        for f, w in zip(labels.flight_id, labels.img_w)]
    labels["side_m"] = labels.side * labels.gsd_cm / 100.0
    lines.append("\n## Box size (geometric mean of width and height)\n")
    size = labels.groupby("class").agg(n=("side", "size"), px_p10=("side", lambda s: s.quantile(.1)),
                                       px_median=("side", "median"), px_p90=("side", lambda s: s.quantile(.9)),
                                       m_median=("side_m", "median"), m_p90=("side_m", lambda s: s.quantile(.9)))
    lines.append(size.round(1).to_markdown())

    lines.append("\n## Suggested tiling scale\n")
    for cname, row in size.iterrows():
        if row.px_median > 0:
            lines.append(f"- {cname}: median box {row.px_median:.0f} px in the frame -> scale about "
                         f"{min(1.0, args.target_box_px / row.px_median):.2f} for a {args.target_box_px:.0f} px median box")
    lines.append("- If the two classes need very different scales, use two scale passes (see design 4.2).")

    lines.append("\n## Boxes by flight\n")
    lines.append(labels.groupby(["flight_id", "class"]).size().unstack(fill_value=0).to_markdown())

    lines.append("\n## Consistency with the image inventory\n")
    exp_names, man_names = set(images.image_name), set(m.index)
    lines.append(f"- in export but not on disk: {len(exp_names - man_names)} {sorted(exp_names - man_names)[:10]}")
    lines.append(f"- on disk but not in export: {len(man_names - exp_names)}")
    unread = sorted(set(labels.image_name) & set(m.index[~m.readable]))
    lines.append(f"- labelled images that are unreadable on disk: {len(unread)} {unread[:10]}")
    wh = images.dropna(subset=["width"]).merge(m[["width", "height"]].reset_index(), on="image_name",
                                               suffixes=("_lb", "_disk"))
    swapped = wh[(wh.width_lb == wh.height_disk) & (wh.height_lb == wh.width_disk) & (wh.width_lb != wh.width_disk)]
    mism = wh[(wh.width_lb != wh.width_disk) & ~wh.index.isin(swapped.index)]
    lines.append(f"- Labelbox image size differs from the upright size on disk: {len(mism)} (rotated 90 deg: {len(swapped)}). "
                 "Any mismatch means boxes are in a different pixel space: resolve before training.")

    lines.append("\n## Label QA flags\n")
    flag_cols = [c for c in qa.columns if c.startswith("flag_")]
    lines.append(qa[flag_cols].sum().rename("boxes").to_frame().to_markdown())

    if warnings:
        lines.append(f"\n## Parser warnings ({len(warnings)})\n")
        lines.extend(f"- {w}" for w in warnings[:30])

    report = "\n".join(lines)
    (work / "stats_report.md").write_text(report, encoding="utf-8")
    qa.to_csv(work / "labels_qa.csv", index=False)
    print(report)
    print(f"\nwritten: {work / 'stats_report.md'}, {work / 'labels_qa.csv'}")


if __name__ == "__main__":
    main()
