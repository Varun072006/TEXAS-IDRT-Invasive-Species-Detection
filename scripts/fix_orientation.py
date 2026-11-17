"""Create data/images/corrected: EXIF-safe copies of every image + unchanged labels + mapping + verification.

Root cause (see corrected/VERIFICATION_REPORT.md): Labelbox boxes are in the STORED pixel grid (what Labelbox shows),
but the JPGs carry EXIF Orientation=3. Anything that applies EXIF (export_annotations.py previews, browsers, PIL
exif_transpose) shows the image rotated 180 deg while the box coordinates stay put. The fix keeps the pixel grid and
resets the Orientation tag to 1 (lossless, 2 bytes), so every viewer agrees with Labelbox. Boxes are NOT transformed.

Usage: python scripts/fix_orientation.py [--samples 6]
Originals (data/images/*.JPG, labels, preview, ndjson) are never modified.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from invasive.exif_fix import set_exif_orientation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "images"
OUT = SRC / "corrected"
NDJSON = ROOT / "data" / "Export  project - Invasive Species - 11_10_2026.ndjson"
COLORS = {0: "red", 1: "yellow"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def overlay(img_path: Path, lab_path: Path, max_side=2000, width=3) -> Image.Image:
    """Draw labels on the image exactly as an EXIF-aware viewer would show it (no manual rotation)."""
    im = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    W, H = im.size
    k = max_side / max(W, H)
    im = im.resize((int(W * k), int(H * k)))
    d = ImageDraw.Draw(im)
    for ln in lab_path.read_text().splitlines():
        if not ln.strip():
            continue
        c, cx, cy, w, h = ln.split()
        c = int(c)
        cx, cy, w, h = float(cx) * im.width, float(cy) * im.height, float(w) * im.width, float(h) * im.height
        d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=COLORS[c], width=width)
    return im


def titled(im: Image.Image, text: str, width=1000) -> Image.Image:
    im = im.resize((width, int(im.height * width / im.width)))
    canvas = Image.new("RGB", (im.width, im.height + 28), "black")
    canvas.paste(im, (0, 28))
    ImageDraw.Draw(canvas).text((8, 8), text, fill="white")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()
    for sub in ("labels", "preview", "comparison"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    lb = {}
    for line in NDJSON.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            objs = [o for p in r["projects"].values() for lab in p["labels"] for o in lab["annotations"]["objects"]]
            lb[r["data_row"]["external_id"]] = (r["media_attributes"], objs)

    rows = []
    for src in sorted(SRC.glob("*.JPG")):
        name = src.name
        data = src.read_bytes()
        new, old_o, pos = set_exif_orientation(data, 1)
        dst = OUT / name
        dst.write_bytes(new)
        lab_src = SRC / "labels" / f"{src.stem}.txt"
        lab_dst = OUT / "labels" / lab_src.name
        shutil.copyfile(lab_src, lab_dst)  # labels are copied byte-for-byte, never transformed

        with Image.open(src) as a, Image.open(dst) as b:
            ra, rb = np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))
            upright_new = np.asarray(ImageOps.exif_transpose(b).convert("RGB"))
            new_o = b.getexif().get(0x0112, 1)
            W, H = a.size
        diff_bytes = int(sum(x != y for x, y in zip(data, new))) if len(data) == len(new) else -1
        ma, objs = lb[name]
        txt = [l.split() for l in lab_dst.read_text().splitlines() if l.strip()]
        box_ok = len(txt) == len(objs)
        max_err_px = 0.0
        if box_ok:
            for ob, t in zip(objs, txt):
                b_ = ob["bounding_box"]
                gt = np.array([b_["left"] + b_["width"] / 2, b_["top"] + b_["height"] / 2, b_["width"], b_["height"]])
                pr = np.array([float(t[1]) * W, float(t[2]) * H, float(t[3]) * W, float(t[4]) * H])
                max_err_px = max(max_err_px, float(np.abs(gt - pr).max()))
                box_ok &= int(t[0]) == (0 if ob["value"] == "macartney_rose" else 1)
        inside = all(float(t[1]) - float(t[3]) / 2 >= -1e-6 and float(t[1]) + float(t[3]) / 2 <= 1 + 1e-6
                     and float(t[2]) - float(t[4]) / 2 >= -1e-6 and float(t[2]) + float(t[4]) / 2 <= 1 + 1e-6 for t in txt)
        rows.append(dict(
            image=name, original_path=str(src.relative_to(ROOT)), corrected_path=str(dst.relative_to(ROOT)),
            label_original=str(lab_src.relative_to(ROOT)), label_corrected=str(lab_dst.relative_to(ROOT)),
            exif_orientation_original=old_o, exif_orientation_corrected=new_o,
            stored_width=W, stored_height=H, labelbox_width=ma["width"], labelbox_height=ma["height"],
            labelbox_exif_rotation=ma.get("exif_rotation"),
            size_matches_labelbox=(W, H) == (ma["width"], ma["height"]),
            bytes_changed=diff_bytes, pixels_identical=bool(np.array_equal(ra, rb)),
            exif_aware_view_equals_stored_pixels=bool(np.array_equal(upright_new, ra)),
            labels_byte_identical=sha(lab_src) == sha(lab_dst), boxes_match_labelbox=bool(box_ok),
            max_box_error_px=round(max_err_px, 3), boxes_inside_image=bool(inside), n_boxes=len(txt),
            n_rose=sum(t[0] == "0" for t in txt), n_tallow=sum(t[0] == "1" for t in txt)))
        overlay(dst, lab_dst).save(OUT / "preview" / f"{src.stem}.jpg", quality=85)  # the old, misaligned data/images/preview/ was deleted by the user
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mapping.csv", index=False)
    shutil.copyfile(SRC / "classes.txt", OUT / "classes.txt")

    # annotations.json: identical except the (wrong) coordinate-frame text; boxes untouched
    orig_ann = json.loads((SRC / "annotations.json").read_text(encoding="utf-8"))
    ann = json.loads((SRC / "annotations.json").read_text(encoding="utf-8"))
    ann["info"]["coordinate_frame"] = (
        "pixels of the STORED image grid as shown by Labelbox (6000x4000); the JPGs carry EXIF orientation 3 which is "
        "NOT applied to these coordinates; the corrected/ copies reset the tag to 1")
    ann["info"]["orientation_fix"] = "see data/images/corrected/VERIFICATION_REPORT.md; bounding boxes unchanged"
    for im in ann["images"]:
        im["corrected_file"] = f"corrected/{im['file_name']}"
        im["corrected_preview"] = f"corrected/preview/{Path(im['file_name']).stem}.jpg"
    (OUT / "annotations.json").write_text(json.dumps(ann, indent=1), encoding="utf-8")
    boxes_same = orig_ann["annotations"] == ann["annotations"] and orig_ann["categories"] == ann["categories"]

    # before/after sheets: the image from the Labelbox screenshot plus evenly spaced others that contain Tallow
    pick = ["AB_Oppenheimer_Flight_01_00023.JPG"]
    cand = df[(df.n_tallow > 0) & ~df.image.isin(pick)].image.tolist()
    step = max(1, len(cand) // max(1, args.samples - 1))
    pick += cand[::step][:args.samples - 1]
    for name in pick:
        stem = Path(name).stem
        if not (SRC / "preview" / f"{stem}.jpg").exists():
            continue  # old misaligned preview folder was deleted; existing comparison sheets are kept
        before = Image.open(SRC / "preview" / f"{stem}.jpg").convert("RGB")
        after = Image.open(OUT / "preview" / f"{stem}.jpg").convert("RGB")
        a = titled(before, f"BEFORE: old preview ({stem}): rotated 180, boxes misplaced")
        b = titled(after, "AFTER: corrected (stored grid = Labelbox view)")
        sheet = Image.new("RGB", (a.width + b.width + 10, max(a.height, b.height)), "black")
        sheet.paste(a, (0, 0))
        sheet.paste(b, (a.width + 10, 0))
        sheet.save(OUT / "comparison" / f"{stem}.jpg", quality=88)

    ok = df[["size_matches_labelbox", "pixels_identical", "exif_aware_view_equals_stored_pixels", "labels_byte_identical",
             "boxes_match_labelbox", "boxes_inside_image"]].all(axis=1)
    summary = dict(images=len(df), fully_ok=int(ok.sum()),
                   orientation_values_before={int(k): int(v) for k, v in df.exif_orientation_original.value_counts().items()},
                   orientation_values_after={int(k): int(v) for k, v in df.exif_orientation_corrected.value_counts().items()},
                   max_bytes_changed_per_file=int(df.bytes_changed.max()), max_box_error_px=float(df.max_box_error_px.max()),
                   annotations_json_boxes_unchanged=bool(boxes_same), total_boxes=int(df.n_boxes.sum()),
                   rose=int(df.n_rose.sum()), tallow=int(df.n_tallow.sum()), problem_images=df[~ok].image.tolist(),
                   comparison_sheets=pick)
    (OUT / "verification_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
