"""Turn the Labelbox NDJSON into training-ready annotations next to the downloaded images.

Outputs (in data/images/):
  labels/<image>.txt   YOLO format: class cx cy w h (normalised), in the STORED pixel grid (no EXIF rotation)
  classes.txt          class id -> name
  annotations.json     COCO format + image metadata, EXIF camera info, GPS geodata, per-box details
  geotags.csv          one row per image: lat/lon/alt, timestamp, box counts
  preview/<image>.jpg  downscaled image with the boxes drawn (to eyeball the annotations)
Labelbox boxes refer to the stored pixel grid (width x height in the export, exif_rotation is only informational), NOT
the EXIF-upright image. Previews must therefore draw on the stored pixels: do NOT call exif_transpose here (that rotated
the 180-degree EXIF=3 frames and left the boxes behind). See data/images/corrected/VERIFICATION_REPORT.md.
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent / "data"
NDJSON = BASE / "Export  project - Invasive Species - 11_10_2026.ndjson"
IMG = BASE / "images"
(IMG / "labels").mkdir(exist_ok=True)
(IMG / "preview").mkdir(exist_ok=True)

classes = {"macartney_rose": 0, "chinese_tallow": 1}
display = {"macartney_rose": "Macartney Rose", "chinese_tallow": "Chinese Tallow"}
colors = {0: "red", 1: "yellow"}


def dms(v, ref):
    d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    return -d if ref in ("S", "W") else d


def exif_info(path):
    """Camera/EXIF metadata + GPS. Returns (camera dict, geo dict or None, raw size)."""
    im = Image.open(path)
    ex = im.getexif()
    sub = ex.get_ifd(0x8769)
    gps = ex.get_ifd(0x8825)
    cam = {
        "make": ex.get(271), "model": ex.get(272), "orientation": ex.get(274),
        "lens_make": sub.get(0xA433), "lens_model": sub.get(0xA434),
        "focal_length_mm": float(sub[0x920A]) if 0x920A in sub else None,
        "focal_length_35mm": sub.get(0xA405),
        "exposure_time_s": float(sub[0x829A]) if 0x829A in sub else None,
        "iso": sub.get(0x8827),
        "datetime_original": sub.get(0x9003), "utc_offset": sub.get(0x9011),
    }
    geo = None
    if 2 in gps and 4 in gps:
        geo = {
            "latitude": dms(gps[2], gps.get(1)), "longitude": dms(gps[4], gps.get(3)),
            "altitude_m": float(gps[6]) if 6 in gps else None,
            "gps_time_utc": ":".join(f"{int(float(x)):02d}" for x in gps[7]) if 7 in gps else None,
            "gps_date": gps.get(29), "datum": gps.get(18), "source": "EXIF GPS",
        }
    return cam, geo, im.size


coco = {
    "info": {
        "description": "Invasive Species Detection - Labelbox export (project 'Invasive Species'), bounding boxes",
        "version": "1.0", "date_created": datetime.now().isoformat(timespec="seconds"),
        "source_export": NDJSON.name,
        "coordinate_frame": "pixels of the STORED image grid as shown by Labelbox; the JPGs carry EXIF orientation 3 which is NOT applied to these coordinates",
        "bbox_format": "[x_left, y_top, width, height] in pixels",
        "geo_note": "Geodata is the camera position from EXIF GPS (WGS84). The Wingtra geotags CSVs in data/raw do not contain these AB_Oppenheimer images, so no omega/phi/kappa is available for them.",
    },
    "licenses": [],
    "images": [], "annotations": [],
    "categories": [{"id": i, "name": n, "display_name": display[n], "supercategory": "invasive_plant"}
                  for n, i in classes.items()],
}
aid = 0
missing = []
rows = []
for iid, line in enumerate(open(NDJSON, encoding="utf-8")):
    if not line.strip():
        continue
    d = json.loads(line)
    dr, ma = d["data_row"], d["media_attributes"]
    name = dr["external_id"]
    W, H = ma["width"], ma["height"]
    src = IMG / name
    cam = geo = None
    raw_size = None
    if src.exists():
        cam, geo, raw_size = exif_info(src)
    boxes = []
    for pid, p in d["projects"].items():
        for lb in p["labels"]:
            for o in lb["annotations"]["objects"]:
                boxes.append((o, lb["id"], pid, p["name"]))
    counts = {n: sum(1 for b in boxes if b[0]["value"] == n) for n in classes}
    img = {
        "id": iid, "file_name": name, "width": W, "height": H,
        "labelbox_data_row_id": dr["id"], "asset_type": ma["asset_type"], "mime_type": ma["mime_type"],
        "exif_rotation": ma.get("exif_rotation"),
        "raw_file_size_px": list(raw_size) if raw_size else None,
        "file_size_bytes": src.stat().st_size if src.exists() else None,
        "flight": name.rsplit("_", 1)[0], "date_captured": cam and cam["datetime_original"],
        "camera": cam, "geo": geo,
        "num_annotations": len(boxes), "annotations_per_class": counts,
        "preview": f"preview/{Path(name).stem}.jpg",
    }
    coco["images"].append(img)
    rows.append(img)
    with open(IMG / "labels" / (Path(name).stem + ".txt"), "w") as f:
        for o, *_ in boxes:
            b = o["bounding_box"]
            l, t, w, h = b["left"], b["top"], b["width"], b["height"]
            f.write(f"{classes[o['value']]} {(l + w / 2) / W:.6f} {(t + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
    for o, lid, pid, pname in boxes:
        b = o["bounding_box"]
        l, t, w, h = b["left"], b["top"], b["width"], b["height"]
        aid += 1
        coco["annotations"].append({
            "id": aid, "image_id": iid, "category_id": classes[o["value"]],
            "bbox": [l, t, w, h], "area": w * h, "iscrowd": 0,
            "bbox_normalized_cxcywh": [(l + w / 2) / W, (t + h / 2) / H, w / W, h / H],
            "segmentation": [[l, t, l + w, t, l + w, t + h, l, t + h]],
            "labelbox_feature_id": o["feature_id"], "labelbox_label_id": lid,
            "annotation_kind": o["annotation_kind"], "project": pname,
            "classifications": o.get("classifications", []),
        })
    if not src.exists():
        missing.append(name)
        continue
    im = Image.open(src).convert("RGB")  # stored grid = Labelbox grid; no EXIF rotation
    draw = ImageDraw.Draw(im)
    for o, *_ in boxes:
        b = o["bounding_box"]
        c = classes[o["value"]]
        draw.rectangle([b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]],
                       outline=colors[c], width=6)
    im.thumbnail((2000, 2000))
    im.save(IMG / "preview" / (Path(name).stem + ".jpg"), quality=85)

with open(IMG / "geotags.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "latitude", "longitude", "altitude_m", "gps_date", "gps_time_utc", "camera",
                "n_boxes", "n_macartney_rose", "n_chinese_tallow"])
    for r in rows:
        g = r["geo"] or {}
        w.writerow([r["file_name"], g.get("latitude"), g.get("longitude"), g.get("altitude_m"),
                    g.get("gps_date"), g.get("gps_time_utc"), r["camera"] and r["camera"]["model"],
                    r["num_annotations"], *r["annotations_per_class"].values()])

(IMG / "classes.txt").write_text("\n".join(classes) + "\n")
(IMG / "annotations.json").write_text(json.dumps(coco, indent=1))
print(f"{len(coco['images'])} images, {aid} boxes, {sum(1 for r in rows if r['geo'])} with GPS. "
      f"Missing image files: {len(missing)} {missing}")
