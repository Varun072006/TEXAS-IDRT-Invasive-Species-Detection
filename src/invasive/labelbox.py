"""Parse Labelbox NDJSON exports (bounding boxes) into label and image tables.

Handles the current export format ("data_row" / "projects" / "labels") and the legacy format
("External ID" / "Label" / "Skipped"). Returned boxes are in the pixel space of the image as
shown in Labelbox (upright), as x1, y1, x2, y2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import build_alias_map, canonical_class
from .io import canonical_name

LABEL_COLUMNS = ["label_id", "image_name", "class", "raw_class", "x1", "y1", "x2", "y2",
                 "annotator", "workflow_status"]
IMAGE_COLUMNS = ["image_name", "data_row_id", "width", "height", "workflow_status", "n_objects",
                 "skipped"]


def _bbox_to_xyxy(bb: dict) -> tuple[float, float, float, float]:
    left, top = float(bb["left"]), float(bb["top"])
    return left, top, left + float(bb["width"]), top + float(bb["height"])


def parse_export(path: str | Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (labels, images, warnings). Objects of unknown classes are reported, not dropped silently."""
    alias_map = build_alias_map(cfg)
    labels: list[dict] = []
    images: list[dict] = []
    warnings: list[str] = []

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "data_row" in row:
                _parse_current(row, alias_map, labels, images, warnings)
            elif "External ID" in row or "Label" in row:
                _parse_legacy(row, alias_map, labels, images, warnings)
            else:
                warnings.append(f"line {line_no}: unrecognised export row")

    lab = pd.DataFrame(labels, columns=LABEL_COLUMNS)
    img = pd.DataFrame(images, columns=IMAGE_COLUMNS)
    return lab, img, warnings


def _add_object(obj: dict, name: str, alias_map, labels, warnings, annotator, status, label_id):
    raw = obj.get("name") or obj.get("title") or ""
    cls = canonical_class(raw, alias_map)
    bb = obj.get("bounding_box") or obj.get("bbox")
    if bb is None:
        warnings.append(f"{name}: object '{raw}' has no bounding box (kind={obj.get('annotation_kind')})")
        return 0
    if cls is None:
        warnings.append(f"{name}: unknown class '{raw}'")
        return 0
    x1, y1, x2, y2 = _bbox_to_xyxy(bb)
    labels.append(dict(label_id=obj.get("feature_id") or obj.get("featureId") or label_id,
                       image_name=name, **{"class": cls}, raw_class=raw,
                       x1=x1, y1=y1, x2=x2, y2=y2, annotator=annotator, workflow_status=status))
    return 1


def _parse_current(row, alias_map, labels, images, warnings):
    dr = row["data_row"]
    name = canonical_name(dr.get("external_id") or dr.get("id"))
    media = row.get("media_attributes", {}) or {}
    for project in (row.get("projects") or {}).values():
        status = (project.get("project_details") or {}).get("workflow_status", "")
        n = 0
        for li, lab in enumerate(project.get("labels") or []):
            annotator = (lab.get("label_details") or {}).get("created_by", "")
            for oi, obj in enumerate((lab.get("annotations") or {}).get("objects", [])):
                n += _add_object(obj, name, alias_map, labels, warnings, annotator, status, f"{name}#{li}.{oi}")
        images.append(dict(image_name=name, data_row_id=dr.get("id", ""), width=media.get("width"),
                           height=media.get("height"), workflow_status=status, n_objects=n,
                           skipped=str(status).upper() == "SKIPPED"))


def _parse_legacy(row, alias_map, labels, images, warnings):
    name = canonical_name(row.get("External ID") or row.get("ID"))
    skipped = bool(row.get("Skipped")) or row.get("Label") in (None, "Skip")
    n = 0
    if not skipped and isinstance(row.get("Label"), dict):
        for oi, obj in enumerate(row["Label"].get("objects", [])):
            n += _add_object(obj, name, alias_map, labels, warnings, row.get("Created By", ""), "DONE",
                             f"{name}#{oi}")
    images.append(dict(image_name=name, data_row_id=row.get("DataRow ID", ""), width=None, height=None,
                       workflow_status="SKIPPED" if skipped else "DONE", n_objects=n, skipped=skipped))
