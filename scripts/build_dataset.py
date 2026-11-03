"""Build a YOLO tile dataset for one validation fold from a Labelbox export.

Usage: python scripts/build_dataset.py --export export.ndjson --val-fold 0 [--skipped negative|exclude]
Outputs <workdir>/datasets/fold<k>/{images,labels}/{train,val} and dataset.yaml.
"""
import argparse
import json

import _common  # noqa: F401
import pandas as pd

from invasive.config import load_config, resolve_path
from invasive.dataset import (drop_buffer_frames, file_hash, label_qa, spatial_block_folds, write_yolo_dataset)
from invasive.labelbox import parse_export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--skipped", choices=["negative", "exclude"], default="exclude",
                    help="what to do with images skipped in Labelbox (negative = no target present)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = resolve_path(cfg, "workdir")
    manifest = pd.read_csv(work / "manifest.csv")
    labels, images, warnings = parse_export(args.export, cfg)
    for w in warnings[:20]:
        print("warning:", w)

    qa = label_qa(labels, manifest)
    bad = qa.flag_no_image | qa.flag_unreadable_image | qa.flag_duplicate
    print(f"dropping {int(bad.sum())} boxes (no image / unreadable image / duplicate); "
          f"keeping {int((~bad).sum())}; other flags kept: tiny={int(qa.flag_tiny.sum())}, huge={int(qa.flag_huge.sum())}, "
          f"outside={int(qa.flag_outside.sum())}")
    labels = labels[~bad.to_numpy()]

    sp = cfg["dataset"]["split"]
    folds = spatial_block_folds(manifest, sp["folds"], sp["block_size"], sp["seed"])
    fold_of_image = {n: int(f) for n, f in zip(manifest.image_name, folds) if f >= 0}
    excluded_idx = drop_buffer_frames(manifest, folds, args.val_fold, sp["buffer_frames"])
    excluded = set(manifest.loc[list(excluded_idx), "image_name"])

    labelled = set(images.loc[images.n_objects > 0, "image_name"])
    skipped = set(images.loc[images.skipped, "image_name"])
    in_export = set(images.image_name)
    # Only images that Labelbox has finished are used: labelled images, plus skipped ones as negatives if asked.
    use = set(labelled) | (skipped if args.skipped == "negative" else set())
    excluded |= set(manifest.image_name) - use
    print(f"images used: {len(use & set(manifest.image_name))} (labelled {len(labelled)}, skipped {len(skipped)} -> {args.skipped}); "
          f"not in export: {len(set(manifest.image_name) - in_export)}; "
          f"excluded as buffer around validation blocks: {len(excluded_idx)}")

    out = work / "datasets" / f"fold{args.val_fold}"
    yaml_path = write_yolo_dataset(manifest, labels, fold_of_image, args.val_fold, excluded, cfg, out, cfg["classes"],
                                   seed=sp["seed"])
    meta = dict(export=str(args.export), export_hash=file_hash(args.export), val_fold=args.val_fold,
                skipped=args.skipped, n_boxes=int(len(labels)), config=cfg["_config_path"])
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.DataFrame({"image_name": list(fold_of_image), "fold": list(fold_of_image.values())}).to_csv(out / "folds.csv", index=False)
    n_tr = len(list((out / "images" / "train").glob("*.jpg")))
    n_va = len(list((out / "images" / "val").glob("*.jpg")))
    print(f"dataset written: {yaml_path}  (train tiles {n_tr}, val tiles {n_va})")


if __name__ == "__main__":
    main()
