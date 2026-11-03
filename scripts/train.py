"""Train a detector on a dataset built by build_dataset.py (Ultralytics backend; note the AGPL licence).

Usage: python scripts/train.py --dataset work/datasets/fold0/dataset.yaml [--name fold0]
"""
import argparse

import _common  # noqa: F401

from invasive.config import load_config, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="run")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg["train"]

    from ultralytics import YOLO

    model = YOLO(t["model"])
    model.train(data=args.dataset, epochs=args.epochs or t["epochs"], imgsz=t["imgsz"], batch=t["batch"],
                seed=t["seed"], project=str(resolve_path(cfg, "workdir") / "runs"), name=args.name,
                # nadir imagery has no up direction; hue changes stay modest because colour is a species cue
                flipud=0.5, fliplr=0.5, degrees=0, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, mosaic=1.0, copy_paste=0.0)


if __name__ == "__main__":
    main()
