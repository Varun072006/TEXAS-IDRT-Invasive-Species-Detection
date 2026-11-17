"""Approach 1: train YOLO11 on tiles for one fold, then score on the validation FRAMES with SAHI.

Usage: python scripts/ab_train.py --name s05_y11s_f0 --val-fold 0 [--model yolo11s.pt --scale 0.5 --epochs 80]
       python scripts/ab_train.py --name final --final   # train on all non-test folds, no val-based selection
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import pandas as pd
from invasive.ab import evaluate_detections, gt_of, load_items, sahi_predict, write_fold

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "work" / "ab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--test-fold", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--extra-tallow", type=int, default=2)
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--nbs", type=int, default=12, help="nominal batch size; default 64 would accumulate 11 batches and starve the optimiser")
    ap.add_argument("--lr0", type=float, default=0.0005)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--no-sahi", action="store_true")
    args = ap.parse_args()

    tag = f"s{int(round(args.scale * 100)):03d}"
    pool_dir = AB / f"pool_{tag}"
    items = load_items(ROOT / "data" / "images")
    fold_of = json.loads((AB / "folds.json").read_text())
    pool = pd.read_csv(pool_dir / "pool.csv")
    run_dir = AB / "runs" / args.name
    meta = write_fold(pool_dir, pool, items, fold_of, args.val_fold, args.test_fold, run_dir, buffer=1,
                      neg_ratio=args.neg_ratio, extra_per_tallow=args.extra_tallow, final=args.final)
    print(f"[{args.name}] train tiles {meta['train_tiles']} boxes {meta['train_boxes']} | "
          f"val tiles {meta['val_tiles']} boxes {meta['val_boxes']} | buffer dropped {len(meta['buffer_dropped'])}")

    from ultralytics import YOLO
    model = YOLO(args.model)
    r = model.train(data=str(run_dir / "dataset.yaml"), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                    batch=args.batch, nbs=args.nbs, optimizer=args.optimizer, lr0=args.lr0, amp=not args.no_amp, warmup_epochs=2, workers=4, seed=42, project=str(run_dir), name="train", exist_ok=True,
                    flipud=0.5, fliplr=0.5, degrees=0, scale=0.3, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, mosaic=1.0,
                    close_mosaic=10, cos_lr=True, plots=False, verbose=False)
    best = run_dir / "train" / "weights" / ("last.pt" if args.final else "best.pt")
    out = dict(name=args.name, args=vars(args), weights=str(best), tile_val=dict(
        mAP50=float(r.box.map50), mAP50_95=float(r.box.map)) if r is not None and hasattr(r, "box") else None)
    if not args.no_sahi and not args.final:
        val_items = [i for i in items if i.name in set(meta["val_images"])]
        preds = sahi_predict(str(best), val_items, scale=args.scale, imgsz=args.imgsz)
        out["val_frames"] = evaluate_detections(gt_of(val_items), preds)
        v = out["val_frames"]
        print(f"[{args.name}] FULL-IMAGE val  mAP50 {v['mean']['AP50']:.3f} | rose {v['macartney_rose']['AP50']:.3f} "
              f"| tallow {v['chinese_tallow']['AP50']:.3f}")
    (run_dir / "result.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
