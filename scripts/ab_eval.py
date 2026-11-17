"""Full-image evaluation of a checkpoint with SAHI. Usage:
python scripts/ab_eval.py --weights W.pt --run-dir work/ab/runs/<run> [--scale 0.5] [--post NMS|GREEDYNMM|NMM] [--split val|test]
Writes <run-dir>/eval_<split>_<post>.json
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from invasive.ab import evaluate_detections, gt_of, load_items, sahi_predict

ROOT = Path(__file__).resolve().parents[1]


def show(tag, r):
    print(f"{tag}: mAP50 {r['mean']['AP50']:.3f} mAP50-95 {r['mean']['AP50_95']:.3f} P {r['mean']['P']:.3f} R {r['mean']['R']:.3f}")
    for c in ("macartney_rose", "chinese_tallow"):
        x = r[c]
        print(f"   {c:15s} n={x['n_gt']:4d} AP50 {x['AP50']:.3f} AP50-95 {x['AP50_95']:.3f} P {x['P']:.3f} R {x['R']:.3f} @conf {x['best_f1_conf']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--post", default="NMS", choices=["NMS", "GREEDYNMM", "NMM"])
    ap.add_argument("--metric", default="IOU")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()
    items = load_items(ROOT / "data" / "images")
    run = Path(args.run_dir)
    if args.split == "val":
        names = set(json.loads((run / "split.json").read_text())["val_images"])
    else:
        fold_of = json.loads((ROOT / "work/ab/folds.json").read_text())
        names = {n for n, f in fold_of.items() if f == 4}
    sel = [i for i in items if i.name in names]
    preds = sahi_predict(args.weights, sel, scale=args.scale, imgsz=args.imgsz, overlap=args.overlap,
                         postprocess=args.post, match_metric=args.metric, match_thr=args.thr)
    r = evaluate_detections(gt_of(sel), preds)
    tag = f"{args.split}_{args.post}_{args.metric}{args.thr}_s{args.scale}"
    (run / f"eval_{tag}.json").write_text(json.dumps(r, indent=1))
    show(f"[{run.name}] {tag} ({len(sel)} images)", r)


if __name__ == "__main__":
    main()
