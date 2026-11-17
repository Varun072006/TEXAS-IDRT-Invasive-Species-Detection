"""Rebuild the review files from work/ab/runs (safe to run any time, also while training is running).

Writes reports/:
  runs_summary.csv      one row per run: approach, settings, epochs, best tile mAP50, full-image metrics per class
  epochs_all_runs.csv   every epoch of every run (losses, precision, recall, mAP50, mAP50-95)
  RUN_LOG.md            readable summary tables (regenerated)
Hand-written history of every step lives in reports/STEPS_LOG.md (append-only).
"""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "work" / "ab" / "runs"
OUT = ROOT / "reports"

# runs that must not be read as model results
NOTES = {
    "trial_f0": "INVALID: trained before the raw-frame fix (labels misaligned) and with diverging optimiser; stopped",
    "trial2_f0": "INVALID: trained before the raw-frame fix (labels misaligned); stopped after ~5 epochs",
    "A1_y11s_s050_f1": "INVALID/INCOMPLETE: started by the first queue with pre-fix settings (lr auto, nbs 64); killed after 2 epochs; not rerun",
}


def approach_of(name: str) -> str:
    if name.startswith(("A", "B", "C", "F", "T")) and not name.startswith("trial"):
        return "1: YOLO11 tiles + SAHI"
    if name.startswith("S"):
        return "2: SAM3 prompted"
    if name.startswith("H"):
        return "3: Hybrid"
    return "diagnostic"


def main():
    OUT.mkdir(exist_ok=True)
    rows, epochs = [], []
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        res_csv = run / "train" / "results.csv"
        if not res_csv.exists():
            continue
        d = pd.read_csv(res_csv)
        d.columns = [c.strip() for c in d.columns]
        d.insert(0, "run", run.name)
        epochs.append(d)
        args = {}
        if (run / "train" / "args.yaml").exists():
            args = yaml.safe_load((run / "train" / "args.yaml").read_text())
        split = json.loads((run / "split.json").read_text()) if (run / "split.json").exists() else {}
        m50 = d["metrics/mAP50(B)"]
        row = dict(run=run.name, approach=approach_of(run.name), model=args.get("model"), imgsz=args.get("imgsz"),
                   batch=args.get("batch"), lr0=args.get("lr0"), optimizer=args.get("optimizer"), nbs=args.get("nbs"),
                   epochs_planned=args.get("epochs"), epochs_done=len(d),
                   val_fold=split.get("val_fold"), train_tiles=split.get("train_tiles"),
                   train_rose=(split.get("train_boxes") or {}).get("rose"),
                   train_tallow=(split.get("train_boxes") or {}).get("tallow"),
                   tile_best_mAP50=round(float(m50.max()), 4), tile_best_epoch=int(m50.idxmax()) + 1,
                   tile_last_mAP50=round(float(m50.iloc[-1]), 4),
                   status=("running" if time.time() - res_csv.stat().st_mtime < 300 else ("completed" if len(d) >= (args.get("epochs") or 10**9) else "stopped early / interrupted")),
                   note=NOTES.get(run.name, ""))
        # full-image (SAHI) evaluations
        for ev in sorted(run.glob("eval_*.json")):
            r = json.loads(ev.read_text())
            tag = ev.stem.replace("eval_", "")
            row[f"img[{tag}]_mAP50"] = round(r["mean"]["AP50"], 4)
            row[f"img[{tag}]_mAP50_95"] = round(r["mean"]["AP50_95"], 4)
            for c, short in (("macartney_rose", "rose"), ("chinese_tallow", "tallow")):
                x = r[c]
                row[f"img[{tag}]_{short}_AP50"] = round(x["AP50"], 4)
                row[f"img[{tag}]_{short}_P"] = round(x["P"], 4)
                row[f"img[{tag}]_{short}_R"] = round(x["R"], 4)
        if (run / "result.json").exists():
            rj = json.loads((run / "result.json").read_text())
            v = rj.get("val_frames")
            if v:
                row["img[train_script]_mAP50"] = round(v["mean"]["AP50"], 4)
                row["img[train_script]_rose_AP50"] = round(v["macartney_rose"]["AP50"], 4)
                row["img[train_script]_tallow_AP50"] = round(v["chinese_tallow"]["AP50"], 4)
        rows.append(row)

    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "runs_summary.csv", index=False)
    pd.concat(epochs, ignore_index=True).to_csv(OUT / "epochs_all_runs.csv", index=False)

    md = [f"# Run log (auto-generated {datetime.now():%Y-%m-%d %H:%M})", "",
          "Metrics: *tile* = Ultralytics validation on 1024 px tiles; *image* = full 6000x4000 frames after SAHI merge "
          "(the number that matters). Classes: rose = macartney_rose, tallow = chinese_tallow.", ""]
    img_cols = [c for c in summ.columns if c.startswith("img[") and c.endswith("_mAP50") and "95" not in c]
    for _, r in summ.iterrows():
        r = r.dropna()
        md.append(f"## {r['run']}  ({r['approach']})")
        md.append(f"- model {r['model']}, imgsz {r['imgsz']}, batch {r['batch']}, lr0 {r['lr0']}, optimizer {r['optimizer']}, "
                  f"val fold {r['val_fold']}, epochs {r['epochs_done']}/{r['epochs_planned']} ({r['status']})")
        md.append(f"- tile mAP50: best {r['tile_best_mAP50']} (epoch {r['tile_best_epoch']}), last {r['tile_last_mAP50']}")
        for c in [c for c in img_cols if c in r.index]:
            tag = c[4:c.index("]_")]
            md.append(f"- full-image `{tag}`: mAP50 {r[c]} | rose AP50 {r.get(f'img[{tag}]_rose_AP50')} "
                      f"(P {r.get(f'img[{tag}]_rose_P')}, R {r.get(f'img[{tag}]_rose_R')}) | tallow AP50 "
                      f"{r.get(f'img[{tag}]_tallow_AP50')} (P {r.get(f'img[{tag}]_tallow_P')}, R {r.get(f'img[{tag}]_tallow_R')})")
        if r["note"]:
            md.append(f"- **{r.get('note', '')}**")
        md.append("")
    (OUT / "RUN_LOG.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(summ)} runs, {sum(len(e) for e in epochs)} epoch rows -> {OUT}")


if __name__ == "__main__":
    main()
