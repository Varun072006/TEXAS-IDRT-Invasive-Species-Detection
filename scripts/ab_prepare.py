"""Step 0: verify class order, make stratified spatial folds, write the tile pool.

Usage: python scripts/ab_prepare.py [--scale 0.5]
Outputs work/ab/folds.json and work/ab/pool_s<scale>/{images,labels,pool.csv}
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from invasive.ab import CLASS_NAMES, ROSE, TALLOW, build_pool, load_items, make_folds

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "images"
AB = ROOT / "work" / "ab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=1024)
    args = ap.parse_args()
    items = load_items(IMAGES)  # raises if classes.txt order != CLASS_NAMES
    print("classes:", dict(enumerate(CLASS_NAMES)), "| images:", len(items))
    AB.mkdir(parents=True, exist_ok=True)
    fold_of = make_folds(items, 5, 4)
    (AB / "folds.json").write_text(json.dumps(fold_of, indent=1))
    print("fold  images  rose  tallow")
    for f in range(5):
        its = [i for i in items if fold_of[i.name] == f]
        print(f"{f:>4} {len(its):>7} {sum(int((i.cls == ROSE).sum()) for i in its):>5} "
              f"{sum(int((i.cls == TALLOW).sum()) for i in its):>7}")
    tag = f"s{int(round(args.scale * 100)):03d}"
    pool = build_pool(items, AB / f"pool_{tag}", tile_px=args.tile, scale=args.scale)
    g = pool[pool.kind == "g"]
    print(f"pool {tag}: {len(g)} grid tiles ({int(((g.n_rose + g.n_tallow) > 0).sum())} positive), "
          f"{int((pool.kind == 'x').sum())} extra tallow tiles")


if __name__ == "__main__":
    main()
