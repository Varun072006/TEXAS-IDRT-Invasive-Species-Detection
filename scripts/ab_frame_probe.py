"""Which coordinate frame are the labels in? Re-run the CLIP crop probe under 4 frame hypotheses."""
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from invasive.ab import _iou, load_items, read_upright

ROOT = Path(__file__).resolve().parents[1]
items = load_items(ROOT / "data" / "images")
fold_of = json.loads((ROOT / "work/ab/folds.json").read_text())
import open_clip
model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
model = model.cuda().half().eval()
rng = np.random.default_rng(0)
sizes = np.concatenate([it.boxes[:, 2:] - it.boxes[:, :2] for it in items])

def tf(b, w, h, mode):
    b = b.copy()
    if mode in ("rot180", "flipx"): b[:, [0, 2]] = w - b[:, [2, 0]]
    if mode in ("rot180", "flipy"): b[:, [1, 3]] = h - b[:, [3, 1]]
    return b

def emb(crops):
    if not crops: return np.zeros((0, 512), np.float32)
    out = []
    for i in range(0, len(crops), 64):
        t = torch.stack([pre(c) for c in crops[i:i + 64]]).cuda().half()
        with torch.no_grad():
            e = model.encode_image(t); e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)

modes = ["identity", "rot180", "flipx", "flipy"]
D = {m: [[], [], []] for m in modes}
for it in items:
    img = Image.fromarray(read_upright(it.path))
    allb = np.concatenate([tf(it.boxes, it.w, it.h, m) for m in modes]) if len(it.boxes) else np.zeros((0, 4))
    bgs, k, tries = [], 0, 0
    while k < 6 and tries < 300:
        tries += 1
        w, h = sizes[rng.integers(len(sizes))]
        x = rng.uniform(0, it.w - w); y = rng.uniform(0, it.h - h)
        b = np.array([[x, y, x + w, y + h]])
        if len(allb) and _iou(b, allb).max() > 0.05: continue
        bgs.append(img.crop(tuple(int(v) for v in b[0]))); k += 1
    ebg = emb(bgs) if bgs else np.zeros((0, 512))
    for m in modes:
        bx = tf(it.boxes, it.w, it.h, m)
        e = emb([img.crop(tuple(int(v) for v in b)) for b in bx])
        D[m][0].append(np.concatenate([e, ebg])); D[m][1] += list(it.cls) + [2] * len(ebg)
        D[m][2] += [fold_of[it.name]] * (len(e) + len(ebg))
for m in modes:
    X = np.concatenate(D[m][0]); y = np.array(D[m][1]); f = np.array(D[m][2]); pred = np.zeros_like(y)
    for k in range(5):
        pred[f == k] = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(X[f != k], y[f != k]).predict(X[f == k])
    rec = [float((pred[y == c] == c).mean()) for c in (0, 1, 2)]
    print(f"{m:9s} acc {float((pred == y).mean()):.3f} | recall rose {rec[0]:.2f} tallow {rec[1]:.2f} bg {rec[2]:.2f}")
