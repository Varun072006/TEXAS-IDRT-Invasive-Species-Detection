"""Feasibility probe: can a frozen CLIP embedding separate rose / tallow / background crops? (fold-wise CV)"""
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from invasive.ab import load_items, read_frame
from invasive.ab import _iou

ROOT = Path(__file__).resolve().parents[1]
items = load_items(ROOT / "data" / "images")
fold_of = json.loads((ROOT / "work/ab/folds.json").read_text())
import open_clip
model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
model = model.cuda().half().eval()
rng = np.random.default_rng(0)
sizes = np.concatenate([it.boxes[:, 2:] - it.boxes[:, :2] for it in items])
X, y, f = [], [], []

def emb(crops):
    out = []
    for i in range(0, len(crops), 64):
        t = torch.stack([pre(c) for c in crops[i:i + 64]]).cuda().half()
        with torch.no_grad():
            e = model.encode_image(t); e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)

for it in items:
    img = Image.fromarray(read_frame(it.path))
    crops, labs = [], []
    for b, c in zip(it.boxes, it.cls):
        crops.append(img.crop(tuple(int(v) for v in b))); labs.append(int(c))
    k = 0; tries = 0
    while k < 6 and tries < 200:
        tries += 1
        w, h = sizes[rng.integers(len(sizes))]
        x = rng.uniform(0, it.w - w); yy = rng.uniform(0, it.h - h)
        b = np.array([[x, yy, x + w, yy + h]])
        if len(it.boxes) and _iou(b, it.boxes).max() > 0.05:
            continue
        crops.append(img.crop(tuple(int(v) for v in b[0]))); labs.append(2); k += 1
    X.append(emb(crops)); y += labs; f += [fold_of[it.name]] * len(labs)
X = np.concatenate(X); y = np.array(y); f = np.array(f)
print("crops", len(y), {c: int((y == c).sum()) for c in (0, 1, 2)})
pred = np.zeros_like(y)
for k in range(5):
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(X[f != k], y[f != k])
    pred[f == k] = clf.predict(X[f == k])
print(classification_report(y, pred, target_names=["rose", "tallow", "background"], digits=3))
