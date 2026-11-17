# Run log (auto-generated 2026-11-17 21:49)

Metrics: *tile* = Ultralytics validation on 1024 px tiles; *image* = full 6000x4000 frames after SAHI merge (the number that matters). Classes: rose = macartney_rose, tallow = chinese_tallow.

## A1_y11s_s050_f0  (1: YOLO11 tiles + SAHI)
- model yolo11s.pt, imgsz 1024, batch 6, lr0 0.0005, optimizer AdamW, val fold 0, epochs 40/50 (stopped early / interrupted)
- tile mAP50: best 0.282 (epoch 21), last 0.1704
- full-image `val_NMS_IOU0.5_s0.5`: mAP50 0.2093 | rose AP50 0.0516 (P 0.1142, R 0.1608) | tallow AP50 0.3671 (P 0.5, R 0.3929)

## A1_y11s_s050_f1  (1: YOLO11 tiles + SAHI)
- model yolo11s.pt, imgsz 1024, batch 6, lr0 0.01, optimizer auto, val fold 1, epochs 2/50 (stopped early / interrupted)
- tile mAP50: best 0.0002 (epoch 1), last 0.0001
- **INVALID/INCOMPLETE: started by the first queue with pre-fix settings (lr auto, nbs 64); killed after 2 epochs; not rerun**

## B1_y11s_s075_f0  (1: YOLO11 tiles + SAHI)
- model yolo11s.pt, imgsz 1024, batch 6, lr0 0.0005, optimizer AdamW, val fold 0, epochs 60/60 (completed)
- tile mAP50: best 0.3425 (epoch 20), last 0.1872
- full-image `train_script`: mAP50 0.2988 | rose AP50 0.0802 (P None, R None) | tallow AP50 0.5173 (P None, R None)

## B3_y11s_s100_f0  (1: YOLO11 tiles + SAHI)
- model yolo11s.pt, imgsz 1024, batch 6, lr0 0.0005, optimizer AdamW, val fold 0, epochs 21/50 (stopped early / interrupted)
- tile mAP50: best 0.3566 (epoch 13), last 0.2804
- full-image `val_NMS_IOU0.5_s1.0`: mAP50 0.2589 | rose AP50 0.0689 (P 0.1272, R 0.1725) | tallow AP50 0.4489 (P 0.4839, R 0.5357)

## B4_y11m_s075_f0  (1: YOLO11 tiles + SAHI)
- model yolo11m.pt, imgsz 1024, batch 4, lr0 0.0005, optimizer AdamW, val fold 0, epochs 43/50 (stopped early / interrupted)
- tile mAP50: best 0.3473 (epoch 31), last 0.2434
- full-image `val_NMS_IOU0.5_s0.75`: mAP50 0.2922 | rose AP50 0.0808 (P 0.1509, R 0.2196) | tallow AP50 0.5037 (P 0.65, R 0.4643)

## trial2_f0  (diagnostic)
- model yolo11s.pt, imgsz 1024, batch 6, lr0 0.0005, optimizer AdamW, val fold 0, epochs 8/8 (completed)
- tile mAP50: best 0.0005 (epoch 2), last 0.0002
- **INVALID: trained before the raw-frame fix (labels misaligned); stopped after ~5 epochs**

## trial_f0  (diagnostic)
- model yolo11s.pt, imgsz 1024, batch 6, lr0 0.01, optimizer auto, val fold 0, epochs 2/10 (stopped early / interrupted)
- tile mAP50: best 0.0004 (epoch 2), last 0.0004
- **INVALID: trained before the raw-frame fix (labels misaligned) and with diverging optimiser; stopped**
