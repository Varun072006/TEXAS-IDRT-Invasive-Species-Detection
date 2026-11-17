#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
run() { PYTHONUNBUFFERED=1 $PY scripts/ab_train.py "$@"; }
run --name B1_y11s_s075_f0 --scale 0.75 --val-fold 0 --epochs 60 --patience 40 > work/ab/logs/B1_y11s_s075_f0.log 2>&1
run --name B2_y11m_s050_f0 --model yolo11m.pt --batch 4 --scale 0.5 --val-fold 0 --epochs 60 --patience 40 > work/ab/logs/B2_y11m_s050_f0.log 2>&1
echo done > work/ab/logs/queue2.done
