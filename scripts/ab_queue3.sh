#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
run() { PYTHONUNBUFFERED=1 $PY scripts/ab_train.py "$@"; }
run --name B3_y11s_s100_f0 --scale 1.0 --val-fold 0 --epochs 50 --patience 30 > work/ab/logs/B3_y11s_s100_f0.log 2>&1
run --name B4_y11m_s075_f0 --model yolo11m.pt --batch 4 --scale 0.75 --val-fold 0 --epochs 50 --patience 30 > work/ab/logs/B4_y11m_s075_f0.log 2>&1
echo done > work/ab/logs/queue3.done
