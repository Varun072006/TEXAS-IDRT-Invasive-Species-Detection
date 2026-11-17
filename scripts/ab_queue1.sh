#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
for f in 0 1; do
  PYTHONUNBUFFERED=1 $PY scripts/ab_train.py --name A1_y11s_s050_f$f --val-fold $f --epochs 50 --patience 20 > work/ab/logs/A1_y11s_s050_f$f.log 2>&1
done
echo done > work/ab/logs/queue1.done
