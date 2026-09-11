#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/rosacea_step3
PYTHON=/root/miniconda3/bin/python

"$PYTHON" -u step3_train.py \
  --cache-root /root/autodl-tmp/rosacea_step2/cache_768 \
  --label-csv /root/autodl-tmp/rosacea_step2/CEA.csv \
  --baseline-root /root/autodl-tmp/rosacea_step2/results \
  --action audit
"$PYTHON" -u step3_train.py \
  --cache-root /root/autodl-tmp/rosacea_step2/cache_768 \
  --label-csv /root/autodl-tmp/rosacea_step2/CEA.csv \
  --baseline-root /root/autodl-tmp/rosacea_step2/results \
  --action smoke
"$PYTHON" -u step3_train.py \
  --cache-root /root/autodl-tmp/rosacea_step2/cache_768 \
  --label-csv /root/autodl-tmp/rosacea_step2/CEA.csv \
  --baseline-root /root/autodl-tmp/rosacea_step2/results \
  --action train
"$PYTHON" -u step3_train.py \
  --cache-root /root/autodl-tmp/rosacea_step2/cache_768 \
  --label-csv /root/autodl-tmp/rosacea_step2/CEA.csv \
  --baseline-root /root/autodl-tmp/rosacea_step2/results \
  --action summarize
