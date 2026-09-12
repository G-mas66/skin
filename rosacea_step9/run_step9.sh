#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/rosacea_step9
trap 'code=$?; echo "$code" > step9.exit' EXIT

python -u step9_train.py \
  --cache-root /root/autodl-tmp/rosacea_step2/cache_768 \
  --label-csv /root/autodl-tmp/rosacea_step2/CEA.csv \
  --baseline-root /root/autodl-tmp/rosacea_step8/results \
  --action audit
python -u step9_train.py \
  --cache-root /root/autodl-tmp/rosacea_step2/cache_768 \
  --label-csv /root/autodl-tmp/rosacea_step2/CEA.csv \
  --baseline-root /root/autodl-tmp/rosacea_step8/results \
  --action smoke
python -u step9_train.py \
  --cache-root /root/autodl-tmp/rosacea_step2/cache_768 \
  --label-csv /root/autodl-tmp/rosacea_step2/CEA.csv \
  --baseline-root /root/autodl-tmp/rosacea_step8/results \
  --action train
python -u step9_train.py \
  --cache-root /root/autodl-tmp/rosacea_step2/cache_768 \
  --label-csv /root/autodl-tmp/rosacea_step2/CEA.csv \
  --baseline-root /root/autodl-tmp/rosacea_step8/results \
  --action summarize
echo 0 > step9.exit
