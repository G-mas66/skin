#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/rosacea_step10
trap 'code=$?; echo "$code" > step10.exit' EXIT

python -u step10_explain.py \
  --cache-root /root/autodl-tmp/rosacea_step2/cache_768 \
  --label-csv /root/autodl-tmp/rosacea_step2/CEA.csv \
  --step9-root /root/autodl-tmp/rosacea_step9/results \
  --action all \
  --selection-approval USER_CONFIRMED_STEP9_TEST_RANKING_SELECTION
echo 0 > step10.exit
