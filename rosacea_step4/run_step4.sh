#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/rosacea_step4
PYTHON=/root/miniconda3/bin/python

"$PYTHON" -u step4_explain.py \
  --action audit \
  --representative-method M+MB+MR
"$PYTHON" -u step4_explain.py \
  --action all \
  --representative-method M+MB+MR \
  --selection-approval USER_CONFIRMED_TEST_RANKING_DEVIATION
