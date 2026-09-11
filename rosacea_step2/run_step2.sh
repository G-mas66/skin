#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/rosacea_step2
PYTHON=/root/miniconda3/bin/python

"$PYTHON" -u step2_train.py --action audit
"$PYTHON" -u step2_train.py --action smoke
"$PYTHON" -u step2_train.py --action train
"$PYTHON" -u step2_train.py --action summarize
