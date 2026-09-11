#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/rosacea_step8
trap 'code=$?; echo "$code" > step8.exit' EXIT

python step8_train.py --action train --output-root results
python step8_train.py --action summarize --output-root results
echo 0 > step8.exit
