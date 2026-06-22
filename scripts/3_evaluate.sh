#!/usr/bin/env bash
# Run the trained model on all run_0000 to run_0010 and save per-user results
# to data/run_XXXX/seed_predictions_{ls_tag}.csv
#
# Usage: bash scripts/3_evaluate.sh

MODEL="models/snr0/random_3.0/simple_ls_val.pth"
DATA_DIR="data"

set -euo pipefail
cd "$(dirname "$0")/.."

python src/evaluate.py \
    --data-dir   $DATA_DIR \
    --model      $MODEL
