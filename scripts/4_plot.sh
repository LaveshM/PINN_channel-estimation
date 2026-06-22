#!/usr/bin/env bash
# Generate plots from saved evaluation CSVs.
# Reads data/run_XXXX/seed_predictions_snr+0.csv → plots/
#
# Usage: bash scripts/4_plot.sh

DATA_DIR="data"
OUT_DIR="plots"

set -euo pipefail
cd "$(dirname "$0")/.."

python src/make_plots.py \
    --data-dir $DATA_DIR \
    --out-dir  $OUT_DIR
