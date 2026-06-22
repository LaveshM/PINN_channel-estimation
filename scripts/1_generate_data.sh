#!/usr/bin/env bash
# Generate run_0000 to run_0010 (trucks, adaptive LS, SNR=0).
# run_0000 is the clean seed data; runs 1-10 add 5 trucks each.
#
# Usage: bash scripts/generate_data.sh

CSV="data/raw/15GHz_concatenated_data.csv"
OUT_DIR="data"
N_RUNS=10
STEP=5
USER_NOISE=3.0
SEED=42
ATTEN_MIN=2.0
ATTEN_MAX=10.0

set -euo pipefail
cd "$(dirname "$0")/.."

BASE="--csv $CSV --out-dir $OUT_DIR --n-runs $N_RUNS --step $STEP --user-noise $USER_NOISE --seed $SEED --atten-min $ATTEN_MIN --atten-max $ATTEN_MAX --skip-summary"

echo "=== Generating run_0000 to run_0010 ==="
python src/make_augmented_channels.py $BASE \
    --start-run 0 --end-run $N_RUNS \
    --snr-list 0 \
    --ls-modes adaptive

echo "=== Done. Output in $OUT_DIR/ ==="
