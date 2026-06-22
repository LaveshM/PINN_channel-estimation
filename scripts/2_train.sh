#!/usr/bin/env bash
# Train the seed PINN on run_0000 (clean data, SNR=0, random split).
#
# Usage: bash scripts/train.sh

RUN_DIR="data/run_0000"
MODELS_DIR="models"
EPOCHS=500
USER_NOISE=3.0
SNR=0

set -euo pipefail
cd "$(dirname "$0")/.."

python src/train.py \
    --smomp_file          "$RUN_DIR/ls_snr+0.npy" \
    --accurate_file       "$RUN_DIR/channels.npy" \
    --user_positions_file "$RUN_DIR/locations_noisy.txt" \
    --rss_image_path      "data/raw/50_15GHz.jpg" \
    --split_type          random \
    --user_noise          $USER_NOISE \
    --snr                 $SNR \
    --epochs              $EPOCHS \
    --model_dir           $MODELS_DIR \
    --results_csv         "$MODELS_DIR/results.csv" \
    --continue_training
