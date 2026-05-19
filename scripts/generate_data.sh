#!/usr/bin/env bash
# generate_data.sh
# ----------------
# Full data generation pipeline — run this once before training.
#
# Pass 1 — builds ALL run directories (channels, positions, blocked_mask)
#           and generates LS at SNR=0 with all three noise modes for every run.
#           (blockage evaluation uses SNR=0 across all runs)
#
# Pass 2 — run_0000 only: adds LS at the remaining training SNRs {-10,-5,+5}.
#           channels.npy already exists so this is LS-only (fast).
#
# Output layout:
#   data/run_0000/  channels.npy  locations*.txt  blocked_mask.npy
#                   ls_snr-10.npy  ls_snr-5.npy  ls_snr+0.npy  ls_snr+5.npy
#                   ls_snr+0_fixed.npy  ls_snr+0_refnoise.npy
#   data/run_0001…0010/  channels.npy  locations*.txt  blocked_mask.npy
#                         ls_snr+0.npy  ls_snr+0_fixed.npy  ls_snr+0_refnoise.npy
#
# ── CONFIG — edit these ───────────────────────────────────────────────────────
CSV="Dataset/15GHz_concatenated_data.csv"
OUT_DIR="data"
N_RUNS=10       # blocker-count increments  (total dirs = N_RUNS+1)
STEP=2          # trucks added per run
USER_NOISE=3.0  # GPS noise std-dev (m)
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_ARGS="--csv $CSV --out-dir $OUT_DIR --n-runs $N_RUNS --step $STEP --user-noise $USER_NOISE --seed $SEED"

echo "============================================================"
echo "  generate_data.sh"
echo "  out_dir   : $OUT_DIR  ($((N_RUNS+1)) runs, step=$STEP trucks)"
echo "  user_noise: $USER_NOISE m   seed: $SEED"
echo "============================================================"

echo ""
echo "[Pass 1] All runs — channels + LS at SNR=0 (adaptive, fixed, refnoise)"
# shellcheck disable=SC2086
# python3 make_augmented_channels.py $BASE_ARGS \
#     --snr-list 0 \
#     --ls-modes adaptive fixed refnoise

echo ""
echo "[Pass 2] run_0000 only — LS at training SNRs -10, -5, +5 dB (adaptive)"
# shellcheck disable=SC2086
python3 make_augmented_channels.py $BASE_ARGS \
    --start-run 0 --end-run 0 \
    --snr-list -10 -5 5 \
    --ls-modes adaptive

echo ""
echo "============================================================"
echo "  Done."
echo "  run_0000 LS : ls_snr-10  ls_snr-5  ls_snr+0  ls_snr+5"
echo "              : ls_snr+0_fixed  ls_snr+0_refnoise"
echo "  run_0001-$(printf '%04d' $N_RUNS) LS : ls_snr+0  ls_snr+0_fixed  ls_snr+0_refnoise"
echo "  Next        : bash scripts/train_all.sh"
echo "============================================================"
