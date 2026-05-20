#!/usr/bin/env bash
# scripts/train_aug4.sh
# ---------------------
# Experiment 4 — train on augmented data (run_0010, 20 trucks) for all three
# LS input modes: adaptive, fixed, refnoise.
#
# For each mode × split combination a separate model is trained and saved to:
#   models/aug/{mode}/snr0/{split}_3.0/simple_ls_val.pth
#
# ── CONFIG — edit these ───────────────────────────────────────────────────────
GPU_IDS=(0 1 2 3)
MAX_JOBS_PER_GPU=2
AUG_DIR="data/run_0010"
MODELS_DIR="models/aug"
SNR=0
USER_NOISE=3.0
EPOCHS=500
SPLIT_LIST=(random)
# LS mode name → corresponding LS file in AUG_DIR
declare -A LS_FILES
LS_FILES[adaptive]="ls_snr+0.npy"
LS_FILES[fixed]="ls_snr+0_fixed.npy"
LS_FILES[refnoise]="ls_snr+0_refnoise.npy"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

GPU_COUNT=${#GPU_IDS[@]}
MAX_CONCURRENT=$(( MAX_JOBS_PER_GPU * GPU_COUNT ))

CH_FILE="${AUG_DIR}/channels.npy"
POS_FILE="${AUG_DIR}/locations_noisy.txt"
LOG_DIR="logs/aug4"

mkdir -p "$LOG_DIR"
mkdir -p "$MODELS_DIR"

echo "============================================================"
echo "  train_aug4.sh — Experiment 4"
echo "  GPU IDs  : ${GPU_IDS[*]}  ($MAX_CONCURRENT concurrent jobs)"
echo "  Splits   : ${SPLIT_LIST[*]}"
echo "  LS modes : ${!LS_FILES[*]}"
echo "  SNR      : ${SNR} dB   Noise: ${USER_NOISE} m   Epochs: ${EPOCHS}"
echo "  Data     : $AUG_DIR"
echo "  Models   : $MODELS_DIR"
echo "============================================================"

# guard: data must exist
if [ ! -f "$CH_FILE" ]; then
    echo "ERROR: $CH_FILE not found. Run scripts/generate_data.sh first."
    exit 1
fi
if [ ! -f "$POS_FILE" ]; then
    echo "ERROR: $POS_FILE not found. Run scripts/generate_data.sh first."
    exit 1
fi
for mode in "${!LS_FILES[@]}"; do
    ls_file="${AUG_DIR}/${LS_FILES[$mode]}"
    if [ ! -f "$ls_file" ]; then
        echo "ERROR: $ls_file not found."
        exit 1
    fi
done

echo ""
echo "Launching training jobs ..."
job_index=0

for mode in adaptive fixed refnoise; do
for split in "${SPLIT_LIST[@]}"; do

    ls_file="${AUG_DIR}/${LS_FILES[$mode]}"
    model_subdir="${MODELS_DIR}/${mode}"
    gpu=${GPU_IDS[$((job_index % GPU_COUNT))]}
    log="${LOG_DIR}/${mode}_${split}.log"

    echo "  mode=${mode}  split=${split}  → GPU ${gpu}  (log: ${log})"

    CUDA_VISIBLE_DEVICES=$gpu python3 train.py \
        --smomp_file          "$ls_file" \
        --accurate_file       "$CH_FILE" \
        --user_positions_file "$POS_FILE" \
        --split_type          "$split" \
        --user_noise          "$USER_NOISE" \
        --snr                 "$SNR" \
        --epochs              "$EPOCHS" \
        --model_dir           "$model_subdir" \
        --results_csv         "${MODELS_DIR}/results_aug4.csv" \
        > "$log" 2>&1 &

    job_index=$(( job_index + 1 ))

    if (( job_index % MAX_CONCURRENT == 0 )); then
        echo "  → Waiting for batch of $MAX_CONCURRENT jobs ..."
        wait
    fi

done
done

wait
echo ""
echo "============================================================"
echo "  Done. Logs in $LOG_DIR/"
echo "  Results  in ${MODELS_DIR}/results_aug4.csv"
echo "  Models   in ${MODELS_DIR}/{adaptive,fixed,refnoise}/snr0/"
echo "============================================================"
