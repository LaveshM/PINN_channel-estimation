#!/usr/bin/env bash
# train_all.sh
# ------------
# Train the PINN for all SNR × split_type combinations.
# Requires data/run_0000/ to exist (run scripts/generate_data.sh first).
#
# ── CONFIG — edit these ───────────────────────────────────────────────────────
GPU_IDS=(0 1 2 3)
MAX_JOBS_PER_GPU=2
USER_NOISE=0.5
SNR_LIST=(-10 -5 0 5)
SPLIT_LIST=(random bloc)
EPOCHS=500
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

GPU_COUNT=${#GPU_IDS[@]}
MAX_CONCURRENT=$(( MAX_JOBS_PER_GPU * GPU_COUNT ))

RUN0="data/run_0000"
CH_FILE="${RUN0}/channels.npy"
POS_FILE="${RUN0}/locations_noisy.txt"
LOG_DIR="logs/noise_${USER_NOISE}"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  train_all.sh"
echo "  GPUs     : ${GPU_IDS[*]}  ($MAX_CONCURRENT concurrent jobs)"
echo "  SNR list : ${SNR_LIST[*]}"
echo "  Splits   : ${SPLIT_LIST[*]}"
echo "  Noise    : ${USER_NOISE} m   Epochs: ${EPOCHS}"
echo "  Data     : $RUN0"
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
for snr in "${SNR_LIST[@]}"; do
    snr_int=$snr
    snr_tag=$(python3 -c "s=$snr; print(f'+{int(s)}' if s>=0 else str(int(s)))")
    ls_file="${RUN0}/ls_snr${snr_tag}.npy"
    if [ ! -f "$ls_file" ]; then
        echo "ERROR: $ls_file not found. Run scripts/generate_data.sh with --snr-list $snr."
        exit 1
    fi
done

echo ""
echo "Launching training jobs ..."
job_index=0

for snr in "${SNR_LIST[@]}"; do
for split in "${SPLIT_LIST[@]}"; do

    snr_tag=$(python3 -c "s=$snr; print(f'+{int(s)}' if s>=0 else str(int(s)))")
    ls_file="${RUN0}/ls_snr${snr_tag}.npy"
    gpu=${GPU_IDS[$((job_index % GPU_COUNT))]}
    log="$LOG_DIR/snr${snr}_${split}.log"

    echo "  snr=$snr  split=$split  → GPU $gpu  (log: $log)"

    CUDA_VISIBLE_DEVICES=$gpu python3 train.py \
        --smomp_file          "$ls_file" \
        --accurate_file       "$CH_FILE" \
        --user_positions_file "$POS_FILE" \
        --split_type          "$split" \
        --user_noise          "$USER_NOISE" \
        --snr                 "$snr" \
        --epochs              "$EPOCHS" \
        --continue_training \
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
echo "  Results  in data/results_pinn.csv"
echo "============================================================"
