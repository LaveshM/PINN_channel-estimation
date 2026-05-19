#!/usr/bin/env bash
# train_aug.sh
# ------------
# Train the three augmentation scenarios (A, B, C) in parallel on separate GPUs.
#
#   A (GPU 0) — seed data only  (data/run_0000)
#   B (GPU 1) — aug  data only  (data/run_0010)
#   C (GPU 2) — seed + aug combined
#
# ── CONFIG — edit these ───────────────────────────────────────────────────────
SEED_DIR="data/run_0000"
AUG_DIR="data/run_0010"
MODELS_DIR="models"
SNR=0
EPOCHS=500
USER_NOISE=3.0
GPU_A=0
GPU_B=1
GPU_C=2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p "$MODELS_DIR/logs"

for dir in "$SEED_DIR" "$AUG_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: $dir not found. Run scripts/generate_data.sh first."
        exit 1
    fi
done

BASE_ARGS="--seed-dir $SEED_DIR --aug-dir $AUG_DIR --models-dir $MODELS_DIR \
           --snr $SNR --epochs $EPOCHS --user-noise $USER_NOISE"

echo "============================================================"
echo "  train_aug.sh"
echo "  seed dir : $SEED_DIR"
echo "  aug  dir : $AUG_DIR"
echo "  models   : $MODELS_DIR"
echo "  SNR=$SNR   epochs=$EPOCHS   user_noise=$USER_NOISE"
echo "  GPU A=$GPU_A  GPU B=$GPU_B  GPU C=$GPU_C"
echo "============================================================"
echo ""

echo "Launching scenario A on GPU $GPU_A …"
CUDA_VISIBLE_DEVICES=$GPU_A python3 train_augmented.py $BASE_ARGS \
    --scenario A --device cuda \
    > "$MODELS_DIR/logs/scenario_A.log" 2>&1 &
PID_A=$!

echo "Launching scenario B on GPU $GPU_B …"
CUDA_VISIBLE_DEVICES=$GPU_B python3 train_augmented.py $BASE_ARGS \
    --scenario B --device cuda \
    > "$MODELS_DIR/logs/scenario_B.log" 2>&1 &
PID_B=$!

echo "Launching scenario C on GPU $GPU_C …"
CUDA_VISIBLE_DEVICES=$GPU_C python3 train_augmented.py $BASE_ARGS \
    --scenario C --device cuda \
    > "$MODELS_DIR/logs/scenario_C.log" 2>&1 &
PID_C=$!

echo ""
echo "All three jobs running in background."
echo "  Scenario A  PID $PID_A  → $MODELS_DIR/logs/scenario_A.log"
echo "  Scenario B  PID $PID_B  → $MODELS_DIR/logs/scenario_B.log"
echo "  Scenario C  PID $PID_C  → $MODELS_DIR/logs/scenario_C.log"
echo ""
echo "Waiting for all to finish …"
wait $PID_A && echo "  Scenario A done." || echo "  Scenario A FAILED (exit $?)"
wait $PID_B && echo "  Scenario B done." || echo "  Scenario B FAILED (exit $?)"
wait $PID_C && echo "  Scenario C done." || echo "  Scenario C FAILED (exit $?)"

echo ""
echo "============================================================"
echo "  Done. Results in $MODELS_DIR/results_aug.csv"
echo "  Models  in $MODELS_DIR/scenario_{A,B,C}/"
echo "  Logs    in $MODELS_DIR/logs/"
echo "============================================================"
