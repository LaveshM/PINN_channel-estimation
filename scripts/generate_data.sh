#!/usr/bin/env bash
# generate_data.sh
# ----------------
# Full data generation pipeline — run this once before training.
#
# Two blocker models, each swept over 10 runs (step=5 trucks/run):
#
#   Model A — trucks  (atten 2–10 dB/truck, seed 42)
#     run_0000 : 0  trucks  seed data, no blockage
#     run_0001 : 5  trucks
#     run_0002 : 10 trucks
#     ...
#     run_0010 : 50 trucks
#
#   Model B — cars  (atten 1–5 dB/vehicle, seed 43)
#     run_0011 : 5  vehicles   (same sweep positions as model A run_0001)
#     run_0012 : 10 vehicles
#     ...
#     run_0020 : 50 vehicles
#
# LS estimates are generated at SNR=0 (adaptive/fixed/refnoise) for every run.
# run_0000 also gets LS at SNR = -10, -5, +5 dB (adaptive only).
#
# Runs 1-10 and 11-20 are generated in parallel (N_JOBS at a time).
# run_0000 is always built first since all other runs load it as reference.
#
# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV="Dataset/15GHz_concatenated_data.csv"
OUT_DIR="data"
N_RUNS=10          # runs per model (total blocker-count increments)
STEP=5             # vehicles added per run
USER_NOISE=3.0     # GPS noise std-dev (m)

SEED_A=42          # truck model seed
ATTEN_MIN_A=2.0
ATTEN_MAX_A=10.0

SEED_B=43          # car model seed  (different placement from model A)
ATTEN_MIN_B=1.0
ATTEN_MAX_B=5.0

N_JOBS=1   # sequential — each job loads ~5-10 GB; run serially to avoid OOM and race conditions
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--csv $CSV --out-dir $OUT_DIR --n-runs $N_RUNS --step $STEP --user-noise $USER_NOISE --skip-summary"
BASE_A="$COMMON --seed $SEED_A --atten-min $ATTEN_MIN_A --atten-max $ATTEN_MAX_A"
BASE_B="$COMMON --seed $SEED_B --atten-min $ATTEN_MIN_B --atten-max $ATTEN_MAX_B --run-id-offset 10"

echo "============================================================"
echo "  generate_data.sh"
echo "  out_dir    : $OUT_DIR"
echo "  N_RUNS     : $N_RUNS per model  (runs 0-$N_RUNS truck, $((N_RUNS+1))-$((N_RUNS*2)) car)"
echo "  step       : $STEP vehicles/run"
echo "  Model A    : trucks  atten=[${ATTEN_MIN_A},${ATTEN_MAX_A}] dB  seed=$SEED_A"
echo "  Model B    : cars    atten=[${ATTEN_MIN_B},${ATTEN_MAX_B}] dB  seed=$SEED_B"
echo "  N_JOBS     : $N_JOBS parallel workers"
echo "============================================================"

# ── Step 1: run_0000 (seed data, no blockage) — must finish before anything else ──
echo ""
echo "[Step 1] Building run_0000 (seed data, all LS modes and SNRs) ..."
# shellcheck disable=SC2086
python3 make_augmented_channels.py $BASE_A \
    --start-run 0 --end-run 0 \
    --snr-list 0 -10 -5 5 \
    --ls-modes adaptive fixed refnoise

# ── Step 2: runs 1-10 (Model A) and 11-20 (Model B) in parallel ──────────────
echo ""
echo "[Step 2] Generating runs 1-$N_RUNS (trucks) and $((N_RUNS+1))-$((N_RUNS*2)) (cars) — $N_JOBS workers ..."

_run_job() {
    local r=$1 model=$2
    if [ "$model" = "A" ]; then
        # shellcheck disable=SC2086
        python3 make_augmented_channels.py $BASE_A \
            --start-run "$r" --end-run "$r" \
            --snr-list 0 \
            --ls-modes adaptive fixed refnoise \
            >> "$OUT_DIR/logs/run_$(printf '%04d' "$r").log" 2>&1
        echo "  [done] run_$(printf '%04d' "$r") (truck model)"
    else
        # shellcheck disable=SC2086
        python3 make_augmented_channels.py $BASE_B \
            --start-run "$r" --end-run "$r" \
            --snr-list 0 \
            --ls-modes adaptive fixed refnoise \
            >> "$OUT_DIR/logs/run_$(printf '%04d' "$((r+10))").log" 2>&1
        echo "  [done] run_$(printf '%04d' "$((r+10))") (car model, internal r=$r)"
    fi
}
export -f _run_job
export OUT_DIR BASE_A BASE_B N_RUNS

mkdir -p "$OUT_DIR/logs"

# Build job list: "1:A 2:A ... 10:A 1:B 2:B ... 10:B"
JOBS=""
for r in $(seq 1 $N_RUNS); do JOBS="$JOBS $r:A"; done
for r in $(seq 1 $N_RUNS); do JOBS="$JOBS $r:B"; done

# Run with N_JOBS parallelism using a simple slot-based loop
running=0
pids=()
for job in $JOBS; do
    r="${job%%:*}"
    model="${job##*:}"
    _run_job "$r" "$model" &
    pids+=($!)
    running=$((running + 1))
    if [ "$running" -ge "$N_JOBS" ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
        running=$((running - 1))
    fi
done
wait   # drain remaining

# ── Step 3: regenerate summary CSVs from all completed runs ───────────────────
echo ""
echo "[Step 3] Writing summary CSVs ..."
# shellcheck disable=SC2086
python3 make_augmented_channels.py $BASE_A \
    --start-run 0 --end-run 0 \
    --snr-list 0 \
    --ls-modes adaptive

echo ""
echo "============================================================"
echo "  Done."
echo "  run_0000           : seed data — all SNRs and LS modes"
echo "  run_0001-$(printf '%04d' $N_RUNS)      : truck model (${ATTEN_MIN_A}-${ATTEN_MAX_A} dB), SNR=0"
echo "  run_$(printf '%04d' $((N_RUNS+1)))-$(printf '%04d' $((N_RUNS*2)))  : car model   (${ATTEN_MIN_B}-${ATTEN_MAX_B} dB), SNR=0"
echo "  Logs               : $OUT_DIR/logs/"
echo "  Next               : bash scripts/train_all.sh"
echo "============================================================"
