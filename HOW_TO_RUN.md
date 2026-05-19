# How to Run — Full Experiment Guide

Three independent experiments. Each writes to its own directory so they never conflict with each other.

| Experiment | Script | Output dir | What it trains |
|---|---|---|---|
| 1 — Standard PINN | `scripts/train_all.sh` | `data/` | All SNR × split_type, user_noise=0.5 m |
| 2 — Aug blockage (3-12 dB) | `scripts/run_augmented.sh` | `ndata/` | Scenarios A / B / C |
| 3 — Aug blockage (0-1 dB) | `scripts/run_n2data.sh` | `n2data/` | Scenarios A-low / B-low / C-low |

Experiments 1 and 2 are independent and can run simultaneously.
Experiment 3 requires Experiment 2's **seed datagen** to complete first.

---

## Prerequisites

```bash
pip install -r requirements.txt
# Dataset/15GHz_concatenated_data.csv must exist
```

GPU required. All scripts accept a `GPU=N` env var (default 0).

---

## Experiment 1 — Standard PINN (all SNR × split_type)

Generates ground-truth channels, LS estimates, UE positions, then trains one model per combination.

```bash
bash scripts/train_all.sh
```

**What it does internally:**
1. Generates `data/3D_channel_15GHz_2x2_Pt50.npy` from the CSV
2. Generates `data/UE/ue_positions_0.5.txt` (0.5 m GPS noise)
3. Generates LS estimates for SNR ∈ {-10, -5, 0, 5} dB in parallel
4. Pre-converts all `.npy` files to real+imag stacked (`_real.npy`) before any training starts
5. Trains 12 jobs (4 SNR × 3 splits) across 4 GPUs, 2 jobs per GPU

**Config** (top of `train_all.sh`):
```bash
GPU_IDS=(0 1 2 3)
MAX_JOBS_PER_GPU=2
USER_NOISE=0.5
SNR_LIST=(-10 -5 0 5)
SPLIT_LIST=(random bloc loc)
EPOCHS=500
```

**Outputs:**
- Checkpoints: `data/snr{N}/{split}_{noise}/`
- Results appended to: `data/results_pinn.csv`
- Logs: `logs/noise_0.5/snr{N}_{split}.log`

---

## Experiment 2 — Augmented blockage, 3-12 dB (ndata)

Generates seed channels, augmented channels with truck blockage (3-12 dB path gain attenuation), then trains three evaluation scenarios.

### Option A — All-in-one (single GPU)
```bash
GPU=0 bash scripts/run_augmented.sh
```

### Option B — Parallel (recommended, 3 GPUs)
```bash
# Step 1: generate all data on one GPU (~1-2 hours)
GPU=0 bash scripts/run_augmented.sh --datagen-only

# Step 2: train each scenario on its own GPU simultaneously
SCENARIO=A GPU=0 bash scripts/run_augmented.sh --skip-datagen
SCENARIO=B GPU=1 bash scripts/run_augmented.sh --skip-datagen
SCENARIO=C GPU=2 bash scripts/run_augmented.sh --skip-datagen
```

**What it does internally:**
0. Extracts clean UE positions from CSV → `data/UE/ue_positions.txt`
1. Generates seed channels → `ndata/seed_data.npy`
2. Generates aug channels (100 truck-blockage runs, 3-12 dB) → `ndata/aug_data.npy`
3. Copies clean seed positions → `ndata/ue_positions_noisy/`
3b. Generates noisy seed positions (0.5 m) → `ndata/ue_positions_0.5/`
3c. Generates noisy aug positions (0.5 m) → `ndata/aug_locations_0.5/`
4. Generates seed LS estimates → `ndata/seed_ls.npy`
5. Generates aug LS estimates → `ndata/aug_ls.npy`
6. Trains scenario(s) A / B / C

**Outputs:**
- `ndata/scenario_A/model_val.pth`, `results.json`, `training_curves.png`
- `ndata/scenario_B/...`
- `ndata/scenario_C/...`
- `ndata/results.json` (combined, written at the end of a full run)

**Scenarios explained:**
- **A** — Trained on seed data only; tests generalisation to aug test locations
- **B** — Trained on aug data (train-loc × train-run quadrant); 4-way loc/run split eval
- **C** — Trained on aug + seed combined; same 4-way eval + seed test

---

## Experiment 3 — Augmented blockage, 0-1 dB (n2data)

Same structure as Experiment 2 but with near-zero attenuation. Reuses seed channels and positions from `ndata/` via symlinks — only the aug channels and LS estimates are regenerated.

**Prerequisite:** Experiment 2 datagen must be complete.
```bash
ls ndata/seed_data.npy ndata/seed_ls.npy ndata/ue_positions_noisy/ue_positions_noisy.txt
```

### Option A — All-in-one
```bash
GPU=0 bash scripts/run_n2data.sh
```

### Option B — Parallel
```bash
bash scripts/run_n2data.sh --datagen-only

SCENARIO=A GPU=0 bash scripts/run_n2data.sh --skip-datagen
SCENARIO=B GPU=1 bash scripts/run_n2data.sh --skip-datagen
SCENARIO=C GPU=2 bash scripts/run_n2data.sh --skip-datagen
```

**Outputs:**
- `n2data/scenario_A-low/`, `n2data/scenario_B-low/`, `n2data/scenario_C-low/`
- `n2data/aug_data.npy`, `n2data/aug_ls.npy` (fresh, 0-1 dB)
- `n2data/aug_locations/`, `n2data/aug_meta/` (fresh)
- Symlinks into ndata/: `seed_data.npy`, `seed_ls.npy`, `ue_positions_noisy/`, `ue_positions_0.5/`

---

## Evaluation (mid-training or after)

To evaluate any scenario without re-training:

```bash
# Experiment 2 — ndata (3-12 dB, default)
python eval_checkpoint.py --scenario all --snr 0 --device cuda
python eval_checkpoint.py --scenario A   --device cuda:1

# Experiment 3 — n2data (0-1 dB, "low" suffix)
python eval_checkpoint.py --ndata n2data --scenario-suffix low --scenario all --snr 0 --device cuda
python eval_checkpoint.py --ndata n2data --scenario-suffix low --scenario B   --device cuda:1
```

`--ndata` sets the root data/checkpoint directory (default: `ndata`).
`--scenario-suffix` must match the suffix used during training (e.g. `low` → looks for `n2data/scenario_A-low/model_val.pth`).
Training-set entries are marked with `*`.

---

## Plots (after all experiments)

```bash
python scripts/make_plot.py
```

Plots saved to `plots/`:

| File | Content |
|---|---|
| `exp1_nmse_vs_snr.png` | PINN test NMSE vs SNR, one line per split_type (Exp 1) |
| `exp2_aug_scenarios_ndata.png` | All eval subsets for scenarios A/B/C, ndata (3-12 dB) |
| `exp3_aug_scenarios_n2data.png` | All eval subsets for scenarios A/B/C, n2data (0-1 dB) |
| `exp4_ndata_vs_n2data_unseen.png` | Side-by-side: unseen test NMSE, ndata vs n2data |
| `exp5_seed_generalisation.png` | Seed test NMSE, ndata vs n2data |
| `ue_locations.png` | UE spatial map with 5 highlighted aug runs |

Missing experiments are skipped with a message (partial runs produce partial plots).

---

## Conflict analysis — what each experiment reads and writes

| Path | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| `Dataset/15GHz_concatenated_data.csv` | reads | reads | — |
| `data/UE/ue_positions.txt` | — | writes | writes (idempotent) |
| `data/UE/ue_positions_0.5.txt` | writes | writes (idempotent) | writes (idempotent) |
| `data/3D_channel_*.npy` | writes | — | — |
| `data/snr{N}/` | writes | — | — |
| `data/results_pinn.csv` | appends | — | — |
| `ndata/seed_data.npy` | — | writes | symlinks (read-only) |
| `ndata/seed_ls.npy` | — | writes | symlinks (read-only) |
| `ndata/aug_data.npy` | — | writes | — |
| `ndata/aug_locations/` | — | writes | — |
| `ndata/aug_meta/` | — | writes | — |
| `ndata/scenario_{A,B,C}/` | — | writes | — |
| `n2data/aug_data.npy` | — | — | writes |
| `n2data/aug_locations/` | — | — | writes (fresh, not symlinked) |
| `n2data/aug_meta/` | — | — | writes (fresh, not symlinked) |
| `n2data/scenario_{A,B,C}-low/` | — | — | writes |

**No experiment overwrites another's output.**
The only shared writable path is `data/UE/ue_positions_0.5.txt` — all three produce the same content (same CSV, same seed=42), so simultaneous writes are safe.

---

## Common commands

```bash
# Monitor a running training job
tail -f logs/noise_0.5/snr0_random.log

# Check GPU usage
nvidia-smi

# Evaluate mid-training checkpoints (Exp 2 — ndata)
python eval_checkpoint.py --scenario all
# Evaluate mid-training checkpoints (Exp 3 — n2data)
python eval_checkpoint.py --ndata n2data --scenario-suffix low --scenario all

# Quick smoke-test (5 runs, 10 epochs, scenario A only)
N_RUNS=5 EPOCHS=10 SCENARIO=A GPU=0 bash scripts/run_augmented.sh
```


```bash
python scripts/inspect_blockage.py --run 1 --user 5 --data-dir ndata
python scripts/inspect_blockage.py --run 1 --user 5 --data-dir n2data
```
