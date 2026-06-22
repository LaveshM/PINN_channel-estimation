# Physics-Informed Neural Networks for Wireless Channel Estimation

This is a fork of the official implementation of **"Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals"**, Alireza Javid and Nuria González-Prelcic (UC San Diego), NeurIPS 2025 Workshop *AI and ML for Next-Generation Wireless Communications and Networking (AI4NextG)*.

- Paper (OpenReview): <https://openreview.net/pdf?id=r3plaU6DvW>
- Original repository: <https://github.com/AlirezaJav/PINN_channel-estimation>

This fork restructures the repo for reproducibility and adds a dynamic blockage augmentation pipeline to evaluate model robustness under varying levels of NLOSv blockage.


## Repository structure

New files in src are marked with a `*`:`
```
PINN_channel-estimation/
├── src/
│   ├── Model.py                    # U-Net + transformer + cross-attention + loss
│   ├── find_in_map.py              # RSS map processor
│   ├── init_estimation.py          # LS-OFDM initial channel estimator
│   ├── make_correct_channels.py    # Ray-tracing CSV → channel tensor
│   ├── make_augmented_channels.py  # *Blockage augmentation pipeline
│   ├── train.py                    # Step 2: train the PINN
│   ├── train_augmented.py          # *Training with augmented data
│   ├── fine_tune.py                # Transfer learning experiment
│   ├── evaluate.py                 # *Step 3: run inference on all runs
│   └── make_plots.py               # *Step 4: generate plots from CSVs
├── scripts/ 
│   ├── 1_generate_data.sh          # Step 1
│   ├── 2_train.sh                  # Step 2
│   ├── 3_evaluate.sh               # Step 3
│   └── 4_plot.sh                   # Step 4
├── data/
│   ├── raw/                        # Input CSVs and RSS map images from original source
│   └── run_XXXX/                   # Generated per-run data (channels, LS, masks)
├── models/                         # Saved model weights
├── plots/                          # Generated output plots
└── requirements.txt
```


## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `scipy`, `pandas`, `opencv-python`, `Pillow`, `matplotlib`, `tqdm`.


## Pipeline

### Step 1 — Generate data

Generates `run_0000` (clean, no blockers) through `run_0010` (50 trucks) under `data/`. Each run contains the channel tensor, noisy UE positions, LS estimates, and blockage masks.

```bash
python src/make_augmented_channels.py \
    --csv data/raw/15GHz_concatenated_data.csv \
    --out-dir data \
    --n-runs 10 --step 5 \
    --seed 42 --atten-min 2.0 --atten-max 10.0 \
    --user-noise 3.0 --skip-summary \
    --start-run 0 --end-run 10 \
    --snr-list 0 --ls-modes adaptive
```

### Step 2 — Train

Trains the seed PINN on `run_0000` (clean data, SNR=0 dB, random split). Saves weights to `models/snr0/random_3.0/`.

```bash
python src/train.py \
    --smomp_file          data/run_0000/ls_snr+0.npy \
    --accurate_file       data/run_0000/channels.npy \
    --user_positions_file data/run_0000/locations_noisy.txt \
    --rss_image_path      data/raw/50_15GHz.jpg \
    --split_type          random \
    --user_noise          3.0 \
    --snr                 0 \
    --epochs              500 \
    --model_dir           models \
    --results_csv         models/results.csv \
    --continue_training
```

### Step 3 — Evaluate

Runs the trained model on all runs and saves per-user NMSE results to `data/run_XXXX/seed_predictions_snr+0.csv`.

```bash
python src/3_evaluate.py \
    --data-dir data \
    --model    models/snr0/random_3.0/simple_ls_val.pth
```

To use a different model or LS input:

```bash
python src/3_evaluate.py \
    --model   models/snr0/random_3.0/simple_ls_val.pth \
    --ls-file ls_snr+0.npy \
    --overwrite
```

### Step 4 — Plot

Reads the saved CSVs and generates plots under `plots/`.

```bash
python src/4_make_plot.py \
    --data-dir data \
    --out-dir  plots
```

---

All four steps are available as simple shell scripts for convenience:

```bash
bash scripts/1_generate_data.sh
bash scripts/2_train.sh
bash scripts/3_evaluate.sh
bash scripts/4_plot.sh
```

