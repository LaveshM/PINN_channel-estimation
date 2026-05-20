#!/usr/bin/env python3
"""
scripts/experiment6.py
----------------------
Generalisation experiment: seed-trained vs augmentation-trained model.

Four lines, all evaluated on the TEST SPLIT (random 80/10/10, seed=42)
of each run using ls_snr+0_refnoise.npy as input:

  1. LS refnoise        — raw LS estimate baseline (no model)
  2. Seed model → aug  — model trained on run_0000 (adaptive LS),
                         fed refnoise LS from each augmented run
  3. Aug model  → aug  — model trained on run_0010 (refnoise LS),
                         fed refnoise LS from each augmented run
  4. Seed model → seed — dashed reference: seed model on run_0000's own
                         test split (flat line showing clean-data ceiling)

X-axis: run_id (0 → 10)  /  secondary axis: number of trucks (0 → 20)

Model paths (random split, SNR=0, noise=3.0 m):
  seed : models/snr0/random_3.0/simple_ls_val.pth
  aug  : models/aug/refnoise/snr0/random_3.0/simple_ls_val.pth

Usage:
    python3 scripts/experiment6.py
    python3 scripts/experiment6.py --data-dir data --device cpu
"""

import argparse
import glob
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Model import ImprovedPhysicsInformedUNet, create_datasets, set_seed
from find_in_map import RSSMapProcessor

# ── constants ─────────────────────────────────────────────────────────────────
RSS_IMAGE   = "Dataset/50_15GHz.jpg"
BS_PIXEL    = (287, 293)
BS_REAL     = (71.06, 246.29)
IMG_WIDTH_M = 527.5
USER_NOISE  = 3.0
SPLIT_TYPE  = "random"
SEED        = 42
BATCH_SIZE  = 32
TRUCKS_PER_RUN = 2

SEED_MODEL_PATH = "models/snr0/random_3.0/simple_ls_val.pth"
AUG_MODEL_PATH  = "models/aug/refnoise/snr0/random_3.0/simple_ls_val.pth"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def nmse_db_arrays(est: np.ndarray, true: np.ndarray, idx: np.ndarray) -> float:
    e = est[idx].astype(np.float64)
    t = true[idx].astype(np.float64)
    return float(10.0 * np.log10(np.sum((e - t) ** 2) / np.sum(t ** 2)))


def model_nmse_db(model: torch.nn.Module, loader: DataLoader,
                  device: torch.device) -> float:
    total_err = total_pow = 0.0
    with torch.no_grad():
        for smomp, accurate, rss in loader:
            pred      = model(smomp.to(device), rss.to(device))
            accurate  = accurate.to(device)
            total_err += torch.sum((pred - accurate) ** 2).item()
            total_pow += torch.sum(accurate ** 2).item()
    if total_pow == 0:
        return float("nan")
    return float(10.0 * np.log10(total_err / total_pow))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--out-dir",    default="plots")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── load models ────────────────────────────────────────────────────────────
    for label, path in [("seed", SEED_MODEL_PATH), ("aug", AUG_MODEL_PATH)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} model not found: {path}")
            return

    seed_model = load_model(SEED_MODEL_PATH, device)
    aug_model  = load_model(AUG_MODEL_PATH,  device)
    print(f"Loaded seed model ← {SEED_MODEL_PATH}")
    print(f"Loaded aug  model ← {AUG_MODEL_PATH}\n")

    rss_processor = RSSMapProcessor(
        image_path=RSS_IMAGE, bs_pixel_coords=BS_PIXEL,
        bs_real_coords=BS_REAL, image_width_meters=IMG_WIDTH_M,
    )

    run_dirs = sorted(glob.glob(os.path.join(args.data_dir, "run_*")))
    if not run_dirs:
        print(f"No run_XXXX directories found in {args.data_dir}")
        return

    rows = []

    for run_dir in run_dirs:
        run_id  = int(os.path.basename(run_dir).split("_")[1])
        ch_path = os.path.join(run_dir, "channels.npy")
        ls_path = os.path.join(run_dir, "ls_snr+0_refnoise.npy")
        pos_path = os.path.join(run_dir, "locations_noisy.txt")

        for p, label in [(ch_path, "channels.npy"),
                          (ls_path,  "ls_snr+0_refnoise.npy"),
                          (pos_path, "locations_noisy.txt")]:
            if not os.path.exists(p):
                print(f"[run {run_id:04d}] SKIP — {label} missing")
                break
        else:
            pass
        if not (os.path.exists(ch_path) and os.path.exists(ls_path) and os.path.exists(pos_path)):
            continue

        print(f"[run {run_id:04d}]  {run_id * TRUCKS_PER_RUN} trucks  {'─'*44}")

        # -- reproducible test split (same indices across all runs) ------------
        set_seed(SEED)
        _, _, test_ds, _, _, _ = create_datasets(
            smomp_file=ls_path,
            accurate_file=ch_path,
            user_positions_file=pos_path,
            split_type=SPLIT_TYPE,
            user_noise=USER_NOISE,
            rss_processor=rss_processor,
        )

        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
        )

        # LS refnoise baseline
        ch_arr = np.load(ch_path, mmap_mode="r")
        ls_arr = np.load(ls_path, mmap_mode="r")
        ls_nmse = nmse_db_arrays(ls_arr, ch_arr, test_ds.indices)

        # seed model on this run's test split (refnoise LS input)
        seed_nmse = model_nmse_db(seed_model, test_loader, device)

        # aug model on this run's test split (refnoise LS input)
        aug_nmse  = model_nmse_db(aug_model,  test_loader, device)

        print(f"  LS refnoise NMSE :  {ls_nmse:+7.2f} dB")
        print(f"  Seed model NMSE  :  {seed_nmse:+7.2f} dB")
        print(f"  Aug  model NMSE  :  {aug_nmse:+7.2f} dB")

        rows.append({
            "run_id":    run_id,
            "n_trucks":  run_id * TRUCKS_PER_RUN,
            "ls_nmse":   round(ls_nmse,   4),
            "seed_nmse": round(seed_nmse, 4),
            "aug_nmse":  round(aug_nmse,  4),
        })

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows).sort_values("run_id")
    csv_path = os.path.join(args.out_dir, "experiment6.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  → {csv_path}")

    # -- seed-on-seed baseline: run_0000 test-split result (first row) ---------
    seed_on_seed = float(df[df["run_id"] == 0]["seed_nmse"].iloc[0]) if 0 in df["run_id"].values else None

    # ── plot ───────────────────────────────────────────────────────────────────
    run_ids = df["run_id"].values

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(run_ids, df["ls_nmse"].values,
            color="black",   linewidth=1.5,  linestyle="--",
            marker="x", markersize=7,
            label="LS refnoise (baseline)")

    ax.plot(run_ids, df["seed_nmse"].values,
            color="#4878CF", linewidth=2.0,  linestyle="-",
            marker="o", markersize=7,
            label="Seed model → aug test  (trained on run_0000)")

    ax.plot(run_ids, df["aug_nmse"].values,
            color="#D65F5F", linewidth=2.0,  linestyle="-",
            marker="s", markersize=7,
            label="Aug model  → aug test  (trained on run_0010, refnoise LS)")

    if seed_on_seed is not None:
        ax.axhline(seed_on_seed, color="#6ACC65", linewidth=2.0, linestyle=":",
                   label=f"Seed model → seed test  (run_0000 ceiling, {seed_on_seed:+.2f} dB)")

    ax.set_xticks(run_ids)
    ax.set_xticklabels([str(r) for r in run_ids])
    ax.set_xlabel("Run ID  (×2 trucks)", fontsize=11)
    ax.set_ylabel("NMSE (dB)", fontsize=11)
    ax.set_title(
        "Experiment 6 — Seed vs Aug-trained Model\n"
        "Input: ls_snr+0_refnoise  |  Eval: test split (seed=42)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)

    # secondary x-axis: truck count
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(run_ids)
    ax2.set_xticklabels([str(r * TRUCKS_PER_RUN) for r in run_ids], fontsize=8)
    ax2.set_xlabel("Number of trucks", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "experiment6.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot → {out_path}")


if __name__ == "__main__":
    main()
