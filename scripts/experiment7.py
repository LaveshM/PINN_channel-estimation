#!/usr/bin/env python3
"""
scripts/experiment7.py
----------------------
Cross-SNR generalisation experiment.

A single SNR=0 trained model is evaluated with LS inputs from every training
SNR {-10, -5, 0, +5}.  Both the train split and the test split are scored so
we can see whether the gap between splits changes as the mismatch grows.

The split indices (and the normalisation scale) are fixed to those produced by
the SNR=0 dataset build, which matches the model's training conditions.

Results → models/old/experiment7.csv
Plot    → plots/experiment7.png

Usage:
    python3 scripts/experiment7.py
    python3 scripts/experiment7.py --split-type bloc --device cpu
"""

import argparse
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

from Model import (
    ImprovedPhysicsInformedUNet,
    GlobalNormalizedDataset,
    create_datasets,
    set_seed,
)
from find_in_map import RSSMapProcessor

# ── constants ─────────────────────────────────────────────────────────────────
RSS_IMAGE   = "Dataset/50_15GHz.jpg"
BS_PIXEL    = (287, 293)
BS_REAL     = (71.06, 246.29)
IMG_WIDTH_M = 527.5
USER_NOISE  = 3.0
SEED        = 42
BATCH_SIZE  = 32
RUN_DIR     = "data/run_0000"

SNR_LIST    = [-10, -5, 0, 5]   # test SNRs


# ── helpers ───────────────────────────────────────────────────────────────────

def snr_tag(snr):
    return f"+{int(snr)}" if snr >= 0 else str(int(snr))


def model_ckpt(split_type, model_dir):
    return os.path.join(model_dir, "snr0", f"{split_type}_{USER_NOISE}", "simple_ls_val.pth")


def ls_file(snr, run_dir=RUN_DIR):
    return os.path.join(run_dir, f"ls_snr{snr_tag(snr)}.npy")


def load_model(ckpt_path, device):
    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def nmse_db_numpy(est, ch, idx):
    e = est[idx].astype(np.float32)
    t = ch[idx].astype(np.float32)
    return float(10.0 * np.log10(np.sum((e - t) ** 2) / np.sum(t ** 2)))


def model_nmse_db(model, loader, device):
    total_err = total_pow = 0.0
    with torch.no_grad():
        for smomp, accurate, rss in loader:
            pred     = model(smomp.to(device), rss.to(device))
            accurate = accurate.to(device)
            total_err += torch.sum((pred - accurate) ** 2).item()
            total_pow += torch.sum(accurate ** 2).item()
    if total_pow == 0:
        return float("nan")
    return float(10.0 * np.log10(total_err / total_pow))


def make_loader(smomp_path, ch_path, pos_path, rss_processor,
                norm_params, indices, split_label):
    """Build a DataLoader using a fixed normalisation and fixed split indices."""
    ds = GlobalNormalizedDataset(
        smomp_file=smomp_path,
        accurate_file=ch_path,
        user_positions_file=pos_path,
        rss_processor=rss_processor,
        normalization_params=norm_params,
        indices=indices,
        user_noise=USER_NOISE,
        split=split_label,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=2, pin_memory=True)


# ── plot ──────────────────────────────────────────────────────────────────────

def save_plot(df, out_path):
    snrs = sorted(df["test_snr"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, split, split_label in zip(
        axes,
        ["train", "test"],
        ["Train split", "Test split"],
    ):
        ls_col    = f"ls_nmse_{split}"
        model_col = f"model_nmse_{split}"

        sub = df.sort_values("test_snr")

        ax.plot(sub["test_snr"], sub[ls_col],
                color="black", linewidth=1.5, linestyle="--",
                marker="x", markersize=8, label="LS baseline")
        ax.plot(sub["test_snr"], sub[model_col],
                color="#4878CF", linewidth=2.5, linestyle="-",
                marker="o", markersize=8, label="SNR=0 model")

        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.set_xticks(snrs)
        ax.set_xticklabels([snr_tag(s) + " dB" for s in snrs])
        ax.set_xlabel("Test SNR", fontsize=11)
        ax.set_ylabel("NMSE (dB)", fontsize=11)
        ax.set_title(f"Experiment 7 — {split_label}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "SNR=0 model tested with LS inputs from different SNRs\n"
        "(vertical dashed = training SNR)",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir",  default="models")
    parser.add_argument("--run-dir",    default=RUN_DIR)
    parser.add_argument("--out-dir",    default="models/old")
    parser.add_argument("--split-type", default="random", choices=["random", "bloc"])
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snr-list",   nargs="+", type=float, default=SNR_LIST)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt = model_ckpt(args.split_type, args.model_dir)
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    model = load_model(ckpt, device)
    print(f"Loaded model : {ckpt}  →  {device}")

    ch_path  = os.path.join(args.run_dir, "channels.npy")
    pos_path = os.path.join(args.run_dir, "locations_noisy.txt")
    snr0_ls  = ls_file(0, args.run_dir)

    rss_processor = RSSMapProcessor(
        image_path=RSS_IMAGE, bs_pixel_coords=BS_PIXEL,
        bs_real_coords=BS_REAL, image_width_meters=IMG_WIDTH_M,
    )

    # Build the canonical SNR=0 split to get indices + normalisation params.
    print("\nBuilding SNR=0 split (canonical indices + norm params) …")
    set_seed(SEED)
    train_ds0, _, test_ds0, _, _, _ = create_datasets(
        smomp_file=snr0_ls,
        accurate_file=ch_path,
        user_positions_file=pos_path,
        split_type=args.split_type,
        user_noise=USER_NOISE,
        rss_processor=rss_processor,
    )
    norm_params   = train_ds0.normalization_params
    train_indices = train_ds0.indices.copy()
    test_indices  = test_ds0.indices.copy()
    print(f"  train={len(train_indices)}  test={len(test_indices)}")

    ch_arr = np.load(ch_path, mmap_mode="r")

    rows = []

    for test_snr in args.snr_list:
        ls_path = ls_file(test_snr, args.run_dir)
        if not os.path.exists(ls_path):
            print(f"\n[test_snr={test_snr:+.0f}] SKIP — {ls_path} missing")
            continue

        print(f"\n[test_snr={test_snr:+.0f} dB] {'─'*50}")

        ls_arr = np.load(ls_path, mmap_mode="r")
        ls_tr  = nmse_db_numpy(ls_arr, ch_arr, train_indices)
        ls_te  = nmse_db_numpy(ls_arr, ch_arr, test_indices)

        train_loader = make_loader(ls_path, ch_path, pos_path, rss_processor,
                                   norm_params, train_indices, "train")
        test_loader  = make_loader(ls_path, ch_path, pos_path, rss_processor,
                                   norm_params, test_indices,  "test")

        mdl_tr = model_nmse_db(model, train_loader, device)
        mdl_te = model_nmse_db(model, test_loader,  device)

        print(f"  LS   train: {ls_tr:+.2f} dB   test: {ls_te:+.2f} dB")
        print(f"  Model train: {mdl_tr:+.2f} dB   test: {mdl_te:+.2f} dB")

        rows.append({
            "test_snr":       int(test_snr),
            "split_type":     args.split_type,
            "ls_nmse_train":  round(ls_tr,  4),
            "ls_nmse_test":   round(ls_te,  4),
            "model_nmse_train": round(mdl_tr, 4),
            "model_nmse_test":  round(mdl_te, 4),
        })

    if not rows:
        print("No results — check LS files.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "experiment7.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  → {csv_path}")

    save_plot(df, "plots/experiment7.png")


if __name__ == "__main__":
    main()
