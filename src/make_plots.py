#!/usr/bin/env python3
"""
Generate plots from saved evaluation CSVs.
Reads data/run_XXXX/seed_predictions_snr+0.csv → plots/

Usage:
    python src/make_plot.py
    python src/make_plot.py --out-dir plots
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED_NORM_LS   = "ls_snr+0.npy"
SPLIT_SEED     = 42
TRAIN_RATIO    = 0.8
TRUCKS_PER_RUN = 5


def seed_split(data_dir, seed):
    run0    = os.path.join(data_dir, "run_0000")
    ls_path = os.path.join(run0, SEED_NORM_LS)
    ls      = np.load(ls_path, mmap_mode="r")
    np.random.seed(seed)
    perm    = np.random.permutation(len(ls))
    n_train = int(len(ls) * TRAIN_RATIO)
    n_val   = int(len(ls) * 0.1)
    return perm[n_train + n_val:]


def run_meta(run_dir):
    jsp = os.path.join(run_dir, "blocked_summary.json")
    if os.path.exists(jsp):
        with open(jsp) as f:
            d = json.load(f)
        return int(d.get("run_id", -1)), int(d.get("n_trucks", -1))
    try:
        idx = int(os.path.basename(run_dir).split("_")[1])
        return idx, idx * TRUCKS_PER_RUN
    except Exception:
        return -1, -1


def agg_nmse(err, pw):
    return float(10.0 * np.log10(err.sum() / max(float(pw.sum()), 1e-300)))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir",  default="plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    test_idx = seed_split(args.data_dir, SPLIT_SEED)
    test_set = set(test_idx.tolist())
    print(f"Using {len(test_set)} test users\n")

    # load all seed_predictions_snr+0.csv files
    records = {}  # n_trucks → DataFrame
    for run_dir in sorted(glob.glob(os.path.join(args.data_dir, "run_*"))):
        if not os.path.isdir(run_dir):
            continue
        _, n_trucks = run_meta(run_dir)
        if n_trucks < 0:
            print(f"  warn: {os.path.basename(run_dir)} skipped — no metadata")
            continue
        csv_f = os.path.join(run_dir, "seed_predictions_snr+0.csv")
        if not os.path.exists(csv_f):
            print(f"  warn: {os.path.basename(run_dir)} skipped — no CSV")
            continue
        df = pd.read_csv(csv_f)
        df = df[df["user_idx"].isin(test_set)].reset_index(drop=True)
        records[n_trucks] = df

    if not records:
        print("No CSVs found — run evaluate.sh first.")
        return

    # ceiling: seed model on clean data (run_0000)
    ceiling_db = None
    if 0 in records:
        df0 = records[0]
        ceiling_db = agg_nmse(df0["seed_err_sq"].values, df0["ch_pw"].values)
        print(f"Seed-on-clean ceiling = {ceiling_db:+.2f} dB")

    # per-user baseline NMSE from run_0000
    base_nmse = None
    if 0 in records:
        base_nmse = records[0].set_index("user_idx")["seed_nmse_db"]

    # build delta boxes for runs with blockers
    truck_vals  = sorted(k for k in records if k > 0)
    delta_boxes = []
    for n_trucks in truck_vals:
        df = records[n_trucks]
        if base_nmse is not None:
            merged      = df.set_index("user_idx").join(base_nmse.rename("base_nmse_db"), how="inner")
            delta_seed  = (merged["seed_nmse_db"] - merged["base_nmse_db"]).values
        else:
            delta_seed  = df["seed_nmse_db"].values
        delta_boxes.append(delta_seed)

    # plot
    x   = np.arange(len(truck_vals))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bp = ax.boxplot(
        delta_boxes, positions=x, widths=w,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.25,
                        markerfacecolor="gray", markeredgecolor="gray"),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="gray", linewidth=1.0),
        capprops=dict(color="gray", linewidth=1.0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#4DAC26")
        patch.set_alpha(0.75)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in truck_vals], fontsize=9)
    ax.set_xlabel("Number of blockers", fontsize=11)
    ax.set_ylabel("Per-user Δ NMSE (dB)", fontsize=11)
    ax.set_title(
        "NMSE degradation of channel estimation model\n"
        "in the presence of dynamic blockers",
        fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, "experiment1_snr+0.png")
    out_pdf = os.path.join(args.out_dir, "experiment1_snr+0.pdf")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Plot → {out_png}")
    print(f"Plot → {out_pdf}")

    # summary CSV
    rows = []
    for n_trucks, df in records.items():
        rows.append(dict(
            n_trucks=n_trucks,
            n_users=len(df),
            seed_nmse_agg=agg_nmse(df["seed_err_sq"].values, df["ch_pw"].values),
            ls_nmse_agg=agg_nmse(df["ls_err_sq"].values,   df["ch_pw"].values),
            ceiling_db=ceiling_db,
        ))
    csv_out = os.path.join(args.out_dir, "experiment1_summary.csv")
    pd.DataFrame(rows).sort_values("n_trucks").to_csv(csv_out, index=False)
    print(f"Summary CSV → {csv_out}")


if __name__ == "__main__":
    main()
