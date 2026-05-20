#!/usr/bin/env python3
"""
Plot-only version of experiment1.py.

This script reads existing seed_predictions_*.csv files and generates the
experiment 1 plots/summary without importing torch or loading the model.
Run inference first with scripts/experiment1.py --mode infer if the CSVs do
not exist yet.
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


SEED_NORM_LS = "ls_snr+0.npy"
SPLIT_SEED = 42
TRAIN_RATIO = 0.8
SEED_NUM_USERS = 9877
SEED_GLOBAL_MAX = 1.8391441258813757e-08
TRUCKS_PER_RUN = 5


def seed_split(data_dir, norm_ls_tag, seed):
    """Reproduce the seed model's test split without importing torch."""
    run0 = os.path.join(data_dir, "run_0000")
    ls_path = os.path.join(run0, norm_ls_tag)
    if not os.path.exists(ls_path):
        n = SEED_NUM_USERS
    else:
        ls = np.load(ls_path, mmap_mode="r")
        n = len(ls)
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * 0.1)
    return perm[n_train + n_val:]


def run_meta(run_dir):
    """Return (run_id, n_trucks) from available run metadata."""
    jsp = os.path.join(run_dir, "blocked_summary.json")
    if os.path.exists(jsp):
        with open(jsp) as f:
            d = json.load(f)
        return int(d.get("run_id", -1)), int(d.get("n_trucks", -1))

    name = os.path.basename(run_dir)
    try:
        idx = int(name.split("_")[1])
    except Exception:
        idx = -1

    if idx >= 0:
        return idx, idx * TRUCKS_PER_RUN

    return idx, -1


def agg_nmse(err, pw):
    return float(10.0 * np.log10(err.sum() / max(float(pw.sum()), 1e-300)))


def stage_plot(args):
    os.makedirs(args.out_dir, exist_ok=True)

    test_idx = seed_split(args.data_dir, args.seed_norm_ls, SPLIT_SEED)
    test_set = set(test_idx.tolist())
    print(f"Using {len(test_set)} test users (seed model held-out split from n={SEED_NUM_USERS})\n")

    run_dirs = sorted(glob.glob(os.path.join(args.data_dir, "run_*")))
    records = {}
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        _, n_trucks = run_meta(run_dir)
        if n_trucks < 0:
            print(f"  warn: {os.path.basename(run_dir)} has no blocked_summary.json, skipping")
            continue
        for csv_f in sorted(glob.glob(os.path.join(run_dir, "seed_predictions_*.csv"))):
            tag = os.path.basename(csv_f)[len("seed_predictions_"):-len(".csv")]
            if args.ls_tags and tag not in args.ls_tags:
                continue
            df = pd.read_csv(csv_f)
            if "seed_nmse_db" not in df.columns:
                print(f"  warn: {csv_f} missing seed_nmse_db - re-run --mode infer --overwrite")
                continue
            if test_set is not None:
                df = df[df["user_idx"].isin(test_set)].reset_index(drop=True)
            records.setdefault(n_trucks, {})[tag] = df

    if not records:
        print("No seed_predictions CSVs found - run scripts/experiment1.py --mode infer first.")
        return

    ceiling_db = None
    if 0 in records:
        first_tag = next(iter(records[0]))
        df0_test = records[0][first_tag]
        ceiling_db = agg_nmse(df0_test["seed_err_sq"].values, df0_test["ch_pw"].values)
        print(f"Seed-on-clean ceiling = {ceiling_db:+.2f} dB  (run_0000, {len(df0_test)} test users)")

    all_tags = sorted({t for v in records.values() for t in v.keys()})
    plot_tags = args.ls_tags if args.ls_tags else all_tags
    print(f"Plotting LS tags: {plot_tags}")
    print(f"Runs (n_trucks): {sorted(records.keys())}\n")

    base_nmse = {}
    for tag in plot_tags:
        if 0 in records and tag in records[0]:
            base_nmse[tag] = records[0][tag].set_index("user_idx")["seed_nmse_db"]

    for tag in plot_tags:
        truck_vals = sorted(k for k in records if tag in records[k])
        if not truck_vals:
            print(f"  tag={tag}: no runs found, skipping plot")
            continue

        delta_boxes = []
        for n_trucks in truck_vals:
            df = records[n_trucks][tag]
            if tag in base_nmse:
                merged = df.set_index("user_idx").join(
                    base_nmse[tag].rename("base_nmse_db"), how="inner")
                delta_seed = (merged["seed_nmse_db"] - merged["base_nmse_db"]).values
            else:
                delta_seed = df["seed_nmse_db"].values
            delta_boxes.append(delta_seed)

        n_pts = len(truck_vals)
        x = np.arange(n_pts)
        w = 0.35
        box_color = "#4DAC26"

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
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
            patch.set_facecolor(box_color)
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

        out_path = os.path.join(args.out_dir, f"experiment1_{tag}.png")
        fig.savefig(out_path, dpi=150)
        pdf_out_path = os.path.join(args.out_dir, f"experiment1_{tag}.pdf")
        fig.savefig(pdf_out_path)
        plt.close(fig)
        print(f"Plot -> {out_path}")
        print(f"Plot -> {pdf_out_path}")

    rows = []
    for n_trucks, tag_dfs in records.items():
        for tag, df in tag_dfs.items():
            se = df["seed_err_sq"].values
            cp = df["ch_pw"].values
            le = df["ls_err_sq"].values
            rows.append(dict(
                n_trucks=n_trucks,
                ls_tag=tag,
                n_users=len(df),
                seed_nmse_agg=agg_nmse(se, cp),
                ls_nmse_agg=agg_nmse(le, cp),
                ceiling_db=ceiling_db,
            ))
    summary_df = pd.DataFrame(rows).sort_values(["ls_tag", "n_trucks"])
    csv_out = os.path.join(args.out_dir, "experiment1_summary.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"Summary CSV -> {csv_out}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument("--seed-norm-ls", default=SEED_NORM_LS,
                        help="LS filename in run_0000 used for seed model training normalization")
    parser.add_argument("--ls-tags", nargs="*", default=None,
                        help="LS tags to plot, e.g. snr+0 snr+0_refnoise. Default: all detected tags.")
    args = parser.parse_args()
    stage_plot(args)


if __name__ == "__main__":
    main()
