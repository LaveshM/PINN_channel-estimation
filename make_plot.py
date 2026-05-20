#!/usr/bin/env python3
"""
make_plot.py
------------
Diagnostic plots for the PINN channel-estimation project.

Plot 1 — UE locations coloured by bloc split (train / val / test)
Plot 2 — LS NMSE comparison across SNR values and LS modes (run_0000)
Plot 3 — Model test NMSE from models/results_pinn.csv: split_type × SNR

Usage:
    python3 make_plot.py                  # all three plots
    python3 make_plot.py --plots 1 3      # specific plots
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── constants ──────────────────────────────────────────────────────────────────
RUN_0000   = "data/run_0000"
OUT_DIR    = "data/plots/diagnostics"
BS_XY      = (71.06, 246.29)          # base-station real-world coords (m)
RESULTS_CSV = "models/results_pinn.csv"

SNR_COLORS = {-10: "#9467BD", -5: "#D65F5F", 0: "#4878CF", 5: "#6ACC65"}
SPLIT_STYLES = {
    "random": dict(color="#4878CF", marker="o", linestyle="-"),
    "bloc":   dict(color="#D65F5F", marker="s", linestyle="--"),
}
LS_MODES = ["adaptive", "fixed", "refnoise"]
LS_COLORS = {"adaptive": "#4878CF", "fixed": "#D65F5F", "refnoise": "#6ACC65"}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_positions(path):
    return np.loadtxt(path, dtype=np.float32)


def nmse_db(est, true):
    e = est.astype(np.float32)
    t = true.astype(np.float32)
    return float(10.0 * np.log10(np.sum((e - t) ** 2) / np.sum(t ** 2)))


def bloc_split(n_samples, block_size=100, seed=42):
    rng    = np.random.default_rng(seed)
    blocks = np.arange(n_samples // block_size)
    rng.shuffle(blocks)
    n_train = int(0.8 * len(blocks))
    n_val   = int(0.1 * len(blocks))
    train_b = set(blocks[:n_train])
    val_b   = set(blocks[n_train:n_train + n_val])
    test_b  = set(blocks[n_train + n_val:])

    idx = np.arange(n_samples)
    b   = idx // block_size
    return (idx[np.isin(b, list(train_b))],
            idx[np.isin(b, list(val_b))],
            idx[np.isin(b, list(test_b))])


# ── Plot 1 — UE locations by bloc split ────────────────────────────────────────

def plot_ue_locations(run_dir=RUN_0000, out_path=None, bs_xy=BS_XY):
    pos_file = os.path.join(run_dir, "locations_noisy.txt")
    if not os.path.exists(pos_file):
        print(f"[plot1] SKIP — {pos_file} not found")
        return

    pos = load_positions(pos_file)
    n   = len(pos)

    train_idx, val_idx, test_idx = bloc_split(n)

    fig, ax = plt.subplots(figsize=(7, 7))
    for label, idx, color in [("train", train_idx, "#4878CF"),
                               ("val",   val_idx,   "#6ACC65"),
                               ("test",  test_idx,  "#D65F5F")]:
        ax.scatter(pos[idx, 0], pos[idx, 1], s=4, alpha=0.5,
                   color=color, label=f"{label} ({len(idx)})")

    ax.scatter(*bs_xy, s=200, marker="*", color="black", zorder=5, label="BS")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("UE locations — bloc split")
    ax.legend(markerscale=3, fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(OUT_DIR, "ue_locations_bloc.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot 1 → {out_path}")


# ── Plot 2 — LS NMSE comparison ────────────────────────────────────────────────

def plot_ls_comparison(run_dir=RUN_0000, out_path=None):
    ch_file = os.path.join(run_dir, "channels.npy")
    if not os.path.exists(ch_file):
        print(f"[plot2] SKIP — {ch_file} not found")
        return

    ch = np.load(ch_file, mmap_mode="r")

    snr_tags = {-10: "snr-10", -5: "snr-5", 0: "snr+0", 5: "snr+5"}
    mode_files = {
        "adaptive": lambda t: f"ls_{t}.npy",
        "fixed":    lambda t: f"ls_{t}_fixed.npy",
        "refnoise": lambda t: f"ls_{t}_refnoise.npy",
    }

    records = []
    for snr, tag in snr_tags.items():
        for mode, fn in mode_files.items():
            ls_file = os.path.join(run_dir, fn(tag))
            if not os.path.exists(ls_file):
                continue
            ls = np.load(ls_file, mmap_mode="r")
            records.append({"snr": snr, "mode": mode, "nmse_db": nmse_db(ls, ch)})

    if not records:
        print("[plot2] SKIP — no LS files found")
        return

    df  = pd.DataFrame(records)
    snrs = sorted(df["snr"].unique())
    x    = np.arange(len(snrs))
    w    = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mode in enumerate(LS_MODES):
        sub = df[df["mode"] == mode].set_index("snr")
        vals = [sub.loc[s, "nmse_db"] if s in sub.index else float("nan") for s in snrs]
        ax.bar(x + (i - 1) * w, vals, w, label=mode, color=LS_COLORS[mode], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"SNR={s:+d} dB" for s in snrs])
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("LS NMSE — run_0000 (all SNRs and modes)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(OUT_DIR, "ls_nmse_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot 2 → {out_path}")


# ── Plot 3 — Model test NMSE from results_pinn.csv ─────────────────────────────

def plot_results(csv_path=RESULTS_CSV, out_path=None):
    if not os.path.exists(csv_path):
        print(f"[plot3] SKIP — {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # LS rows are the baseline; model rows have split_type in {random, bloc}
    ls_df    = df[df["split_type"] == "LS"].copy()
    model_df = df[df["split_type"] != "LS"].copy()

    if model_df.empty:
        print("[plot3] SKIP — no model rows in CSV")
        return

    # For duplicate runs keep the best val-checkpoint test NMSE per (snr, split_type)
    model_best = (model_df
                  .sort_values("test_nmse_val")
                  .groupby(["snr", "split_type"], as_index=False)
                  .first())

    ls_best = (ls_df
               .sort_values("test_nmse_val")
               .groupby("snr", as_index=False)
               .first())

    snrs = sorted(model_best["snr"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    # LS baseline
    if not ls_best.empty:
        ls_sorted = ls_best.sort_values("snr")
        ax.plot(ls_sorted["snr"], ls_sorted["test_nmse_val"],
                color="black", linewidth=1.5, linestyle="--",
                marker="x", markersize=8, label="LS baseline", zorder=5)

    # Model lines per split type
    for split, style in SPLIT_STYLES.items():
        sub = model_best[model_best["split_type"] == split].sort_values("snr")
        if sub.empty:
            continue
        ax.plot(sub["snr"], sub["test_nmse_val"],
                label=f"Model ({split} split)",
                linewidth=2, markersize=7, **style)

    ax.set_xticks(snrs)
    ax.set_xticklabels([f"{int(s):+d} dB" for s in snrs])
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("Test NMSE (dB)", fontsize=11)
    ax.set_title("Model test NMSE — random vs bloc split (val checkpoint)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(OUT_DIR, "results_pinn.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot 3 → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plots", nargs="+", type=int, default=[1, 2, 3],
                        help="Which plots to generate (1 2 3)")
    parser.add_argument("--run-dir",    default=RUN_0000)
    parser.add_argument("--out-dir",    default=OUT_DIR)
    parser.add_argument("--results-csv", default=RESULTS_CSV)
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = args.out_dir

    if 1 in args.plots:
        plot_ue_locations(run_dir=args.run_dir)
    if 2 in args.plots:
        plot_ls_comparison(run_dir=args.run_dir)
    if 3 in args.plots:
        plot_results(csv_path=args.results_csv)


if __name__ == "__main__":
    main()
