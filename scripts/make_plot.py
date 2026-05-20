#!/usr/bin/env python3
"""
make_plot.py
Diagnostic plots for PINN channel estimation dataset.

Plot 1: UE locations coloured by train/val/test for the bloc split.
Plot 2: LS NMSE comparison — adaptive mode across SNRs, and three modes at SNR=0.

Usage:
    python3 make_plot.py
    python3 make_plot.py --run-dir data/run_0000 --out-dir data/plots/diagnostics
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── helpers ───────────────────────────────────────────────────────────────────

def load_positions(path):
    """Load locations file → (N, 3) float array, skipping header line."""
    positions = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if line:
            x, y, z = map(float, line.split())
            positions.append([x, y, z])
    return np.array(positions, dtype=float)


def nmse_db(est, true):
    """Global-sum NMSE in dB: 10 log10( ||est-true||^2 / ||true||^2 )."""
    return 10.0 * np.log10(np.sum((est - true) ** 2) / np.sum(true ** 2))


def bloc_split(n_samples, block_size=100, seed=42):
    """
    Replicate the bloc split from Model.create_datasets.
    Returns (train_indices, val_indices, test_indices).
    Uses seed=42 to match the default training seed.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    num_blocks = n_samples // block_size
    blocks = np.array_split(np.arange(n_samples), num_blocks)
    np.random.shuffle(blocks)
    n_blocks = len(blocks)
    train_blocks = blocks[:int(0.8 * n_blocks)]
    val_blocks   = blocks[int(0.8 * n_blocks):int(0.9 * n_blocks)]
    test_blocks  = blocks[int(0.9 * n_blocks):]
    train_idx = np.concatenate(train_blocks)
    val_idx   = np.concatenate(val_blocks)
    test_idx  = np.concatenate(test_blocks)
    np.random.set_state(rng_state)
    return train_idx, val_idx, test_idx


# ── Plot 1: UE locations by split ────────────────────────────────────────────

def plot_ue_locations(run_dir, out_path, bs_xy=(71.06, 246.29)):
    """Scatter plot of UE positions coloured by bloc train/val/test split."""
    pos_file = os.path.join(run_dir, "locations_noisy.txt")
    if not os.path.exists(pos_file):
        print(f"[plot1] Skipping — {pos_file} not found")
        return

    pos = load_positions(pos_file)
    n = len(pos)
    train_idx, val_idx, test_idx = bloc_split(n)

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(pos[train_idx, 0], pos[train_idx, 1],
               c="#4878CF", s=4, alpha=0.4, rasterized=True,
               label=f"Train  ({len(train_idx):,})")
    ax.scatter(pos[val_idx, 0],   pos[val_idx, 1],
               c="#6ACC65", s=15, alpha=0.85, zorder=3,
               label=f"Val    ({len(val_idx):,})")
    ax.scatter(pos[test_idx, 0],  pos[test_idx, 1],
               c="#D65F5F", s=15, alpha=0.85, zorder=3,
               label=f"Test   ({len(test_idx):,})")
    ax.scatter(*bs_xy, marker="*", s=300, c="gold",
               edgecolors="black", linewidths=0.8, zorder=5, label="BS")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"UE locations — bloc split  (N={n:,}, block_size=100, seed=42)")
    ax.legend(markerscale=2.5, fontsize=10, loc="upper right")
    ax.set_aspect("equal", "datalim")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot1] Saved → {out_path}")


# ── Plot 2: LS NMSE comparison ────────────────────────────────────────────────

def plot_ls_comparison(run_dir, out_path):
    """
    Left panel:  LS NMSE vs SNR for adaptive mode.
    Right panel: LS NMSE for all three modes at SNR=0.
    """
    ch_path = os.path.join(run_dir, "channels.npy")
    if not os.path.exists(ch_path):
        print(f"[plot2] Skipping — {ch_path} not found")
        return

    true = np.load(ch_path).astype(np.float32)

    # --- adaptive mode: SNR sweep ---
    snr_specs = [("-10", "ls_snr-10.npy"),
                 ("-5",  "ls_snr-5.npy"),
                 ("+0",  "ls_snr+0.npy"),
                 ("+5",  "ls_snr+5.npy")]
    snr_labels, snr_nmses = [], []
    for lbl, fname in snr_specs:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            est = np.load(fpath).astype(np.float32)
            snr_nmses.append(nmse_db(est, true))
            snr_labels.append(lbl)

    # --- three modes at SNR=0 ---
    mode_specs = [("adaptive",  "ls_snr+0.npy"),
                  ("fixed",     "ls_snr+0_fixed.npy"),
                  ("refnoise",  "ls_snr+0_refnoise.npy")]
    mode_labels, mode_nmses = [], []
    for lbl, fname in mode_specs:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            est = np.load(fpath).astype(np.float32)
            mode_nmses.append(nmse_db(est, true))
            mode_labels.append(lbl)

    if not snr_nmses and not mode_nmses:
        print(f"[plot2] No LS files found in {run_dir}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: SNR sweep line plot
    ax = axes[0]
    if snr_nmses:
        x_pos = range(len(snr_labels))
        ax.plot(snr_labels, snr_nmses, marker="o", color="#4878CF",
                linewidth=2, markersize=8)
        for i, (x, y) in enumerate(zip(snr_labels, snr_nmses)):
            ax.annotate(f"{y:.2f} dB", (x, y),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color="#333333")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("LS NMSE vs SNR — adaptive mode")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    # Right: mode comparison bar chart
    ax = axes[1]
    if mode_nmses:
        bar_colors = ["#4878CF", "#D65F5F", "#6ACC65"][:len(mode_labels)]
        bars = ax.bar(mode_labels, mode_nmses, color=bar_colors,
                      edgecolor="black", linewidth=0.7, width=0.5)
        for bar, val in zip(bars, mode_nmses):
            ypos = val + (0.15 if val >= 0 else -0.35)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("LS noise mode")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("LS NMSE at SNR = 0 dB — mode comparison")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot2] Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic plots for PINN channel estimation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default="data/data/run_0000",
                        help="Run directory containing channels.npy and locations_noisy.txt")
    parser.add_argument("--out-dir", default="plots/diagnostics",
                        help="Output directory for saved plots")
    parser.add_argument("--bs-x",   type=float, default=71.06)
    parser.add_argument("--bs-y",   type=float, default=246.29)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    plot_ue_locations(
        args.run_dir,
        os.path.join(args.out_dir, "ue_locations_bloc_split.png"),
        bs_xy=(args.bs_x, args.bs_y),
    )

    plot_ls_comparison(
        args.run_dir,
        os.path.join(args.out_dir, "ls_nmse_comparison.png"),
    )


if __name__ == "__main__":
    main()
