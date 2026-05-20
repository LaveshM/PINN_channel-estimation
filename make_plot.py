#!/usr/bin/env python3
"""
make_plot.py
------------
Diagnostic plots for the PINN channel-estimation project.

Plot 1 — UE locations coloured by bloc split (train / val / test)
Plot 2 — LS NMSE comparison across SNR values and LS modes (run_0000)
Plot 3 — Model test NMSE from models/results_pinn.csv: split_type × SNR
Plot 4 — Aug4 model NMSE on run_0010 test split, 3×3 cross-mode matrix,
          blocked users only
Plot 5 — Mean channel attenuation vs run_0000 by run (blocked vs unblocked)
Plot 6 — Aug4 models on blocked users: new vs old data, run_0010 & run_0020

Usage:
    python3 make_plot.py                  # plots 1-3
    python3 make_plot.py --plots 4        # aug4 3×3 blocked matrix
    python3 make_plot.py --plots 6        # aug4 new vs old data comparison
    python3 make_plot.py --plots 1 3 4 6  # specific plots
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


# ── Plot 4 — Aug4 model NMSE: blocked / unblocked / all on run_0010 ────────────

AUG_RUN_DIR   = "data/run_0010"
AUG_MODEL_DIR = "models/aug"

_LS_MODES = {
    "adaptive": "ls_snr+0.npy",
    "fixed":    "ls_snr+0_fixed.npy",
    "refnoise": "ls_snr+0_refnoise.npy",
}
_MODE_COLORS = {
    "adaptive": "#4878CF",
    "fixed":    "#D65F5F",
    "refnoise": "#6ACC65",
}

RSS_IMAGE   = "Dataset/50_15GHz.jpg"
BS_PIXEL    = (287, 293)
BS_REAL     = (71.06, 246.29)
IMG_WIDTH_M = 527.5
_USER_NOISE = 3.0
_SEED       = 42
_BATCH      = 32


def _load_aug_model(ckpt_path, device):
    try:
        import torch
        from Model import ImprovedPhysicsInformedUNet
    except ImportError as exc:
        raise exc
    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def _model_nmse_subset(model, test_ds, local_positions, device):
    """NMSE (dB) for a subset of the test DataLoader, given local position indices."""
    import torch
    from torch.utils.data import DataLoader, Subset
    if not local_positions:
        return float("nan")
    loader = DataLoader(
        Subset(test_ds, local_positions),
        batch_size=_BATCH, shuffle=False, num_workers=2, pin_memory=True,
    )
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


def _ls_nmse_subset(ls_arr, ch_arr, global_idx):
    if len(global_idx) == 0:
        return float("nan")
    e = ls_arr[global_idx].astype(np.float32)
    t = ch_arr[global_idx].astype(np.float32)
    return float(10.0 * np.log10(np.sum((e - t) ** 2) / np.sum(t ** 2)))


def plot_aug4_blocked(run_dir=AUG_RUN_DIR, aug_model_dir=AUG_MODEL_DIR,
                      out_path=None, device_str=None):
    """
    Plot 4 — 3×3 cross-mode NMSE matrix on run_0010 blocked test users.
    Rows = LS mode the model was trained with.
    Cols = LS mode fed at test time.
    Values = NMSE (dB) on blocked users only.
    Each model is evaluated with its own training normalization applied to the
    test-time LS input so the scale mismatch is exactly what the model sees.
    Requires torch, Model.py, find_in_map.py.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from Model import (ImprovedPhysicsInformedUNet, GlobalNormalizedDataset,
                           create_datasets, set_seed)
        from find_in_map import RSSMapProcessor
    except ImportError as exc:
        print(f"[plot4] SKIP — {exc}")
        return

    ch_path   = os.path.join(run_dir, "channels.npy")
    mask_path = os.path.join(run_dir, "blocked_mask.npy")
    pos_path  = os.path.join(run_dir, "locations_noisy.txt")
    for p in [ch_path, mask_path, pos_path]:
        if not os.path.exists(p):
            print(f"[plot4] SKIP — {p} not found")
            return

    model_paths = {
        mode: os.path.join(aug_model_dir, mode, "snr0", "random_3.0", "simple_ls_val.pth")
        for mode in _LS_MODES
    }
    ls_paths = {mode: os.path.join(run_dir, fname) for mode, fname in _LS_MODES.items()}
    for label, paths in [("checkpoint", model_paths), ("LS file", ls_paths)]:
        for mode, p in paths.items():
            if not os.path.exists(p):
                print(f"[plot4] SKIP — {label} not found: {p}")
                return

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    rss_proc     = RSSMapProcessor(
        image_path=RSS_IMAGE, bs_pixel_coords=BS_PIXEL,
        bs_real_coords=BS_REAL, image_width_meters=IMG_WIDTH_M,
    )
    blocked_mask = np.load(mask_path).astype(bool)
    ch_arr       = np.load(ch_path, mmap_mode="r")

    modes = list(_LS_MODES.keys())  # ["adaptive", "fixed", "refnoise"]
    n     = len(modes)

    # nmse_matrix[i, j] = model trained on modes[i], fed modes[j] LS, blocked only
    nmse_matrix  = np.full((n, n), np.nan)
    # ls_nmse[j]   = LS baseline NMSE for modes[j] on blocked users
    ls_nmse_vec  = np.full(n, np.nan)

    blocked_local  = None  # will be set on first train_mode pass
    blocked_global = None
    n_blocked = n_test = 0

    for i, train_mode in enumerate(modes):
        print(f"\n[plot4] train_mode={train_mode}")

        set_seed(_SEED)
        train_ds, _, test_ds, _, _, _ = create_datasets(
            smomp_file=ls_paths[train_mode],
            accurate_file=ch_path,
            user_positions_file=pos_path,
            split_type="random",
            user_noise=_USER_NOISE,
            rss_processor=rss_proc,
        )
        norm_params  = train_ds.normalization_params
        test_indices = test_ds.indices

        if blocked_local is None:
            blocked_local  = [k for k, ri in enumerate(test_indices) if blocked_mask[ri]]
            blocked_global = (test_indices[np.array(blocked_local, dtype=int)]
                              if blocked_local else np.array([], dtype=int))
            n_blocked = len(blocked_local)
            n_test    = len(test_indices)
            print(f"  test split: {n_test} total, {n_blocked} blocked "
                  f"({100*n_blocked/n_test:.1f}%)")

        model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
        ckpt  = torch.load(model_paths[train_mode], map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model = model.to(device).eval()

        for j, test_mode in enumerate(modes):
            # build cross-mode dataset: test_mode LS + train_mode normalization
            cross_ds = GlobalNormalizedDataset(
                smomp_file=ls_paths[test_mode],
                accurate_file=ch_path,
                user_positions_file=pos_path,
                rss_processor=rss_proc,
                normalization_params=norm_params,
                indices=test_indices,
                user_noise=_USER_NOISE,
                split="test",
            )

            # LS NMSE for this test_mode (only need once, independent of train_mode)
            if i == 0:
                ls_arr = np.load(ls_paths[test_mode], mmap_mode="r")
                ls_nmse_vec[j] = _ls_nmse_subset(ls_arr, ch_arr, blocked_global)

            # model NMSE on blocked users
            nmse_matrix[i, j] = _model_nmse_subset(model, cross_ds, blocked_local, device)
            print(f"  test={test_mode:9s}: model={nmse_matrix[i,j]:+.2f} dB  "
                  f"LS={ls_nmse_vec[j]:+.2f} dB")

        del model

    print(f"\n[plot4] Blocked users in run_0010 test split: {n_blocked} / {n_test}")

    # ── heatmap ───────────────────────────────────────────────────────────────
    # Build a (n+1) × n display matrix: first row = LS baseline (repeated),
    # rows 1..n = model rows.
    disp = np.vstack([ls_nmse_vec[np.newaxis, :], nmse_matrix])   # shape (4, 3)
    row_labels = ["LS baseline"] + [f"Model\n(trained: {m})" for m in modes]
    col_labels = [f"{m}\n(LS: {ls_nmse_vec[j]:+.1f} dB)" for j, m in enumerate(modes)]

    vmin = np.nanmin(disp)
    vmax = np.nanmax(disp)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(disp, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)

    # annotate cells
    for r in range(disp.shape[0]):
        for c in range(disp.shape[1]):
            val = disp[r, c]
            if not np.isnan(val):
                text_color = "white" if (val - vmin) / (vmax - vmin + 1e-9) > 0.7 else "black"
                ax.text(c, r, f"{val:+.1f} dB", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    # highlight diagonal of model block (rows 1..n, cols 0..n-1)
    for k in range(n):
        rect = plt.Rectangle((k - 0.5, k + 1 - 0.5), 1, 1,
                              linewidth=2.5, edgecolor="navy", facecolor="none")
        ax.add_patch(rect)

    # horizontal separator between LS row and model rows
    ax.axhline(0.5, color="black", linewidth=2)

    ax.set_xticks(range(n))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Test-time LS mode", fontsize=10)
    ax.set_ylabel("Model / baseline", fontsize=10)
    ax.set_title(
        f"Aug4 — blocked-user NMSE (dB), run_0010 test split\n"
        f"n_blocked={n_blocked}  n_test={n_test}  SNR=0  random split (seed=42)\n"
        f"Diagonal (navy box) = same LS mode for training and testing",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="NMSE (dB)", fraction=0.03, pad=0.02)
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(OUT_DIR, "aug4_blocked_matrix.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot 4 → {out_path}")


# ── Plot 5 — Mean attenuation vs run_0000 by run, blocked vs unblocked ─────────

def plot_attenuation_by_run(data_dir="data", out_path=None):
    """
    Plot 5 — For each run 1-20, compute per-user channel power loss relative to
    run_0000, separately for blocked and unblocked users. Shows mean ± std as a
    grouped bar chart with a shaded band marking the real-world 3-12 dB range.
    """
    ch0_path = os.path.join(data_dir, "run_0000", "channels.npy")
    if not os.path.exists(ch0_path):
        print(f"[plot5] SKIP — {ch0_path} not found")
        return

    ch0 = np.load(ch0_path, mmap_mode="r")
    pwr0 = np.sum(np.abs(ch0) ** 2, axis=(1, 2, 3)).astype(np.float64)

    run_ids, blk_mean, blk_std, ublk_mean, ublk_std, blk_pct = [], [], [], [], [], []

    r = 1
    while True:
        run_dir = os.path.join(data_dir, f"run_{r:04d}")
        ch_path   = os.path.join(run_dir, "channels.npy")
        mask_path = os.path.join(run_dir, "blocked_mask.npy")
        if not os.path.exists(ch_path) or not os.path.exists(mask_path):
            break

        ch   = np.load(ch_path, mmap_mode="r")
        mask = np.load(mask_path).astype(bool)
        pwr  = np.sum(np.abs(ch) ** 2, axis=(1, 2, 3)).astype(np.float64)

        # per-user attenuation vs run_0000 (positive = signal weaker than seed)
        atten = 10.0 * np.log10(pwr0 / (pwr + 1e-300))

        blk_vals  = atten[mask]
        ublk_vals = atten[~mask]

        run_ids.append(r)
        blk_mean.append(blk_vals.mean()  if len(blk_vals)  > 0 else np.nan)
        blk_std.append(blk_vals.std()    if len(blk_vals)  > 0 else np.nan)
        ublk_mean.append(ublk_vals.mean() if len(ublk_vals) > 0 else np.nan)
        ublk_std.append(ublk_vals.std()   if len(ublk_vals) > 0 else np.nan)
        blk_pct.append(100.0 * mask.sum() / len(mask))

        r += 1

    if not run_ids:
        print("[plot5] SKIP — no runs found beyond run_0000")
        return

    run_ids  = np.array(run_ids)
    blk_mean  = np.array(blk_mean)
    blk_std   = np.array(blk_std)
    ublk_mean = np.array(ublk_mean)
    ublk_std  = np.array(ublk_std)
    blk_pct   = np.array(blk_pct)

    x   = np.arange(len(run_ids))
    w   = 0.38
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={"height_ratios": [3, 1]})

    # shaded real-world reference band (3–12 dB)
    ax.axhspan(3, 12, color="#F0E68C", alpha=0.4, label="Real-world range (3–12 dB)")

    bars_blk = ax.bar(x - w / 2, blk_mean, w, yerr=blk_std, capsize=3,
                      label="Blocked vs run_0000",
                      color="#D65F5F", alpha=0.85, edgecolor="black", linewidth=0.5,
                      error_kw=dict(elinewidth=1, ecolor="black", capthick=1))

    bars_ublk = ax.bar(x + w / 2, ublk_mean, w, yerr=ublk_std, capsize=3,
                       label="Unblocked vs run_0000",
                       color="#4878CF", alpha=0.85, edgecolor="black", linewidth=0.5,
                       error_kw=dict(elinewidth=1, ecolor="black", capthick=1))

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"run_{i:04d}" for i in run_ids], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Attenuation vs run_0000 (dB)", fontsize=11)
    ax.set_title("Mean channel power attenuation relative to run_0000\n"
                 "Blocked vs unblocked users  |  error bars = ±1 std dev", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # add a vertical separator between runs 1-10 and 11+
    n_old = sum(1 for i in run_ids if i <= 10)
    if n_old > 0 and n_old < len(run_ids):
        ax.axvline(n_old - 0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.text(n_old - 0.5 + 0.1, ax.get_ylim()[1] * 0.95,
                "new model →", fontsize=8, va="top", color="black", alpha=0.7)

    # lower panel: blockage percentage per run
    ax2.bar(x, blk_pct, color="#888888", alpha=0.7, edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"run_{i:04d}" for i in run_ids], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Blocked (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.axhline(100, color="gray", linewidth=0.6, linestyle=":")
    ax2.grid(True, axis="y", alpha=0.3)
    if n_old > 0 and n_old < len(run_ids):
        ax2.axvline(n_old - 0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join("data", "plots", "attenuation_by_run.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot 5 → {out_path}")


# ── Plot 6 — Aug4 models on blocked test users: new vs old data, run_0010 & run_0020

_TARGET_RUNS = [
    ("new", "data",     10),
    ("new", "data",     20),
    ("old", "data_old", 10),
    ("old", "data_old", 20),
]


def plot_aug4_cross_dataset(aug_model_dir=AUG_MODEL_DIR, out_path=None, device_str=None):
    """
    Plot 6 — Evaluate the 3 aug models (adaptive / fixed / refnoise) on blocked
    test users only, across 4 datasets:
        new data  run_0010,  new data  run_0020,
        old data  run_0010,  old data  run_0020.
    Each model is evaluated using its own matching LS mode.
    Produces one subplot per LS mode; x-axis = dataset; bars = LS vs model.
    """
    try:
        import torch
        from Model import ImprovedPhysicsInformedUNet, create_datasets, set_seed
        from find_in_map import RSSMapProcessor
    except ImportError as exc:
        print(f"[plot6] SKIP — {exc}")
        return

    model_paths = {
        mode: os.path.join(aug_model_dir, mode, "snr0", "random_3.0", "simple_ls_val.pth")
        for mode in _LS_MODES
    }
    for mode, mp in model_paths.items():
        if not os.path.exists(mp):
            print(f"[plot6] SKIP — checkpoint not found: {mp}")
            return

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    rss_proc = RSSMapProcessor(
        image_path=RSS_IMAGE, bs_pixel_coords=BS_PIXEL,
        bs_real_coords=BS_REAL, image_width_meters=IMG_WIDTH_M,
    )

    # results[mode][dataset_label] = {"ls": float, "model": float,
    #                                  "n_blocked": int, "n_test": int}
    results = {mode: {} for mode in _LS_MODES}

    for vintage, data_root, run_id in _TARGET_RUNS:
        run_dir  = os.path.join(data_root, f"run_{run_id:04d}")
        ch_path  = os.path.join(run_dir, "channels.npy")
        mask_path = os.path.join(run_dir, "blocked_mask.npy")
        pos_path  = os.path.join(run_dir, "locations_noisy.txt")
        label    = f"{vintage}\nrun_{run_id:04d}"

        for p in [ch_path, mask_path, pos_path]:
            if not os.path.exists(p):
                print(f"[plot6] SKIP {label} — {p} missing")
                break
        else:
            blocked_mask = np.load(mask_path).astype(bool)
            ch_arr       = np.load(ch_path, mmap_mode="r")

            for mode, ls_fname in _LS_MODES.items():
                ls_path = os.path.join(run_dir, ls_fname)
                if not os.path.exists(ls_path):
                    print(f"[plot6] SKIP mode={mode} {label} — {ls_path} missing")
                    continue

                print(f"[plot6] {label.replace(chr(10),' ')}  mode={mode} …")

                set_seed(_SEED)
                train_ds, _, test_ds, _, _, _ = create_datasets(
                    smomp_file=ls_path,
                    accurate_file=ch_path,
                    user_positions_file=pos_path,
                    split_type="random",
                    user_noise=_USER_NOISE,
                    rss_processor=rss_proc,
                )
                test_idx     = test_ds.indices
                blocked_local  = [k for k, ri in enumerate(test_idx) if blocked_mask[ri]]
                blocked_global = (test_idx[np.array(blocked_local, dtype=int)]
                                  if blocked_local else np.array([], dtype=int))

                n_test    = len(test_idx)
                n_blocked = len(blocked_local)

                ls_arr  = np.load(ls_path, mmap_mode="r")
                ls_nmse = _ls_nmse_subset(ls_arr, ch_arr, blocked_global)

                model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
                ckpt  = torch.load(model_paths[mode], map_location=device)
                state = (ckpt["model_state_dict"]
                         if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
                model.load_state_dict(state)
                model = model.to(device).eval()

                mdl_nmse = _model_nmse_subset(model, test_ds, blocked_local, device)
                del model

                results[mode][label] = {
                    "ls": ls_nmse, "model": mdl_nmse,
                    "n_blocked": n_blocked, "n_test": n_test,
                }
                print(f"  n_blocked={n_blocked}/{n_test}  "
                      f"LS={ls_nmse:+.2f} dB  Model={mdl_nmse:+.2f} dB")

    # ── build ordered label list ──────────────────────────────────────────────
    dataset_labels = [f"{v}\nrun_{r:04d}" for v, _, r in _TARGET_RUNS]
    modes = list(_LS_MODES.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    bw = 0.35
    x  = np.arange(len(dataset_labels))

    for ax, mode in zip(axes, modes):
        color    = _MODE_COLORS[mode]
        ls_vals  = [results[mode].get(lbl, {}).get("ls",    float("nan")) for lbl in dataset_labels]
        mdl_vals = [results[mode].get(lbl, {}).get("model", float("nan")) for lbl in dataset_labels]
        n_blks   = [results[mode].get(lbl, {}).get("n_blocked", 0) for lbl in dataset_labels]
        n_tests  = [results[mode].get(lbl, {}).get("n_test",    0) for lbl in dataset_labels]

        bars_ls  = ax.bar(x - bw / 2, ls_vals,  bw, label="LS baseline",
                          color=color, alpha=0.4, hatch="//",
                          edgecolor="black", linewidth=0.6)
        bars_mdl = ax.bar(x + bw / 2, mdl_vals, bw, label="Aug model",
                          color=color, alpha=0.9,
                          edgecolor="black", linewidth=0.6)

        for bar, val in list(zip(bars_ls, ls_vals)) + list(zip(bars_mdl, mdl_vals)):
            if not np.isnan(val):
                yoff = 0.3 if val >= 0 else -1.2
                ax.text(bar.get_x() + bar.get_width() / 2, val + yoff,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)

        # show n_blocked/n_test under each x tick
        x_tick_labels = [
            f"{lbl}\n(blk={nb}/{nt})"
            for lbl, nb, nt in zip(dataset_labels, n_blks, n_tests)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, fontsize=8)
        ax.set_title(f"LS mode: {mode}", fontsize=10)
        ax.set_ylabel("NMSE (dB) — blocked users only", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

        # vertical separator between new / old data groups
        ax.axvline(1.5, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.text(0.75, ax.get_ylim()[0] * 0.97, "new data",
                ha="center", fontsize=8, color="gray")
        ax.text(2.5, ax.get_ylim()[0] * 0.97, "old data",
                ha="center", fontsize=8, color="gray")

    fig.suptitle(
        "Aug4 models — blocked-user NMSE on new vs old data (run_0010 & run_0020)\n"
        "Each model evaluated with its own matching LS mode  |  random split seed=42",
        fontsize=11,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(OUT_DIR, "aug4_cross_dataset.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot 6 → {out_path}")


# ── Plot 7 — Seed vs Aug (refnoise) across new data runs 1-20 ──────────────────

_SEED_MODEL_PATH = "models/snr0/random_3.0/simple_ls_val.pth"
_AUG_REFNOISE_PATH = "models/aug/refnoise/snr0/random_3.0/simple_ls_val.pth"


def plot_refnoise_runs(data_dir="data", aug_model_dir=AUG_MODEL_DIR,
                       out_path=None, device_str=None):
    """
    Plot 7 — Seed model vs Aug-refnoise model across new data run_0001..run_0020.
    Input: ls_snr+0_refnoise.npy.  Three panels: all / blocked / unblocked users.
    Lines: LS refnoise baseline, seed model, aug refnoise model.
    Each model is evaluated using its own training normalization params so the
    model receives correctly scaled inputs.
    Results are also saved to plots/plot7_results.csv.
    """
    try:
        import torch
        from Model import (ImprovedPhysicsInformedUNet, GlobalNormalizedDataset,
                           create_datasets, set_seed)
        from find_in_map import RSSMapProcessor
    except ImportError as exc:
        print(f"[plot7] SKIP — {exc}")
        return

    seed_path = _SEED_MODEL_PATH
    aug_path  = os.path.join(aug_model_dir, "refnoise", "snr0", "random_3.0",
                             "simple_ls_val.pth")
    for label, p in [("seed model", seed_path), ("aug model", aug_path)]:
        if not os.path.exists(p):
            print(f"[plot7] SKIP — {label} not found: {p}")
            return

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    def _load(path):
        m = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
        ck = torch.load(path, map_location=device)
        st = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        m.load_state_dict(st)
        return m.to(device).eval()

    rss_proc = RSSMapProcessor(
        image_path=RSS_IMAGE, bs_pixel_coords=BS_PIXEL,
        bs_real_coords=BS_REAL, image_width_meters=IMG_WIDTH_M,
    )

    # Pre-compute normalization params from each model's training run so the
    # model always sees correctly scaled inputs during evaluation.
    seed_train_dir = os.path.join("data", "run_0000")
    aug_train_dir  = os.path.join(data_dir, "run_0010")
    for label, d in [("seed training dir", seed_train_dir),
                     ("aug training dir",  aug_train_dir)]:
        ch  = os.path.join(d, "channels.npy")
        ls  = os.path.join(d, "ls_snr+0_refnoise.npy")
        pos = os.path.join(d, "locations_noisy.txt")
        for p in [ch, ls, pos]:
            if not os.path.exists(p):
                print(f"[plot7] SKIP — {label} missing: {p}")
                return

    set_seed(_SEED)
    _td, _, _, _, _, _ = create_datasets(
        smomp_file=os.path.join(seed_train_dir, "ls_snr+0_refnoise.npy"),
        accurate_file=os.path.join(seed_train_dir, "channels.npy"),
        user_positions_file=os.path.join(seed_train_dir, "locations_noisy.txt"),
        split_type="random", user_noise=_USER_NOISE, rss_processor=rss_proc,
    )
    seed_norm = _td.normalization_params
    del _td
    print(f"[plot7] seed norm  global_max={seed_norm['global_max']:.4e}")

    set_seed(_SEED)
    _td, _, _, _, _, _ = create_datasets(
        smomp_file=os.path.join(aug_train_dir, "ls_snr+0_refnoise.npy"),
        accurate_file=os.path.join(aug_train_dir, "channels.npy"),
        user_positions_file=os.path.join(aug_train_dir, "locations_noisy.txt"),
        split_type="random", user_noise=_USER_NOISE, rss_processor=rss_proc,
    )
    aug_norm = _td.normalization_params
    del _td
    print(f"[plot7] aug  norm  global_max={aug_norm['global_max']:.4e}")

    seed_model = _load(seed_path)
    aug_model  = _load(aug_path)
    print(f"[plot7] seed ← {seed_path}")
    print(f"[plot7] aug  ← {aug_path}\n")

    records = []

    for run_id in range(1, 21):
        run_dir   = os.path.join(data_dir, f"run_{run_id:04d}")
        ch_path   = os.path.join(run_dir, "channels.npy")
        ls_path   = os.path.join(run_dir, "ls_snr+0_refnoise.npy")
        pos_path  = os.path.join(run_dir, "locations_noisy.txt")
        mask_path = os.path.join(run_dir, "blocked_mask.npy")

        for p in [ch_path, ls_path, pos_path, mask_path]:
            if not os.path.exists(p):
                print(f"[plot7] run_{run_id:04d} SKIP — {p} missing")
                break
        else:
            print(f"[plot7] run_{run_id:04d} …")

            # Get split indices (normalization used here is discarded)
            set_seed(_SEED)
            _, _, _ref_ds, _, _, _ = create_datasets(
                smomp_file=ls_path,
                accurate_file=ch_path,
                user_positions_file=pos_path,
                split_type="random",
                user_noise=_USER_NOISE,
                rss_processor=rss_proc,
            )
            all_idx = _ref_ds.indices

            mask      = np.load(mask_path).astype(bool)
            blk_loc   = [k for k, ri in enumerate(all_idx) if mask[ri]]
            ublk_loc  = [k for k, ri in enumerate(all_idx) if not mask[ri]]
            blk_glob  = all_idx[np.array(blk_loc,  dtype=int)] if blk_loc  else np.array([], dtype=int)
            ublk_glob = all_idx[np.array(ublk_loc, dtype=int)] if ublk_loc else np.array([], dtype=int)

            ch_arr = np.load(ch_path, mmap_mode="r")
            ls_arr = np.load(ls_path, mmap_mode="r")

            # Build per-model test datasets with correct training normalization
            def _make_ds(norm_params):
                return GlobalNormalizedDataset(
                    smomp_file=ls_path,
                    accurate_file=ch_path,
                    user_positions_file=pos_path,
                    rss_processor=rss_proc,
                    normalization_params=norm_params,
                    indices=all_idx,
                    user_noise=_USER_NOISE,
                    split="test",
                )

            seed_ds = _make_ds(seed_norm)
            aug_ds  = _make_ds(aug_norm)

            subsets = {
                "all":       (list(range(len(all_idx))), all_idx),
                "blocked":   (blk_loc,                   blk_glob),
                "unblocked": (ublk_loc,                  ublk_glob),
            }

            row = {"run_id": run_id,
                   "n_blocked": len(blk_loc), "n_unblocked": len(ublk_loc)}
            for sname, (local, global_idx) in subsets.items():
                row[f"ls_{sname}"]   = _ls_nmse_subset(ls_arr, ch_arr, global_idx)
                row[f"seed_{sname}"] = _model_nmse_subset(seed_model, seed_ds, local, device)
                row[f"aug_{sname}"]  = _model_nmse_subset(aug_model,  aug_ds,  local, device)

            print(f"  blk={row['n_blocked']}  "
                  f"all → LS={row['ls_all']:+.2f} seed={row['seed_all']:+.2f} aug={row['aug_all']:+.2f}")
            records.append(row)

    del seed_model, aug_model

    if not records:
        print("[plot7] No results.")
        return

    df = pd.DataFrame(records).sort_values("run_id")

    # Save CSV
    csv_out = os.path.join("plots", "plot7_results.csv")
    os.makedirs("plots", exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[plot7] results → {csv_out}")

    run_ids = df["run_id"].values
    subsets = ["all", "blocked", "unblocked"]
    subset_titles = ["All test users", "Blocked users only", "Unblocked users only"]

    line_styles = {
        "ls":   dict(color="black",   linestyle="--", linewidth=1.5, marker="x", markersize=6),
        "seed": dict(color="#4878CF", linestyle="-",  linewidth=2.0, marker="o", markersize=6),
        "aug":  dict(color="#D65F5F", linestyle="-",  linewidth=2.0, marker="s", markersize=6),
    }
    line_labels = {
        "ls":   "LS refnoise baseline",
        "seed": "Seed model (trained on run_0000)",
        "aug":  "Aug refnoise model (trained on run_0010)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)

    for ax, sname, stitle in zip(axes, subsets, subset_titles):
        for key in ["ls", "seed", "aug"]:
            col = f"{key}_{sname}"
            if col in df.columns:
                ax.plot(run_ids, df[col].values, label=line_labels[key],
                        **line_styles[key])

        ax.set_xticks(run_ids)
        ax.set_xticklabels([str(r) for r in run_ids], fontsize=8)
        ax.set_xlabel("Run ID  (×2 trucks)", fontsize=10)
        ax.set_ylabel("NMSE (dB)", fontsize=10)
        ax.set_title(stitle, fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":", alpha=0.4)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(run_ids)
        ax2.set_xticklabels([str(r * 2) for r in run_ids], fontsize=7)
        ax2.set_xlabel("Trucks", fontsize=8)

    fig.suptitle(
        "Seed vs Aug-refnoise model — new data, runs 1-20 | input: ls_snr+0_refnoise\n"
        "(each model evaluated with its own training normalization)",
        fontsize=11,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join("plots", "refnoise_runs_1_20.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot 7 → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plots", nargs="+", type=int, default=[1, 2, 3],
                        help="Which plots to generate (1 2 3 4 5 6 7)")
    parser.add_argument("--run-dir",       default=RUN_0000)
    parser.add_argument("--out-dir",       default=OUT_DIR)
    parser.add_argument("--results-csv",   default=RESULTS_CSV)
    parser.add_argument("--aug-run-dir",   default=AUG_RUN_DIR,
                        help="Run directory for plot 4 (default: data/run_0010)")
    parser.add_argument("--aug-model-dir", default=AUG_MODEL_DIR,
                        help="Aug model directory for plot 4 (default: models/aug)")
    parser.add_argument("--device",        default=None,
                        help="Torch device for plot 4 (default: auto-detect cuda/cpu)")
    parser.add_argument("--data-dir",      default="data",
                        help="Root data directory for plot 5 (default: data)")
    args = parser.parse_args()

    OUT_DIR = args.out_dir

    if 1 in args.plots:
        plot_ue_locations(run_dir=args.run_dir)
    if 2 in args.plots:
        plot_ls_comparison(run_dir=args.run_dir)
    if 3 in args.plots:
        plot_results(csv_path=args.results_csv)
    if 4 in args.plots:
        plot_aug4_blocked(
            run_dir=args.aug_run_dir,
            aug_model_dir=args.aug_model_dir,
            device_str=args.device,
        )
    if 5 in args.plots:
        plot_attenuation_by_run(data_dir=args.data_dir)
    if 6 in args.plots:
        plot_aug4_cross_dataset(
            aug_model_dir=args.aug_model_dir,
            device_str=args.device,
        )
    if 7 in args.plots:
        plot_refnoise_runs(
            data_dir=args.data_dir,
            aug_model_dir=args.aug_model_dir,
            device_str=args.device,
        )


if __name__ == "__main__":
    main()
