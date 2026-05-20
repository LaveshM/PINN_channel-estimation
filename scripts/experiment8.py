#!/usr/bin/env python3
"""
scripts/experiment8.py
----------------------
Generalization stress-test: per-user NMSE vs blockage attenuation.

Evaluates seed and aug-refnoise models on a run from data/ that has
a range of per-user attenuations (default: run_0010, 20 trucks).

For each test user we compute their individual NMSE and their attenuation
relative to run_0000.  Results are binned by attenuation severity and plotted
as a grouped box chart with three reference lines:
  - Seed-on-clean ceiling: seed model NMSE on run_0000 test split (green dashed)
  - Per-bin LS NMSE: raw LS estimate baseline per bin (black marker)

Expected outcome:
  - Seed model: NMSE worsens with attenuation (clean-data baseline is ceiling)
  - Aug model: NMSE stays somewhat flatter but still cannot match the clean ceiling
  - Both models degrade vs the seed-on-clean reference as blockage increases

Usage:
    python3 scripts/experiment8.py
    python3 scripts/experiment8.py --run-id 10 --data-dir data
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Model import (ImprovedPhysicsInformedUNet, GlobalNormalizedDataset,
                   create_datasets, set_seed)
from find_in_map import RSSMapProcessor

# ── constants ──────────────────────────────────────────────────────────────────
RSS_IMAGE   = "Dataset/50_15GHz.jpg"
BS_PIXEL    = (287, 293)
BS_REAL     = (71.06, 246.29)
IMG_WIDTH_M = 527.5
USER_NOISE  = 3.0
SEED        = 42
BATCH_SIZE  = 32

# Models always live in the same paths regardless of which run is evaluated
SEED_MODEL_PATH = "models/snr0/random_3.0/simple_ls_val.pth"
AUG_MODEL_PATH  = "models/aug/refnoise/snr0/random_3.0/simple_ls_val.pth"

# Normalization always comes from training data (new data/ directory)
NORM_DATA_DIR = "data"

# Attenuation bins (dB): [left, right)
ATTEN_BINS   = [0, 0.1, 2, 5, 10, 20, 999]
ATTEN_LABELS = ["0–0.1", "0.1–2", "2–5", "5–10", "10–20", ">20"]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_model(path, device):
    m = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    ck = torch.load(path, map_location=device)
    st = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    m.load_state_dict(st)
    return m.to(device).eval()


def get_norm_params(ls_file, ch_file, pos_file, rss_proc):
    set_seed(SEED)
    td, _, _, _, _, _ = create_datasets(
        smomp_file=ls_file, accurate_file=ch_file,
        user_positions_file=pos_file,
        split_type="random", user_noise=USER_NOISE,
        rss_processor=rss_proc,
    )
    return td.normalization_params


def per_user_err_pw(model, ds, device, batch_size=BATCH_SIZE):
    """
    Returns (err_arr, pw_arr): per-user total squared error and channel power.
    Use for aggregate NMSE per bin  and  per-user NMSE scatter.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    err_list, pw_list = [], []
    with torch.no_grad():
        for smomp, accurate, rss in loader:
            pred     = model(smomp.to(device), rss.to(device))
            accurate = accurate.to(device)
            err_list.append(torch.sum((pred - accurate) ** 2, dim=(1, 2, 3)).cpu().numpy())
            pw_list.append( torch.sum( accurate          ** 2, dim=(1, 2, 3)).cpu().numpy())
    return np.concatenate(err_list), np.concatenate(pw_list)


def agg_nmse(err_arr, pw_arr, mask=None):
    """Global-sum NMSE in dB for (optionally masked) users."""
    if mask is not None:
        err_arr, pw_arr = err_arr[mask], pw_arr[mask]
    return float(10.0 * np.log10(err_arr.sum() / max(float(pw_arr.sum()), 1e-300)))



def pwr_chunked(arr, idx, chunk=300):
    out = np.empty(len(idx), dtype=np.float64)
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        out[s:s + chunk] = np.sum(arr[sl].astype(np.float64) ** 2, axis=(1, 2, 3))
    return out


def ls_err_pw_chunked(ls_arr, ch_arr, idx, chunk=300):
    """Returns (err_arr, pw_arr) for LS estimate vs true channel."""
    err_out = np.empty(len(idx), dtype=np.float64)
    pw_out  = np.empty(len(idx), dtype=np.float64)
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        e  = ls_arr[sl].astype(np.float64)
        t  = ch_arr[sl].astype(np.float64)
        err_out[s:s + chunk] = np.sum((e - t) ** 2, axis=(1, 2, 3))
        pw_out[s:s + chunk]  = np.sum( t       ** 2, axis=(1, 2, 3))
    return err_out, pw_out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir",  default="data",
                        help="Directory containing eval run (default: data)")
    parser.add_argument("--run-id",    type=int, default=10,
                        help="Which run_XXXX to evaluate on (default: 10)")
    parser.add_argument("--out-dir",   default="plots")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    run_dir   = os.path.join(args.data_dir, f"run_{args.run_id:04d}")
    ch_path   = os.path.join(run_dir, "channels.npy")
    ls_path   = os.path.join(run_dir, "ls_snr+0_refnoise.npy")
    pos_path  = os.path.join(run_dir, "locations_noisy.txt")
    ch0_path  = os.path.join(args.data_dir, "run_0000", "channels.npy")

    for p in [ch_path, ls_path, pos_path, ch0_path]:
        if not os.path.exists(p):
            print(f"SKIP — missing: {p}")
            return

    rss_proc = RSSMapProcessor(
        image_path=RSS_IMAGE, bs_pixel_coords=BS_PIXEL,
        bs_real_coords=BS_REAL, image_width_meters=IMG_WIDTH_M,
    )

    # ── normalization: always from training data (data/) ──────────────────────
    seed_train_dir = os.path.join(NORM_DATA_DIR, "run_0000")
    aug_train_dir  = os.path.join(NORM_DATA_DIR, "run_0010")

    print(f"Seed norm ← {seed_train_dir} (adaptive LS)")
    seed_norm = get_norm_params(
        os.path.join(seed_train_dir, "ls_snr+0.npy"),
        os.path.join(seed_train_dir, "channels.npy"),
        os.path.join(seed_train_dir, "locations_noisy.txt"),
        rss_proc,
    )
    print(f"  global_max = {seed_norm['global_max']:.4e}")

    print(f"Aug  norm ← {aug_train_dir} (refnoise LS)")
    aug_norm = get_norm_params(
        os.path.join(aug_train_dir, "ls_snr+0_refnoise.npy"),
        os.path.join(aug_train_dir, "channels.npy"),
        os.path.join(aug_train_dir, "locations_noisy.txt"),
        rss_proc,
    )
    print(f"  global_max = {aug_norm['global_max']:.4e}\n")

    # ── test split (deterministic) ────────────────────────────────────────────
    n_samples = np.load(ch_path, mmap_mode="r").shape[0]
    set_seed(SEED)
    perm     = np.random.permutation(n_samples)
    n_train  = int(n_samples * 0.8)
    n_val    = int(n_samples * 0.1)
    test_idx = perm[n_train + n_val:]
    print(f"Test split: {len(test_idx)} users\n")

    # ── load models ───────────────────────────────────────────────────────────
    seed_model = load_model(SEED_MODEL_PATH, device)
    aug_model  = load_model(AUG_MODEL_PATH,  device)
    print(f"seed ← {SEED_MODEL_PATH}")
    print(f"aug  ← {AUG_MODEL_PATH}\n")

    # ── seed-on-clean ceiling: seed model on run_0000 test split ─────────────
    print("Computing seed-on-clean ceiling (run_0000 test split) …")
    clean_ch_path  = os.path.join(NORM_DATA_DIR, "run_0000", "channels.npy")
    clean_ls_path  = os.path.join(NORM_DATA_DIR, "run_0000", "ls_snr+0_refnoise.npy")
    clean_pos_path = os.path.join(NORM_DATA_DIR, "run_0000", "locations_noisy.txt")
    # Use same n_samples as eval run for consistent split (both should match)
    clean_ds = GlobalNormalizedDataset(
        smomp_file=clean_ls_path, accurate_file=clean_ch_path,
        user_positions_file=clean_pos_path,
        rss_processor=rss_proc,
        normalization_params=seed_norm,
        indices=test_idx,
        user_noise=USER_NOISE,
        split="test",
    )
    ceil_err, ceil_pw = per_user_err_pw(seed_model, clean_ds, device)
    seed_ceiling_nmse = agg_nmse(ceil_err, ceil_pw)
    print(f"  Seed-on-clean ceiling = {seed_ceiling_nmse:+.2f} dB\n")

    # ── per-user attenuation vs run_0000 ─────────────────────────────────────
    ch0 = np.load(ch0_path, mmap_mode="r")
    ch  = np.load(ch_path,  mmap_mode="r")
    ls  = np.load(ls_path,  mmap_mode="r")

    pwr0_test = pwr_chunked(ch0, test_idx)
    pwr_test  = pwr_chunked(ch,  test_idx)
    atten_db  = 10.0 * np.log10((pwr0_test + 1e-300) / (pwr_test + 1e-300))
    print(f"run_{args.run_id:04d} test users — attenuation vs run_0000:")
    print(f"  mean={atten_db.mean():.2f} dB  median={np.median(atten_db):.2f} dB  "
          f"min={atten_db.min():.2f} dB  max={atten_db.max():.2f} dB\n")

    # ── per-user LS err/pw ────────────────────────────────────────────────────
    print("Computing per-user LS errors …")
    ls_err_arr, ls_pw_arr = ls_err_pw_chunked(ls, ch, test_idx)

    # ── run model inference ───────────────────────────────────────────────────
    def make_ds(norm):
        return GlobalNormalizedDataset(
            smomp_file=ls_path, accurate_file=ch_path,
            user_positions_file=pos_path,
            rss_processor=rss_proc,
            normalization_params=norm,
            indices=test_idx,
            user_noise=USER_NOISE,
            split="test",
        )

    print("Running inference — seed model …")
    seed_err, seed_pw = per_user_err_pw(seed_model, make_ds(seed_norm), device)
    print("Running inference — aug  model …")
    aug_err, aug_pw   = per_user_err_pw(aug_model,  make_ds(aug_norm),  device)

    # Per-user NMSE for scatter plots
    seed_nmse_per_user = 10.0 * np.log10(seed_err / seed_pw.clip(1e-30))
    aug_nmse_per_user  = 10.0 * np.log10(aug_err  / aug_pw.clip(1e-30))
    ls_nmse_per_user   = 10.0 * np.log10(ls_err_arr / ls_pw_arr.clip(1e-30))

    # ── bin by attenuation ────────────────────────────────────────────────────
    bins, labels, n_bins = ATTEN_BINS, ATTEN_LABELS, len(ATTEN_LABELS)
    bin_masks = [(atten_db >= bins[b]) & (atten_db < bins[b + 1])
                 for b in range(n_bins)]
    bin_counts = [int(m.sum()) for m in bin_masks]

    # Aggregate (global-sum) NMSE per bin — robust to per-user power variation
    seed_agg = [agg_nmse(seed_err, seed_pw, m) if bin_counts[b] else float("nan")
                for b, m in enumerate(bin_masks)]
    aug_agg  = [agg_nmse(aug_err,  aug_pw,  m) if bin_counts[b] else float("nan")
                for b, m in enumerate(bin_masks)]
    ls_agg   = [agg_nmse(ls_err_arr, ls_pw_arr, m) if bin_counts[b] else float("nan")
                for b, m in enumerate(bin_masks)]

    print("\nBin        N    LS_agg   seed_agg  aug_agg   ceiling")
    for b in range(n_bins):
        if bin_counts[b] == 0:
            continue
        print(f"  {labels[b]:>6} dB  {bin_counts[b]:4d}   {ls_agg[b]:+7.2f}"
              f"   {seed_agg[b]:+7.2f}   {aug_agg[b]:+7.2f}   {seed_ceiling_nmse:+7.2f}")
    overall_seed = agg_nmse(seed_err, seed_pw)
    overall_aug  = agg_nmse(aug_err,  aug_pw)
    overall_ls   = agg_nmse(ls_err_arr, ls_pw_arr)
    print(f"\n  {'Overall':>8}  {len(test_idx):4d}   {overall_ls:+7.2f}"
          f"   {overall_seed:+7.2f}   {overall_aug:+7.2f}   {seed_ceiling_nmse:+7.2f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    clr_seed    = "#4878CF"
    clr_aug     = "#D65F5F"
    clr_ls      = "#888888"
    clr_ceiling = "#2CA02C"

    # ── Left: grouped bar chart — aggregate NMSE per bin ─────────────────────
    ax   = axes[0]
    x    = np.arange(n_bins)
    w    = 0.25
    valid = [b for b in range(n_bins) if bin_counts[b] > 0]

    ax.bar(x[valid] - w, [ls_agg[b]   for b in valid], width=w,
           color=clr_ls,   alpha=0.85, label="LS refnoise")
    ax.bar(x[valid],      [seed_agg[b] for b in valid], width=w,
           color=clr_seed, alpha=0.85, label="Seed model")
    ax.bar(x[valid] + w,  [aug_agg[b]  for b in valid], width=w,
           color=clr_aug,  alpha=0.85, label="Aug-refnoise model")

    ax.axhline(seed_ceiling_nmse, color=clr_ceiling, linewidth=2.0,
               linestyle="--", zorder=5,
               label=f"Seed-on-clean ceiling ({seed_ceiling_nmse:+.1f} dB)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels[b]} dB\n(n={bin_counts[b]})" for b in range(n_bins)],
                       fontsize=8)
    ax.set_xlabel("Per-user attenuation vs run_0000 (blockage severity)", fontsize=10)
    ax.set_ylabel("Aggregate NMSE (dB)", fontsize=10)
    ax.set_title(f"Aggregate NMSE per blockage bin — run_{args.run_id:04d}\n"
                 f"Green dashed = seed model on CLEAN data (no blockers)",
                 fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    # ── Right: scatter + per-bin aggregate trend ─────────────────────────────
    ax2 = axes[1]
    cap = 30
    s_plot = np.clip(seed_nmse_per_user, -30, cap)
    g_plot = np.clip(aug_nmse_per_user,  -30, cap)
    l_plot = np.clip(ls_nmse_per_user,   -30, cap)

    ax2.scatter(atten_db, s_plot, s=4, alpha=0.2, color=clr_seed)
    ax2.scatter(atten_db, g_plot, s=4, alpha=0.2, color=clr_aug)
    ax2.scatter(atten_db, l_plot, s=4, alpha=0.1, color=clr_ls)

    # Trend from per-bin aggregate NMSE (more meaningful than running per-user median)
    bin_ctrs = [np.mean(atten_db[bin_masks[b]]) if bin_counts[b] else float("nan")
                for b in range(n_bins)]
    v = [(bin_ctrs[b], seed_agg[b], aug_agg[b], ls_agg[b])
         for b in valid if not np.isnan(bin_ctrs[b])]
    if v:
        bc, sa_v, ag_v, ls_v = zip(*v)
        ax2.plot(bc, sa_v, "o-", color=clr_seed, linewidth=2.5, markersize=8,
                 label="Seed (bin aggregate)")
        ax2.plot(bc, ag_v, "s-", color=clr_aug,  linewidth=2.5, markersize=8,
                 label="Aug-refnoise (bin aggregate)")
        ax2.plot(bc, ls_v, "^--", color=clr_ls,  linewidth=1.5, markersize=6,
                 label="LS refnoise (bin aggregate)")

    ax2.axhline(seed_ceiling_nmse, color=clr_ceiling, linewidth=2.0, linestyle="--",
                label=f"Seed-on-clean ceiling ({seed_ceiling_nmse:+.1f} dB)", zorder=5)
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)
    ax2.set_xlabel("Per-user attenuation vs run_0000 (dB)", fontsize=10)
    ax2.set_ylabel("Per-user NMSE (dB, capped ±30)", fontsize=10)
    ax2.set_title(f"Per-user scatter + bin-aggregate trend — run_{args.run_id:04d}\n"
                  f"Lines = aggregate NMSE per bin  |  Green = clean-data ceiling",
                  fontsize=10)
    ax2.legend(fontsize=9, markerscale=2.5, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Generalization: seed vs aug-refnoise on run_{args.run_id:04d} ({args.data_dir})\n"
        f"Input: ls_snr+0_refnoise  |  Each model uses its training normalization  |  "
        f"Green dashed = seed model on clean data (no blockers)",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = os.path.join(args.out_dir, f"experiment8_run{args.run_id:04d}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot → {out_path}")

    # ── CSV: per-bin aggregate ────────────────────────────────────────────────
    import pandas as pd
    df_bin = pd.DataFrame({
        "bin_label":    labels,
        "n_users":      bin_counts,
        "ls_nmse_db":   ls_agg,
        "seed_nmse_db": seed_agg,
        "aug_nmse_db":  aug_agg,
        "ceiling_db":   [seed_ceiling_nmse] * n_bins,
    })
    csv_path = os.path.join(args.out_dir, f"experiment8_run{args.run_id:04d}.csv")
    df_bin.to_csv(csv_path, index=False)
    print(f"CSV  → {csv_path}")
    print(f"Seed-on-clean ceiling: {seed_ceiling_nmse:+.2f} dB")


if __name__ == "__main__":
    main()
