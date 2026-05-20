#!/usr/bin/env python3
"""
scripts/experiment1.py
----------------------
Two-stage evaluation of the seed model across all generated runs.

Stage 1  --mode infer (or all):
  For every run_XXXX directory:
    - Auto-detect LS files present (ls_snr*.npy)
    - Run the seed model on ALL users with seed-training normalization
    - Save per-user (err_sq, ch_pw, atten_db) as
        data/run_XXXX/seed_predictions_{ls_tag}.csv

Stage 2  --mode plot (or all):
  Load saved CSVs across runs and produce:
    - Box plot: x = n_trucks, y = aggregate NMSE
      one box per run from N_DRAWS bootstrap draws of N_SUBSET users
      comparing seed model vs LS baseline, with ceiling line

Usage:
    python3 scripts/experiment1.py                          # infer + plot
    python3 scripts/experiment1.py --mode infer             # inference only
    python3 scripts/experiment1.py --mode plot              # plot only
    python3 scripts/experiment1.py --ls-tags snr+0_refnoise # specific LS
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from Model import ImprovedPhysicsInformedUNet, set_seed
from find_in_map import RSSMapProcessor

# ── defaults ──────────────────────────────────────────────────────────────────
RSS_IMAGE   = "Dataset/50_15GHz.jpg"
BS_PIXEL    = (287, 293)
BS_REAL     = (71.06, 246.29)
IMG_WIDTH_M = 527.5

SEED_MODEL_PATH  = "models/snr0/random_3.0/simple_ls_val.pth"
SEED_NORM_LS     = "ls_snr+0.npy"          # LS used during seed model training
SPLIT_SEED       = 42
TRAIN_RATIO      = 0.8
BATCH_SIZE       = 64

N_SUBSET   = 300    # users per bootstrap draw
N_DRAWS    = 30     # draws per run for box plot
PLOT_SEED  = 7


# ── model helpers ─────────────────────────────────────────────────────────────

def load_model(path, device):
    m = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    ck = torch.load(path, map_location=device)
    st = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    m.load_state_dict(st)
    return m.to(device).eval()


def _seed_split(data_dir, norm_ls_tag, seed):
    """Reproduce the exact train/test split used during seed model training.
    Returns (global_max, test_idx) — same permutation as create_datasets(split_type='random').
    """
    set_seed(seed)
    run0  = os.path.join(data_dir, "run_0000")
    ls    = np.load(os.path.join(run0, norm_ls_tag), mmap_mode="r")
    ch    = np.load(os.path.join(run0, "channels.npy"), mmap_mode="r")
    n     = len(ls)
    perm  = np.random.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * 0.1)
    train   = perm[:n_train]
    test    = perm[n_train + n_val:]
    gmax    = float(max(np.max(np.abs(ls[train])), np.max(np.abs(ch[train]))))
    return gmax, test


# ── per-user inference (no Dataset class needed) ──────────────────────────────

def infer_all_users(model, ls_arr, ch_arr, rss_cache, global_max, device,
                    batch_size=BATCH_SIZE):
    """
    Run model on every user.
    Returns err_sq (N,), ch_pw (N,)  — unnormalised squared sums.
    """
    n = len(ls_arr)
    err_sq = np.empty(n, dtype=np.float64)
    ch_pw  = np.empty(n, dtype=np.float64)

    scale  = float(global_max)
    ls_t   = torch.from_numpy(ls_arr.astype(np.float32))  / scale
    ch_t   = torch.from_numpy(ch_arr.astype(np.float32))  / scale
    rss_t  = torch.from_numpy(rss_cache.astype(np.float32))

    ds     = TensorDataset(ls_t, ch_t, rss_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    start = 0
    with torch.no_grad():
        for smomp, accurate, rss in loader:
            pred = model(smomp.to(device), rss.to(device))
            accurate = accurate.to(device)
            e = torch.sum((pred - accurate) ** 2, dim=(1, 2, 3)).cpu().numpy()
            p = torch.sum( accurate          ** 2, dim=(1, 2, 3)).cpu().numpy()
            # rescale back to raw units
            err_sq[start:start + len(e)] = e.astype(np.float64) * scale ** 2
            ch_pw [start:start + len(p)] = p.astype(np.float64) * scale ** 2
            start += len(e)

    return err_sq, ch_pw


def ls_err_pw(ls_arr, ch_arr):
    ls = ls_arr.astype(np.float64)
    ch = ch_arr.astype(np.float64)
    err = np.sum((ls - ch) ** 2, axis=(1, 2, 3))
    pw  = np.sum( ch       ** 2, axis=(1, 2, 3))
    return err, pw


def atten_vs_run0(ch0_arr, ch_arr):
    pw0 = np.sum(ch0_arr.astype(np.float64) ** 2, axis=(1, 2, 3))
    pw  = np.sum(ch_arr .astype(np.float64) ** 2, axis=(1, 2, 3))
    return 10.0 * np.log10((pw0 + 1e-300) / (pw + 1e-300))


# ── scan helpers ──────────────────────────────────────────────────────────────

def detect_ls_tags(run_dir):
    """Return list of ls tags present, e.g. ['snr+0', 'snr+0_refnoise']."""
    tags = []
    for f in sorted(os.listdir(run_dir)):
        if f.startswith("ls_") and f.endswith(".npy"):
            tags.append(f[len("ls_"):-len(".npy")])
    return tags


def run_meta(run_dir):
    """Return (run_id, n_trucks) from blocked_summary.json."""
    jsp = os.path.join(run_dir, "blocked_summary.json")
    if os.path.exists(jsp):
        with open(jsp) as f:
            d = json.load(f)
        return int(d.get("run_id", -1)), int(d.get("n_trucks", -1))
    # fallback: parse folder name
    name = os.path.basename(run_dir)
    try:
        idx = int(name.split("_")[1])
    except Exception:
        idx = -1
    return idx, -1


# ── stage 1: inference ────────────────────────────────────────────────────────

def stage_infer(args):
    device     = torch.device(args.device)
    global_max, _ = _seed_split(args.data_dir, args.seed_norm_ls, SPLIT_SEED)
    print(f"Seed global_max = {global_max:.4e}  (from run_0000/{args.seed_norm_ls})")

    model = load_model(args.seed_model, device)
    print(f"Loaded model: {args.seed_model}\n")

    ch0_path = os.path.join(args.data_dir, "run_0000", "channels.npy")
    ch0_arr  = np.load(ch0_path, mmap_mode="r")

    rss_proc = RSSMapProcessor(RSS_IMAGE, BS_PIXEL, BS_REAL, IMG_WIDTH_M)

    run_dirs = sorted(glob.glob(os.path.join(args.data_dir, "run_*")))
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue

        ch_path    = os.path.join(run_dir, "channels.npy")
        cache_path = os.path.join(run_dir, "rss_cache.npy")

        if not os.path.exists(ch_path):
            print(f"[{os.path.basename(run_dir)}] skip — no channels.npy")
            continue
        if not os.path.exists(cache_path):
            print(f"[{os.path.basename(run_dir)}] skip — no rss_cache.npy  (run generate_rss.py first)")
            continue

        ls_tags = detect_ls_tags(run_dir)
        if args.ls_tags:
            ls_tags = [t for t in ls_tags if t in args.ls_tags]
        if not ls_tags:
            print(f"[{os.path.basename(run_dir)}] skip — no matching LS files")
            continue

        _, n_trucks = run_meta(run_dir)
        print(f"[{os.path.basename(run_dir)}]  {n_trucks} trucks  LS tags: {ls_tags}")

        ch_arr    = np.load(ch_path,    mmap_mode="r")
        rss_cache = np.load(cache_path, mmap_mode="r")
        atten_db  = atten_vs_run0(ch0_arr, ch_arr)

        for tag in ls_tags:
            out_csv = os.path.join(run_dir, f"seed_predictions_{tag}.csv")
            if os.path.exists(out_csv) and not args.overwrite:
                print(f"  {tag}: exists, skip")
                continue

            ls_path = os.path.join(run_dir, f"ls_{tag}.npy")
            ls_arr  = np.load(ls_path, mmap_mode="r")

            print(f"  {tag}: running inference on {len(ls_arr)} users …", end="", flush=True)
            seed_err, seed_pw = infer_all_users(
                model, ls_arr, ch_arr, rss_cache, global_max, device)
            ls_err, ls_pw = ls_err_pw(ls_arr, ch_arr)

            safe_pw = np.maximum(seed_pw, 1e-300)
            df = pd.DataFrame({
                "user_idx":     np.arange(len(ls_arr)),
                "seed_err_sq":  seed_err,
                "ch_pw":        seed_pw,
                "ls_err_sq":    ls_err,
                "atten_db":     atten_db,
                "seed_nmse_db": 10.0 * np.log10(seed_err / safe_pw),
                "ls_nmse_db":   10.0 * np.log10(ls_err   / safe_pw),
            })
            df.to_csv(out_csv, index=False)
            print(f"  saved ({len(df)} rows)")

        print()


# ── stage 2: plot ─────────────────────────────────────────────────────────────

def agg_nmse(err, pw, idx=None):
    if idx is not None:
        err, pw = err[idx], pw[idx]
    return float(10.0 * np.log10(err.sum() / max(float(pw.sum()), 1e-300)))


def stage_plot(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ── reproduce seed model's test split ────────────────────────────────────
    _, test_idx = _seed_split(args.data_dir, args.seed_norm_ls, SPLIT_SEED)
    test_set = set(test_idx.tolist())
    print(f"Using {len(test_set)} test users (seed model held-out split)\n")

    # ── load all CSVs, filtered to test users ────────────────────────────────
    run_dirs = sorted(glob.glob(os.path.join(args.data_dir, "run_*")))

    # collect data per (n_trucks, ls_tag)
    records = {}   # n_trucks → {tag → DataFrame}
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
                print(f"  warn: {csv_f} missing seed_nmse_db — re-run --mode infer --overwrite")
                continue
            df = df[df["user_idx"].isin(test_set)].reset_index(drop=True)
            records.setdefault(n_trucks, {})[tag] = df

    if not records:
        print("No seed_predictions CSVs found — run --mode infer first.")
        return

    # ── seed-on-clean ceiling (test users only) ───────────────────────────────
    ceiling_db = None
    if 0 in records:
        first_tag = next(iter(records[0]))
        df0_test   = records[0][first_tag]
        ceiling_db = agg_nmse(df0_test["seed_err_sq"].values, df0_test["ch_pw"].values)
        print(f"Seed-on-clean ceiling = {ceiling_db:+.2f} dB  (run_0000, {len(df0_test)} test users)")

    # ── pick tags to plot ─────────────────────────────────────────────────────
    all_tags = sorted({t for v in records.values() for t in v.keys()})
    plot_tags = args.ls_tags if args.ls_tags else all_tags
    print(f"Plotting LS tags: {plot_tags}")
    print(f"Runs (n_trucks): {sorted(records.keys())}\n")

    # ── load run_0000 baseline NMSE per user ─────────────────────────────────
    base_nmse = {}   # tag → Series indexed by user_idx
    for tag in plot_tags:
        key = 0   # n_trucks=0 is run_0000
        if key in records and tag in records[key]:
            base_nmse[tag] = records[key][tag].set_index("user_idx")["seed_nmse_db"]

    for tag in plot_tags:
        truck_vals = sorted(k for k in records if tag in records[k] and k > 0)
        if not truck_vals:
            print(f"  tag={tag}: no blocked runs found, skipping plot")
            continue

        delta_boxes = []
        ls_delta_boxes = []

        for n_trucks in truck_vals:
            df = records[n_trucks][tag]
            if tag in base_nmse:
                merged = df.set_index("user_idx").join(
                    base_nmse[tag].rename("base_nmse_db"), how="inner")
                delta_seed = (merged["seed_nmse_db"] - merged["base_nmse_db"]).values
                delta_ls   = (merged["ls_nmse_db"]   - merged["base_nmse_db"]).values
            else:
                delta_seed = df["seed_nmse_db"].values
                delta_ls   = df["ls_nmse_db"].values
            delta_boxes.append(delta_seed)
            ls_delta_boxes.append(delta_ls)

        n_pts = len(truck_vals)
        x     = np.arange(n_pts)
        w     = 0.35

        clr_seed = "#4878CF"
        clr_ls   = "#888888"

        truck_counts = np.array(truck_vals, dtype=float)
        norm_c  = (truck_counts - truck_counts.min()) / (truck_counts.max() - truck_counts.min() + 1e-9)
        colors  = plt.cm.RdYlGn_r(0.15 + 0.7 * norm_c)

        fig, ax = plt.subplots(figsize=(max(10, n_pts * 1.4), 6))

        def _boxplot(data_list, positions, color_list, label):
            bp = ax.boxplot(
                data_list, positions=positions, widths=w,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.25,
                                markerfacecolor="gray", markeredgecolor="gray"),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color="gray", linewidth=1.0),
                capprops=dict(color="gray", linewidth=1.0),
            )
            for patch, c in zip(bp["boxes"], color_list):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)

        _boxplot(delta_boxes, x, colors, "Seed model NMSE delta")

        ls_medians = [float(np.median(d)) for d in ls_delta_boxes]
        ax.plot(x, ls_medians, "^--", color=clr_ls,
                linewidth=1.5, markersize=7, label="LS delta (median)")

        ax.axhline(0, color="steelblue", linewidth=1.5, linestyle="--",
                   label="delta = 0  (same as run_0000)", zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}\ntrucks" for v in truck_vals], fontsize=9)
        ax.set_xlabel("Number of trucks", fontsize=11)
        ax.set_ylabel("Per-user NMSE delta (dB)  [run_k − run_0000]", fontsize=11)
        ax.set_title(
            f"Seed model NMSE degradation vs blockage — LS tag: {tag}\n"
            f"Each box = all users in that run  |  delta > 0 = worse than clean",
            fontsize=10)
        ax.legend(handles=[
            mpatches.Patch(color="tomato", alpha=0.75, label="Seed model NMSE delta"),
            mlines.Line2D([0], [0], marker="^", color=clr_ls, linestyle="--",
                          markersize=7, label="LS delta (median)"),
            mlines.Line2D([0], [0], color="steelblue", linewidth=1.5,
                          linestyle="--", label="delta = 0  (run_0000 baseline)"),
        ], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(args.out_dir, f"experiment1_{tag}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Plot → {out_path}")

    # ── combined CSV summary ───────────────────────────────────────────────────
    rows = []
    for n_trucks, tag_dfs in records.items():
        for tag, df in tag_dfs.items():
            se, cp, le = df["seed_err_sq"].values, df["ch_pw"].values, df["ls_err_sq"].values
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
    print(f"Summary CSV → {csv_out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode",         choices=["infer", "plot", "all"], default="all")
    parser.add_argument("--data-dir",     default="data")
    parser.add_argument("--out-dir",      default="plots")
    parser.add_argument("--seed-model",   default=SEED_MODEL_PATH)
    parser.add_argument("--seed-norm-ls", default=SEED_NORM_LS,
                        help="LS filename in run_0000 used for seed model training normalization")
    parser.add_argument("--ls-tags",      nargs="*", default=None,
                        help="LS tags to process, e.g. snr+0 snr+0_refnoise. "
                             "Default: all detected tags.")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Overwrite existing seed_predictions CSVs.")
    args = parser.parse_args()

    if args.mode in ("infer", "all"):
        stage_infer(args)
    if args.mode in ("plot", "all"):
        stage_plot(args)


if __name__ == "__main__":
    main()
