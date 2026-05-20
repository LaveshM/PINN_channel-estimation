#!/usr/bin/env python3
"""
scripts/experiment2.py
----------------------
Evaluate a pre-trained model (random split, SNR=0, noise=3.0 m) across all
run_XXXX blockage directories.

For each run the model is fed each of the three LS inputs separately:
  adaptive  → ls_snr+0.npy
  fixed     → ls_snr+0_fixed.npy
  refnoise  → ls_snr+0_refnoise.npy

With --blocked-only: also evaluates on the subset of test users that are
blocked (blocked_mask.npy == True) and generates a separate plot for them.

Results → models/old/eval_results.csv
Plots   → models/old/eval_results.png  [+ eval_results_blocked.png]

Usage (from repo root or from scripts/):
    python3 scripts/experiment2.py --model models/snr0/random_3.0/simple_ls_val.pth
    python3 scripts/experiment2.py --model ... --blocked-only
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
from torch.utils.data import DataLoader, Subset
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

LS_MODES = {
    "adaptive": "ls_snr+0.npy",
    "fixed":    "ls_snr+0_fixed.npy",
    "refnoise": "ls_snr+0_refnoise.npy",
}

MODE_STYLES = {
    "adaptive": dict(color="#4878CF", marker="o", linestyle="-"),
    "fixed":    dict(color="#D65F5F", marker="s", linestyle="--"),
    "refnoise": dict(color="#6ACC65", marker="^", linestyle=":"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def nmse_db(est, ch, idx):
    """Global-sum NMSE in dB."""
    e = est[idx].astype(np.float32)
    t = ch[idx].astype(np.float32)
    return float(10.0 * np.log10(np.sum((e - t) ** 2) / np.sum(t ** 2)))


def model_nmse_db(model, loader, device):
    """Global-sum NMSE in dB (data already normalised by dataset)."""
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


def blocked_subset(test_ds, blocked_mask):
    """
    Return (local_positions, real_indices) for test samples that are blocked.
    local_positions: indices into test_ds (for Subset)
    real_indices: indices into the full array (for direct numpy NMSE)
    """
    blocked_set = set(np.where(blocked_mask)[0])
    local = [i for i, ri in enumerate(test_ds.indices) if ri in blocked_set]
    real  = test_ds.indices[np.array(local)] if local else np.array([], dtype=int)
    return local, real


def save_plot(df, out_path, title, ls_col="ls_nmse_test", model_col="model_nmse_test"):
    run_ids = sorted(df["run_id"].unique())
    modes   = [m for m in LS_MODES if m in df["ls_mode"].values]

    fig, ax = plt.subplots(figsize=(10, 5))
    for mode in modes:
        sub = df[df["ls_mode"] == mode].sort_values("run_id")
        sub = sub.dropna(subset=[ls_col, model_col])
        if sub.empty:
            continue
        s = MODE_STYLES[mode]
        ax.plot(sub["run_id"], sub[ls_col],
                label=f"LS {mode}", linewidth=1.5, alpha=0.6,
                color=s["color"], marker=s["marker"], linestyle="--")
        ax.plot(sub["run_id"], sub[model_col],
                label=f"Model {mode}", linewidth=2,
                color=s["color"], marker=s["marker"], linestyle=s["linestyle"])

    ax.set_xlabel("Run ID  (trucks = run_id × 2)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(title)
    ax.set_xticks(run_ids)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot    → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model",        required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir",     default="data")
    parser.add_argument("--out-dir",      default="models/old")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--blocked-only", action="store_true",
                        help="Also evaluate on blocked users only and save a separate plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    model = load_model(args.model, device)
    print(f"Loaded  : {args.model}  →  {device}")
    print(f"Blocked-only eval: {args.blocked_only}\n")

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
        run_id   = int(os.path.basename(run_dir).split("_")[1])
        ch_path  = os.path.join(run_dir, "channels.npy")
        pos_path = os.path.join(run_dir, "locations_noisy.txt")
        mask_path= os.path.join(run_dir, "blocked_mask.npy")

        if not os.path.exists(ch_path) or not os.path.exists(pos_path):
            print(f"[run {run_id:04d}] SKIP — channels.npy or locations_noisy.txt missing")
            continue

        ch           = np.load(ch_path,   mmap_mode="r")
        blocked_mask = np.load(mask_path) if os.path.exists(mask_path) else None
        print(f"[run {run_id:04d}] {'─'*52}")

        for mode, ls_fname in LS_MODES.items():
            ls_path = os.path.join(run_dir, ls_fname)
            if not os.path.exists(ls_path):
                print(f"  {mode:10s} SKIP — {ls_fname} missing")
                continue

            set_seed(SEED)
            train_ds, _, test_ds, _, _, _ = create_datasets(
                smomp_file=ls_path,
                accurate_file=ch_path,
                user_positions_file=pos_path,
                split_type=SPLIT_TYPE,
                user_noise=USER_NOISE,
                rss_processor=rss_processor,
            )

            ls    = np.load(ls_path, mmap_mode="r")
            ls_tr = nmse_db(ls, ch, train_ds.indices)
            ls_te = nmse_db(ls, ch, test_ds.indices)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=2, pin_memory=True)
            test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=2, pin_memory=True)

            mdl_tr = model_nmse_db(model, train_loader, device)
            mdl_te = model_nmse_db(model, test_loader,  device)

            row = {
                "run_id":           run_id,
                "ls_mode":          mode,
                "ls_nmse_train":    round(ls_tr,  4),
                "ls_nmse_test":     round(ls_te,  4),
                "model_nmse_train": round(mdl_tr, 4),
                "model_nmse_test":  round(mdl_te, 4),
                "ls_nmse_blocked":      None,
                "model_nmse_blocked":   None,
            }

            print(f"  {mode:10s}  LS  train:{ls_tr:+6.2f} dB  test:{ls_te:+6.2f} dB  |"
                  f"  model  train:{mdl_tr:+6.2f} dB  test:{mdl_te:+6.2f} dB", end="")

            # ── blocked-only subset ───────────────────────────────────────────
            if args.blocked_only and blocked_mask is not None:
                local_pos, real_idx = blocked_subset(test_ds, blocked_mask)
                if len(local_pos) > 0:
                    ls_bl  = nmse_db(ls, ch, real_idx)
                    bl_loader = DataLoader(
                        Subset(test_ds, local_pos),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True,
                    )
                    mdl_bl = model_nmse_db(model, bl_loader, device)
                    row["ls_nmse_blocked"]    = round(ls_bl,  4)
                    row["model_nmse_blocked"] = round(mdl_bl, 4)
                    print(f"  |  blocked({len(real_idx)})  LS:{ls_bl:+6.2f} dB  model:{mdl_bl:+6.2f} dB",
                          end="")
                else:
                    print(f"  |  blocked(0) — no blocked users in test set", end="")

            print()
            rows.append(row)

    if not rows:
        print("No results to save.")
        return

    df = pd.DataFrame(rows)

    csv_path = os.path.join(args.out_dir, "eval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV     → {csv_path}")

    save_plot(df, os.path.join(args.out_dir, "eval_results.png"),
              title="Experiment 2 — model vs LS baselines across blockage runs (all test users)")

    if args.blocked_only and df["ls_nmse_blocked"].notna().any():
        save_plot(df, os.path.join(args.out_dir, "eval_results_blocked.png"),
                  title="Experiment 2 — model vs LS baselines across blockage runs (blocked users only)",
                  ls_col="ls_nmse_blocked", model_col="model_nmse_blocked")


if __name__ == "__main__":
    main()
