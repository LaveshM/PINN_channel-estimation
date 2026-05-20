#!/usr/bin/env python3
"""
scripts/experiment3_rss.py
--------------------------
RSS map perturbation experiment.

For each SNR in {-10, -5, 0, +5} (random split, noise=3.0 m, run_0000):
  - loads the trained model for that SNR
  - evaluates on the test set at increasing levels of Gaussian noise
    added to the RSS crop cache (both grayscale and dBm channels)
  - records model NMSE and the LS NMSE baseline (RSS-independent)

This shows how sensitive each SNR model is to errors in the RSS map.

Results → models/old/experiment3_rss.csv + experiment3_rss.png

Usage:
    python3 scripts/experiment3_rss.py
    python3 scripts/experiment3_rss.py --sigmas 0 0.05 0.1 0.2 0.5 1.0
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
RUN_DIR     = "data/run_0000"

SNR_LIST    = [-10, -5, 0, 5]
DEFAULT_SIGMAS = [1.0, 0.0]

SNR_COLORS  = {-10: "#9467BD", -5: "#D65F5F", 0: "#4878CF", 5: "#6ACC65"}


# ── helpers ───────────────────────────────────────────────────────────────────

def snr_tag(snr):
    return f"+{int(snr)}" if snr >= 0 else str(int(snr))


def model_path(snr, model_dir="models"):
    return os.path.join(model_dir, f"snr{int(snr)}", f"random_{USER_NOISE}",
                        "simple_ls_val.pth")


def ls_path(snr, run_dir=RUN_DIR):
    return os.path.join(run_dir, f"ls_snr{snr_tag(snr)}.npy")


def load_model(ckpt_path, device):
    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def nmse_db_arrays(est, ch, idx):
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
    return float(10.0 * np.log10(total_err / total_pow))


def _save_plot(df, out_dir):
    """
    Single plot comparing model NMSE with clean RSS (sigma=0) vs perturbed RSS.
    X axis: SNR.  Two groups per SNR: clean and each perturbed sigma.
    LS baseline shown as a separate line.
    Saved to plots/experiment3_rss.png.
    """
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "experiment3_rss.png")

    snrs   = sorted(df["snr"].unique())
    sigmas = sorted(df["sigma"].unique())
    perturbed_sigmas = [s for s in sigmas if s > 0]

    fig, ax = plt.subplots(figsize=(9, 5))

    # LS baseline — one value per SNR (RSS-independent)
    ls_vals = [df[df["snr"] == snr]["ls_nmse"].iloc[0] for snr in snrs]
    ax.plot(snrs, ls_vals, color="black", linewidth=1.5, linestyle="--",
            marker="x", label="LS baseline", zorder=5)

    # Clean RSS (sigma = 0)
    clean = df[df["sigma"] == 0.0].sort_values("snr")
    ax.plot(clean["snr"], clean["model_nmse"],
            color="#4878CF", linewidth=2.5, marker="o", markersize=8,
            label="Model — clean RSS (σ=0)", zorder=6)

    # Perturbed RSS lines
    cmap = plt.cm.Oranges
    n_p  = max(len(perturbed_sigmas), 1)
    for i, sigma in enumerate(perturbed_sigmas):
        sub = df[df["sigma"] == sigma].sort_values("snr")
        color = cmap(0.35 + 0.55 * i / (n_p - 1) if n_p > 1 else 0.7)
        ax.plot(sub["snr"], sub["model_nmse"],
                color=color, linewidth=1.5, marker="o", markersize=6,
                linestyle="--", label=f"Model — perturbed RSS (σ={sigma})", alpha=0.85)

    ax.set_xticks(snrs)
    ax.set_xticklabels([snr_tag(s) + " dB" for s in snrs])
    ax.set_xlabel("SNR", fontsize=11)
    ax.set_ylabel("NMSE (dB)", fontsize=11)
    ax.set_title("Experiment 3 — Model NMSE: clean vs perturbed RSS map", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot → {out_path}")


def perturb_cache(clean_cache, sigma, seed=0):
    """Add Gaussian noise to RSS cache and clip to the original value ranges."""
    rng = np.random.default_rng(seed)
    noisy = clean_cache + rng.normal(0.0, sigma, clean_cache.shape).astype(np.float32)
    noisy[:, 0] = np.clip(noisy[:, 0], 0.0, 1.0)   # grayscale [0, 1]
    noisy[:, 1] = np.clip(noisy[:, 1], -1.0, 1.0)  # dBm-normalised [-1, 1]
    if sigma > 0:
        noisy = np.ones_like(clean_cache)*0   # default to NaN for any out-of-range values

    return noisy


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--run-dir",   default=RUN_DIR)
    parser.add_argument("--out-dir",   default="models/old")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sigmas",    nargs="+", type=float, default=DEFAULT_SIGMAS,
                        help="Noise std-dev values to sweep (applied to normalised RSS cache)")
    parser.add_argument("--snr-list",  nargs="+", type=float, default=SNR_LIST)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    ch_path  = os.path.join(args.run_dir, "channels.npy")
    pos_path = os.path.join(args.run_dir, "locations_noisy.txt")

    rss_processor = RSSMapProcessor(
        image_path=RSS_IMAGE, bs_pixel_coords=BS_PIXEL,
        bs_real_coords=BS_REAL, image_width_meters=IMG_WIDTH_M,
    )

    rows = []

    for snr in args.snr_list:
        ckpt = model_path(snr, args.model_dir)
        ls   = ls_path(snr, args.run_dir)

        if not os.path.exists(ckpt):
            print(f"[SNR={snr:+.0f}] SKIP — checkpoint not found: {ckpt}")
            continue
        if not os.path.exists(ls):
            print(f"[SNR={snr:+.0f}] SKIP — LS file not found: {ls}")
            continue

        print(f"\n[SNR={snr:+.0f} dB] {'─'*50}")
        model = load_model(ckpt, device)

        set_seed(SEED)
        train_ds, _, test_ds, _, _, _ = create_datasets(
            smomp_file=ls,
            accurate_file=ch_path,
            user_positions_file=pos_path,
            split_type=SPLIT_TYPE,
            user_noise=USER_NOISE,
            rss_processor=rss_processor,
        )

        # LS NMSE — RSS-independent baseline
        ch_arr  = np.load(ch_path, mmap_mode="r")
        ls_arr  = np.load(ls,      mmap_mode="r")
        ls_nmse = nmse_db_arrays(ls_arr, ch_arr, test_ds.indices)

        # Grab a writeable copy of the clean RSS cache
        clean_cache = np.array(test_ds.rss_cache)   # copy out of mmap

        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=2, pin_memory=True)

        print(f"  LS NMSE (baseline, no RSS): {ls_nmse:+.2f} dB")

        for sigma in args.sigmas:
            test_ds.rss_cache = perturb_cache(clean_cache, sigma) if sigma > 0 else clean_cache

            nmse = model_nmse_db(model, test_loader, device)
            print(f"  sigma={sigma:.2f}  model NMSE: {nmse:+.2f} dB")
            print(test_ds.rss_cache[:1, :, :5,:5])  # debug print of perturbed cache values
            rows.append({
                "snr":        int(snr),
                "sigma":      sigma,
                "ls_nmse":    round(ls_nmse, 4),
                "model_nmse": round(nmse,    4),
            })

        # restore clean cache for next SNR (though model changes anyway)
        test_ds.rss_cache = clean_cache

    if not rows:
        print("No results — check model paths.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "experiment3_rss.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  → {csv_path}")

    _save_plot(df, args.out_dir)


if __name__ == "__main__":
    main()
