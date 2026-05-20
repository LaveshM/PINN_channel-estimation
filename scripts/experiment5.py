#!/usr/bin/env python3
"""
scripts/experiment5.py
----------------------
Adaptive SNR model selection experiment across augmented blockage runs.

For each run (run_0000 … run_0010):
  - Input  : ls_snr+0_refnoise.npy  (LS estimate with fixed noise from unblocked ref)
  - Targets: channels.npy

  Three conditions are compared:
    1. LS refnoise baseline         — no neural model, just the LS input itself
    2. SNR=0 model (fixed baseline) — all samples evaluated with the SNR=0 trained model
    3. Adaptive model               — per-sample effective SNR estimated from channel power
                                      relative to the unblocked run_0000 reference, then
                                      the nearest trained SNR model is selected

  Effective SNR for sample i in run k:
      eff_snr_dB = 10 * log10( sum(ch_k[i]^2) / sum(ch_0000[i]^2) )
    (= 0 dB in run_0000, drops as blockage attenuates channels)

  Trained SNR models available: -10, -5, 0, +5 dB  →  models/snr{SNR}/random_3.0/simple_ls_val.pth

Outputs:
    plots/experiment5.csv
    plots/experiment5.png  (main NMSE panel + model-usage stacked-bar panel)

Usage:
    python3 scripts/experiment5.py
    python3 scripts/experiment5.py --data-dir data --device cpu
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Model import ImprovedPhysicsInformedUNet

# ── constants ─────────────────────────────────────────────────────────────────
SNR_LIST   = [-10, -5, 0, 5]
SNR_COLORS = {-10: "#9467BD", -5: "#D65F5F", 0: "#4878CF", 5: "#6ACC65"}
BATCH_SIZE = 64
TRUCKS_PER_RUN = 2   # step size used when generating augmented data


# ── helpers ───────────────────────────────────────────────────────────────────

def snr_tag(snr: int) -> str:
    return f"+{snr}" if snr >= 0 else str(snr)


def model_path(snr: int) -> str:
    return os.path.join("models", f"snr{snr}", "random_3.0", "simple_ls_val.pth")


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def nmse_db(pred: np.ndarray, true: np.ndarray) -> float:
    """Global NMSE in dB from float32/64 arrays."""
    e = pred.astype(np.float64) - true.astype(np.float64)
    return float(10.0 * np.log10(np.sum(e ** 2) / np.sum(true.astype(np.float64) ** 2)))


def forward_batched(model: torch.nn.Module,
                    ls_norm: np.ndarray,
                    rss: np.ndarray,
                    device: torch.device,
                    batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Forward pass on numpy arrays, return predictions in the same normalised scale."""
    preds = np.empty_like(ls_norm)
    with torch.no_grad():
        for i in range(0, len(ls_norm), batch_size):
            s = torch.as_tensor(ls_norm[i:i + batch_size]).float().to(device)
            r = torch.as_tensor(rss[i:i + batch_size]).float().to(device)
            preds[i:i + batch_size] = model(s, r).cpu().numpy()
    return preds


def snap_to_snr(eff_snr: np.ndarray, snr_list: list) -> np.ndarray:
    """Map each float SNR to the nearest discrete training SNR (index into snr_list)."""
    snr_arr = np.array(snr_list, dtype=float)
    diffs   = np.abs(snr_arr[None, :] - eff_snr[:, None])   # (N, 4)
    return np.argmin(diffs, axis=1)                          # (N,)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir",  default="data")
    parser.add_argument("--out-dir",   default="plots")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── load all SNR models ────────────────────────────────────────────────────
    models: dict[int, torch.nn.Module] = {}
    for snr in SNR_LIST:
        p = model_path(snr)
        if not os.path.exists(p):
            print(f"[WARNING] model for SNR={snr:+d} dB not found: {p}")
            continue
        models[snr] = load_model(p, device)
        print(f"Loaded  SNR={snr:+d} dB  ← {p}")

    if 0 not in models:
        print("ERROR: SNR=0 model is required as the fixed baseline.")
        return

    # ── reference channel powers from run_0000 (unblocked) ────────────────────
    run0_ch_path = os.path.join(args.data_dir, "run_0000", "channels.npy")
    if not os.path.exists(run0_ch_path):
        print(f"ERROR: run_0000 channels not found: {run0_ch_path}")
        return
    ch_run0 = np.load(run0_ch_path, mmap_mode="r")
    # per-sample total power: shape (N,)
    power_run0 = np.sum(ch_run0.astype(np.float64) ** 2, axis=(1, 2, 3))
    print(f"run_0000 reference: {ch_run0.shape}  mean power={power_run0.mean():.3e}\n")

    snr_arr = np.array(SNR_LIST, dtype=float)

    # ── iterate over runs ──────────────────────────────────────────────────────
    rows        = []
    usage_counts: dict[int, dict[int, int]] = {}

    run_dirs = sorted(glob.glob(os.path.join(args.data_dir, "run_*")))
    if not run_dirs:
        print(f"No run_XXXX directories found in {args.data_dir}")
        return

    for run_dir in run_dirs:
        run_id = int(os.path.basename(run_dir).split("_")[1])

        ch_path  = os.path.join(run_dir, "channels.npy")
        ls_path  = os.path.join(run_dir, "ls_snr+0_refnoise.npy")
        rss_path = os.path.join(run_dir, "rss_cache.npy")

        for p, label in [(ch_path, "channels.npy"),
                         (ls_path,  "ls_snr+0_refnoise.npy"),
                         (rss_path, "rss_cache.npy")]:
            if not os.path.exists(p):
                print(f"[run {run_id:04d}] SKIP — {label} missing")
                break
        else:
            pass  # all files present
        if not (os.path.exists(ch_path) and os.path.exists(ls_path) and os.path.exists(rss_path)):
            continue

        print(f"[run {run_id:04d}]  {run_id * TRUCKS_PER_RUN} trucks  {'─'*40}")

        ch  = np.load(ch_path,  mmap_mode="r")    # (N, 32, 4, 576) float32
        ls  = np.load(ls_path,  mmap_mode="r")    # (N, 32, 4, 576) float32
        rss = np.load(rss_path, mmap_mode="r")    # (N, 2, 64, 64)  float32
        n   = len(ch)

        # ── LS refnoise baseline ───────────────────────────────────────────────
        ls_nmse = nmse_db(ls, ch)
        print(f"  LS refnoise NMSE:      {ls_nmse:+7.2f} dB")

        # ── per-sample effective SNR ───────────────────────────────────────────
        power_k = np.sum(ch.astype(np.float64) ** 2, axis=(1, 2, 3))  # (N,)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_snr_db = np.where(
                power_run0 > 0,
                10.0 * np.log10(
                    np.maximum(power_k,     1e-300) /
                    np.maximum(power_run0,  1e-300)
                ),
                0.0,
            )
        # clamp to [min_snr, max_snr] before snapping
        eff_snr_clamped = np.clip(eff_snr_db, snr_arr.min(), snr_arr.max())
        assigned_idx    = snap_to_snr(eff_snr_clamped, SNR_LIST)   # (N,) index into SNR_LIST
        assigned_snr    = snr_arr[assigned_idx]                     # (N,) float

        counts = {snr: int(np.sum(assigned_snr == snr)) for snr in SNR_LIST}
        usage_counts[run_id] = counts
        print(f"  Model assignment:      " +
              "  ".join(f"SNR={snr_tag(s)}: {counts[s]:5d}" for s in SNR_LIST))

        # ── normalisation (global max over all samples in this run) ───────────
        global_max = float(max(np.max(np.abs(ls.astype(np.float32))),
                               np.max(np.abs(ch.astype(np.float32)))))

        ls_norm  = (ls.astype(np.float32)  / global_max)
        rss_copy = rss.astype(np.float32).copy()   # writeable copy for torch

        # ── SNR0 fixed baseline ────────────────────────────────────────────────
        preds_snr0 = forward_batched(models[0], ls_norm, rss_copy, device, args.batch_size)
        snr0_nmse  = nmse_db(preds_snr0 * global_max, ch.astype(np.float32))
        print(f"  SNR=0 model NMSE:      {snr0_nmse:+7.2f} dB")

        # ── adaptive model ─────────────────────────────────────────────────────
        preds_adaptive = np.empty_like(ls_norm)
        for snr in SNR_LIST:
            if snr not in models:
                # fall back to SNR0 model for any missing model
                fallback_idx = np.where(assigned_snr == snr)[0]
                if len(fallback_idx) == 0:
                    continue
                preds_adaptive[fallback_idx] = forward_batched(
                    models[0], ls_norm[fallback_idx], rss_copy[fallback_idx],
                    device, args.batch_size,
                )
                continue
            idx = np.where(assigned_snr == snr)[0]
            if len(idx) == 0:
                continue
            preds_adaptive[idx] = forward_batched(
                models[snr], ls_norm[idx], rss_copy[idx],
                device, args.batch_size,
            )
        adaptive_nmse = nmse_db(preds_adaptive * global_max, ch.astype(np.float32))
        print(f"  Adaptive model NMSE:   {adaptive_nmse:+7.2f} dB  "
              f"(Δ vs SNR0: {adaptive_nmse - snr0_nmse:+.2f} dB)")

        rows.append({
            "run_id":        run_id,
            "n_trucks":      run_id * TRUCKS_PER_RUN,
            "ls_nmse":       round(ls_nmse,       4),
            "snr0_nmse":     round(snr0_nmse,     4),
            "adaptive_nmse": round(adaptive_nmse, 4),
        })

    if not rows:
        print("No results — check data paths and model checkpoints.")
        return

    # ── save CSV ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows).sort_values("run_id")
    csv_path = os.path.join(args.out_dir, "experiment5.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  → {csv_path}")

    # ── plot ───────────────────────────────────────────────────────────────────
    run_ids = df["run_id"].values

    fig, (ax_main, ax_bar) = plt.subplots(
        1, 2, figsize=(15, 5.5),
        gridspec_kw={"width_ratios": [1.3, 1]},
    )

    # ── left panel: NMSE vs run ────────────────────────────────────────────────
    ax_main.plot(run_ids, df["ls_nmse"].values,
                 color="black", linewidth=1.5, linestyle="--", marker="x", markersize=7,
                 label="LS refnoise baseline")
    ax_main.plot(run_ids, df["snr0_nmse"].values,
                 color="#4878CF", linewidth=2.0, linestyle="-", marker="o", markersize=7,
                 label="SNR=0 model (fixed baseline)")
    ax_main.plot(run_ids, df["adaptive_nmse"].values,
                 color="#6ACC65", linewidth=2.5, linestyle="-", marker="^", markersize=8,
                 label="Adaptive SNR model")

    ax_main.set_xticks(run_ids)
    ax_main.set_xticklabels([str(r) for r in run_ids])
    ax_main.set_xlabel("Run ID  (×2 trucks)", fontsize=11)
    ax_main.set_ylabel("NMSE (dB)", fontsize=11)
    ax_main.set_title("Experiment 5 — Adaptive SNR Model Selection\nAugmented channels, LS refnoise input",
                      fontsize=11)
    ax_main.legend(fontsize=9, loc="best")
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

    # secondary x-axis: number of trucks
    ax_main2 = ax_main.twiny()
    ax_main2.set_xlim(ax_main.get_xlim())
    ax_main2.set_xticks(run_ids)
    ax_main2.set_xticklabels([str(r * TRUCKS_PER_RUN) for r in run_ids], fontsize=8)
    ax_main2.set_xlabel("Number of trucks", fontsize=9)

    # ── right panel: model-usage stacked bar ──────────────────────────────────
    x         = np.arange(len(run_ids))
    bar_width = 0.65
    bottom    = np.zeros(len(run_ids))

    for snr in SNR_LIST:
        counts_vec = np.array([usage_counts.get(rid, {}).get(snr, 0) for rid in run_ids],
                              dtype=float)
        ax_bar.bar(x, counts_vec, bar_width, bottom=bottom,
                   label=f"SNR={snr_tag(snr)} dB", color=SNR_COLORS[snr], alpha=0.88)
        bottom += counts_vec

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(r) for r in run_ids])
    ax_bar.set_xlabel("Run ID  (×2 trucks)", fontsize=11)
    ax_bar.set_ylabel("Samples assigned to model", fontsize=11)
    ax_bar.set_title("Model selection frequency per run", fontsize=11)
    ax_bar.legend(fontsize=9, loc="upper right")
    ax_bar.grid(True, alpha=0.3, axis="y")

    # secondary x-axis: number of trucks
    ax_bar2 = ax_bar.twiny()
    ax_bar2.set_xlim(ax_bar.get_xlim())
    ax_bar2.set_xticks(x)
    ax_bar2.set_xticklabels([str(r * TRUCKS_PER_RUN) for r in run_ids], fontsize=8)
    ax_bar2.set_xlabel("Number of trucks", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "experiment5.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot → {out_path}")


if __name__ == "__main__":
    main()
