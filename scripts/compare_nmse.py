"""
Compare three NMSE formulations between true channels and LS estimates.

Usage:
    python scripts/compare_nmse.py --true data/3D_channel_15GHz_2x2_Pt50_real.npy \
                                   --est  data/snr0/initial_estimate_ls_real.npy

Both files must be real-stacked arrays of shape (N, 2*D, Nr, Nt) where the
first D slices are real parts and the last D slices are imaginary parts.
"""

import argparse
import sys
import os

import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def to_complex(arr):
    """Split real-stacked (N, 2D, Nr, Nt) → complex (N, D, Nr, Nt)."""
    n_ch = arr.shape[1] // 2
    return arr[:, :n_ch] + 1j * arr[:, n_ch:]


def db(x):
    return 10.0 * np.log10(x)


# ── three NMSE methods ────────────────────────────────────────────────────────

def nmse_global_mean(true_c, est_c):
    """
    Method 1 — global mean (same as PhysicsInformedLoss.calculate_nmse).

        NMSE = mean(|est - true|²) / mean(|true|²)

    Equivalent to summing squared errors over every element and normalising
    by the total signal power.  A single scalar for the whole dataset.
    """
    mse    = np.mean(np.abs(est_c - true_c) ** 2)
    power  = np.mean(np.abs(true_c) ** 2)
    return mse / power


def nmse_per_sample_mean(true_c, est_c):
    """
    Method 2 — per-sample sum then average (matches evaluate_test_set).

        nmse_i = sum_dims(|est_i - true_i|²) / sum_dims(|true_i|²)
        NMSE   = mean_i(nmse_i)

    Each sample is normalised by its own power before averaging, so
    low-power samples are not drowned out by high-power ones.
    """
    # sum over all non-batch dimensions
    axes = tuple(range(1, true_c.ndim))
    err_per_sample = np.sum(np.abs(est_c - true_c) ** 2, axis=axes)
    pow_per_sample = np.sum(np.abs(true_c) ** 2,          axis=axes)
    return float(np.mean(err_per_sample / pow_per_sample))


def nmse_per_element_mean(true_c, est_c):
    """
    Method 3 — per-element ratio then average.

        NMSE = mean( |est - true|² / |true|² )

    Every element is normalised independently before averaging.
    Elements near zero in the true channel dominate, so this is
    the most sensitive to small-amplitude entries.
    """
    per_element = np.abs(est_c - true_c) ** 2 / (np.abs(true_c) ** 2 + 1e-12)
    return float(np.mean(per_element))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare NMSE formulations")
    parser.add_argument("--true", required=True,
                        help="Path to ground-truth channel file (.npy, real-stacked)")
    parser.add_argument("--est",  required=True,
                        help="Path to LS estimate file (.npy, real-stacked)")
    args = parser.parse_args()

    for path in (args.true, args.est):
        if not os.path.exists(path):
            sys.exit(f"File not found: {path}")

    true_arr = np.load(args.true)
    est_arr  = np.load(args.est)

    if true_arr.shape != est_arr.shape:
        sys.exit(f"Shape mismatch: true {true_arr.shape} vs est {est_arr.shape}")

    print(f"Loaded  true : {args.true}  shape={true_arr.shape}")
    print(f"Loaded  est  : {args.est}  shape={est_arr.shape}")
    print()

    true_c = to_complex(true_arr)
    est_c  = to_complex(est_arr)

    m1 = nmse_global_mean(true_c, est_c)
    m2 = nmse_per_sample_mean(true_c, est_c)
    m3 = nmse_per_element_mean(true_c, est_c)

    col = 52
    print("=" * col)
    print(f"{'NMSE comparison':^{col}}")
    print("=" * col)
    print(f"{'Method':<38} {'Linear':>6}  {'dB':>7}")
    print("-" * col)
    print(f"{'1. Global mean  (sum / sum)' :<38} {m1:>6.4f}  {db(m1):>+7.2f}")
    print(f"{'2. Per-sample   (mean of ratios)':<38} {m2:>6.4f}  {db(m2):>+7.2f}")
    print(f"{'3. Per-element  (mean of ratios)':<38} {m3:>6.4f}  {db(m3):>+7.2f}")
    print("=" * col)


if __name__ == "__main__":
    main()
