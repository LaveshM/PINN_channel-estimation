#!/usr/bin/env python3
"""
train_augmented.py — Three-scenario PINN training for blockage augmentation study.

Scenarios
---------
A  : Train on seed data (run_0000) only.
     Eval on seed train/val/test  +  all aug data (run_0010).
B  : Train on aug  data (run_0010) only.
     Eval on aug  train/val/test  +  all seed data (run_0000).
C  : Train on combined seed + aug.
     Eval on combined train/val/test.
     Temp concat files live in models/scenario_C/ and are deleted after training.

Usage
-----
python3 train_augmented.py --scenario all --snr 0 --epochs 500
python3 train_augmented.py --scenario A   --snr 0 --device cuda:0
"""

import os
import sys
import time
import copy
import argparse
import csv

import numpy as np
import torch

from Model import (
    ImprovedPhysicsInformedUNet,
    GlobalNormalizedDataset,
    create_datasets,
    evaluate_test_set,
    train_model,
    set_seed,
    RSSMapProcessor,
)
from torch.utils.data import DataLoader

SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snr_tag(snr: float) -> str:
    s = int(snr)
    return f"+{s}" if s >= 0 else str(s)


def run_paths(run_dir: str, snr: float) -> dict:
    tag = _snr_tag(snr)
    return {
        "ch":  os.path.join(run_dir, "channels.npy"),
        "ls":  os.path.join(run_dir, f"ls_snr{tag}.npy"),
        "pos": os.path.join(run_dir, "locations_noisy.txt"),
    }


def _make_loader(ds, batch_size: int, shuffle: bool = False):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=2, persistent_workers=True,
        prefetch_factor=2, pin_memory=True,
    )


def _ls_nmse(ls_file: str, ch_file: str, indices) -> float:
    s = np.load(ls_file, mmap_mode='r')[indices].astype(np.float64)
    a = np.load(ch_file, mmap_mode='r')[indices].astype(np.float64)
    return float(10.0 * np.log10(np.sum((s - a) ** 2) / np.sum(a ** 2)))


def _all_loader(ls_file, ch_file, pos_file, norm, rss_proc, user_noise, batch_size):
    """DataLoader covering every sample in the given files, using provided norm params."""
    N = np.load(ls_file, mmap_mode='r').shape[0]
    ds = GlobalNormalizedDataset(
        smomp_file=ls_file,
        accurate_file=ch_file,
        user_positions_file=pos_file,
        rss_processor=rss_proc,
        normalization_params=norm,
        indices=np.arange(N),
        user_noise=user_noise,
        split='eval',
    )
    return _make_loader(ds, batch_size)


# ---------------------------------------------------------------------------
# Concat helpers (scenario C)
# ---------------------------------------------------------------------------

def _concat_files(seed_p: dict, aug_p: dict, tmp_dir: str):
    """Write concatenated seed+aug into tmp_dir. Returns (ls, ch, pos) paths."""
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_ls  = os.path.join(tmp_dir, "_tmp_ls.npy")
    tmp_ch  = os.path.join(tmp_dir, "_tmp_ch.npy")
    tmp_pos = os.path.join(tmp_dir, "_tmp_pos.txt")

    if not os.path.exists(tmp_ls):
        np.save(tmp_ls, np.concatenate([
            np.load(seed_p["ls"], mmap_mode='r'),
            np.load(aug_p["ls"],  mmap_mode='r'),
        ], axis=0))

    if not os.path.exists(tmp_ch):
        np.save(tmp_ch, np.concatenate([
            np.load(seed_p["ch"], mmap_mode='r'),
            np.load(aug_p["ch"],  mmap_mode='r'),
        ], axis=0))

    if not os.path.exists(tmp_pos):
        with open(seed_p["pos"]) as f:
            seed_lines = f.readlines()
        with open(aug_p["pos"]) as f:
            aug_lines = f.readlines()
        with open(tmp_pos, 'w') as f:
            f.writelines(seed_lines + aug_lines[1:])  # keep one header

    return tmp_ls, tmp_ch, tmp_pos


def _delete_tmp(tmp_dir: str):
    for name in ("_tmp_ls.npy", "_tmp_ch.npy", "_tmp_pos.txt", "rss_cache.npy"):
        p = os.path.join(tmp_dir, name)
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Eval / print / save
# ---------------------------------------------------------------------------

def _nmse_db(x: float) -> float:
    return 10.0 * np.log10(x)


def _eval(model, loader, device) -> float:
    return _nmse_db(evaluate_test_set(model, loader, device=device))


def _print_table(rows):
    """rows: list of (scen, eval_set, ls_db, pinn_db)"""
    hdr = f"  {'Scen':<6}  {'Eval set':<30}  {'LS (dB)':>9}  {'PINN (dB)':>10}"
    sep = "  " + "-" * 6 + "  " + "-" * 30 + "  " + "-" * 9 + "  " + "-" * 10
    print(hdr)
    print(sep)
    for scen, label, ls, pinn in rows:
        print(f"  {scen:<6}  {label:<30}  {ls:>9.2f}  {pinn:>10.2f}")


def _save_csv(rows, csv_path: str, train_times: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["scenario", "eval_set", "ls_nmse_db", "pinn_nmse_db", "train_time_min"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for scen, label, ls, pinn in rows:
            w.writerow({
                "scenario":      scen,
                "eval_set":      label,
                "ls_nmse_db":    round(ls, 4),
                "pinn_nmse_db":  round(pinn, 4),
                "train_time_min": round(train_times.get(scen, 0.0), 2),
            })


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def _train_scenario(name, ls_file, ch_file, pos_file, rss_proc, args, models_dir):
    """
    Train on (ls_file, ch_file, pos_file) with random 80/10/10 split.
    Returns (model_val, norm, tr_ld, va_ld, te_ld, train_time_min, ls_tr, ls_va, ls_te).
    """
    scen_dir = os.path.join(models_dir, f"scenario_{name}")
    os.makedirs(scen_dir, exist_ok=True)
    model_val_path   = os.path.join(scen_dir, "model_val.pth")
    model_train_path = os.path.join(scen_dir, "model_train.pth")

    set_seed(SEED)
    train_ds, val_ds, test_ds, ls_tr, ls_va, ls_te = create_datasets(
        smomp_file=ls_file,
        accurate_file=ch_file,
        user_positions_file=pos_file,
        split_type="random",
        user_noise=args.user_noise,
        rss_processor=rss_proc,
    )

    norm = train_ds.normalization_params

    tr_ld = _make_loader(train_ds, args.batch_size, shuffle=True)
    va_ld = _make_loader(val_ds,   args.batch_size)
    te_ld = _make_loader(test_ds,  args.batch_size)

    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))

    print(f"\n[Scenario {name}]  {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    print(f"  LS baseline — train: {ls_tr:.2f} dB  val: {ls_va:.2f} dB  test: {ls_te:.2f} dB")

    t0 = time.time()
    train_model(
        model, tr_ld, va_ld,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        model_name_val=model_val_path,
        model_name_train=model_train_path,
        continue_=args.resume,
    )
    train_time = (time.time() - t0) / 60.0
    print(f"  Training done in {train_time:.1f} min")

    model_val = copy.deepcopy(model)
    model_val.load_state_dict(torch.load(model_val_path, map_location=args.device))
    model_val.to(args.device)

    return model_val, norm, tr_ld, va_ld, te_ld, train_time, ls_tr, ls_va, ls_te


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_A(seed_p, aug_p, rss_proc, args, models_dir):
    model_val, norm, tr_ld, va_ld, te_ld, t_min, ls_tr, ls_va, ls_te = _train_scenario(
        "A", seed_p["ls"], seed_p["ch"], seed_p["pos"], rss_proc, args, models_dir
    )

    aug_all_ld = _all_loader(
        aug_p["ls"], aug_p["ch"], aug_p["pos"],
        norm, rss_proc, args.user_noise, args.batch_size,
    )
    N_aug = np.load(aug_p["ls"], mmap_mode='r').shape[0]
    ls_aug = _ls_nmse(aug_p["ls"], aug_p["ch"], np.arange(N_aug))

    rows = [
        ("A", "seed / train", ls_tr,  _eval(model_val, tr_ld,      args.device)),
        ("A", "seed / val",   ls_va,  _eval(model_val, va_ld,      args.device)),
        ("A", "seed / test",  ls_te,  _eval(model_val, te_ld,      args.device)),
        ("A", "aug  / all",   ls_aug, _eval(model_val, aug_all_ld, args.device)),
    ]
    return rows, {"A": t_min}


def scenario_B(seed_p, aug_p, rss_proc, args, models_dir):
    model_val, norm, tr_ld, va_ld, te_ld, t_min, ls_tr, ls_va, ls_te = _train_scenario(
        "B", aug_p["ls"], aug_p["ch"], aug_p["pos"], rss_proc, args, models_dir
    )

    seed_all_ld = _all_loader(
        seed_p["ls"], seed_p["ch"], seed_p["pos"],
        norm, rss_proc, args.user_noise, args.batch_size,
    )
    N_seed = np.load(seed_p["ls"], mmap_mode='r').shape[0]
    ls_seed = _ls_nmse(seed_p["ls"], seed_p["ch"], np.arange(N_seed))

    rows = [
        ("B", "aug  / train", ls_tr,   _eval(model_val, tr_ld,       args.device)),
        ("B", "aug  / val",   ls_va,   _eval(model_val, va_ld,       args.device)),
        ("B", "aug  / test",  ls_te,   _eval(model_val, te_ld,       args.device)),
        ("B", "seed / all",   ls_seed, _eval(model_val, seed_all_ld, args.device)),
    ]
    return rows, {"B": t_min}


def scenario_C(seed_p, aug_p, rss_proc, args, models_dir):
    tmp_dir = os.path.join(models_dir, "scenario_C")
    tmp_ls, tmp_ch, tmp_pos = _concat_files(seed_p, aug_p, tmp_dir)
    try:
        model_val, norm, tr_ld, va_ld, te_ld, t_min, ls_tr, ls_va, ls_te = _train_scenario(
            "C", tmp_ls, tmp_ch, tmp_pos, rss_proc, args, models_dir
        )
        rows = [
            ("C", "combined / train", ls_tr, _eval(model_val, tr_ld, args.device)),
            ("C", "combined / val",   ls_va, _eval(model_val, va_ld, args.device)),
            ("C", "combined / test",  ls_te, _eval(model_val, te_ld, args.device)),
        ]
    finally:
        _delete_tmp(tmp_dir)

    return rows, {"C": t_min}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PINN — 3 scenarios (A/B/C)")
    parser.add_argument("--scenario",    default="all", choices=["A", "B", "C", "all"],
                        help="Which scenario(s) to run")
    parser.add_argument("--seed-dir",    default="data/run_0000",
                        help="Directory with unblocked (seed) data")
    parser.add_argument("--aug-dir",     default="data/run_0010",
                        help="Directory with blocked (aug) data")
    parser.add_argument("--models-dir",  default="models",
                        help="Output directory for saved models and results CSV")
    parser.add_argument("--snr",         type=float, default=0.0)
    parser.add_argument("--epochs",      type=int,   default=500)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--resume",      action="store_true",
                        help="Continue training from existing checkpoint")
    parser.add_argument("--user-noise",  type=float, default=3.0,
                        help="GPS noise std-dev used during data generation (m)")
    parser.add_argument("--rss-image",   default="Dataset/50_15GHz.jpg")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.models_dir, exist_ok=True)

    seed_p = run_paths(args.seed_dir, args.snr)
    aug_p  = run_paths(args.aug_dir,  args.snr)

    for domain, p in [("seed", seed_p), ("aug", aug_p)]:
        for key, path in p.items():
            if not os.path.exists(path):
                print(f"ERROR: {domain}/{key} not found: {path}")
                sys.exit(1)

    rss_proc = RSSMapProcessor(
        image_path=args.rss_image,
        bs_pixel_coords=(287, 293),
        bs_real_coords=(71.06, 246.29),
        image_width_meters=527.5,
    )

    all_rows  = []
    all_times = {}
    csv_path  = os.path.join(args.models_dir, "results_aug.csv")

    if args.scenario in ("A", "all"):
        rows, times = scenario_A(seed_p, aug_p, rss_proc, args, args.models_dir)
        all_rows.extend(rows)
        all_times.update(times)

    if args.scenario in ("B", "all"):
        rows, times = scenario_B(seed_p, aug_p, rss_proc, args, args.models_dir)
        all_rows.extend(rows)
        all_times.update(times)

    if args.scenario in ("C", "all"):
        rows, times = scenario_C(seed_p, aug_p, rss_proc, args, args.models_dir)
        all_rows.extend(rows)
        all_times.update(times)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    _print_table(all_rows)
    print("=" * 65)

    _save_csv(all_rows, csv_path, all_times)
    print(f"\nResults saved to {csv_path}")
    for name, t in sorted(all_times.items()):
        print(f"  Scenario {name}: {t:.1f} min training time")


if __name__ == "__main__":
    main()
