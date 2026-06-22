#!/usr/bin/env python3
"""
Run a trained model on all run_XXXX directories and save per-user results
to data/run_XXXX/seed_predictions_{ls_tag}.csv

Usage:
    python src/evaluate.py
    python src/evaluate.py --model models/snr0/random_3.0/simple_ls_val.pth --ls-file ls_snr+0.npy
    python src/evaluate.py --overwrite
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from Model import ImprovedPhysicsInformedUNet, set_seed
from find_in_map import RSSMapProcessor

RSS_IMAGE   = "data/raw/50_15GHz.jpg"
BS_PIXEL    = (287, 293)
BS_REAL     = (71.06, 246.29)
IMG_WIDTH_M = 527.5
SPLIT_SEED  = 42
TRAIN_RATIO = 0.8
BATCH_SIZE  = 64


def load_model(path, device):
    m  = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    ck = torch.load(path, map_location=device)
    st = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    m.load_state_dict(st)
    return m.to(device).eval()


def global_max_from_run0(data_dir, ls_file, seed):
    set_seed(seed)
    run0    = os.path.join(data_dir, "run_0000")
    ls      = np.load(os.path.join(run0, ls_file), mmap_mode="r")
    ch      = np.load(os.path.join(run0, "channels.npy"), mmap_mode="r")
    perm    = np.random.permutation(len(ls))
    train   = perm[:int(len(ls) * TRAIN_RATIO)]
    return float(max(np.max(np.abs(ls[train])), np.max(np.abs(ch[train]))))


def infer_all_users(model, ls_arr, ch_arr, rss_cache, global_max, device):
    scale  = float(global_max)
    ls_t   = torch.from_numpy(ls_arr.astype(np.float32)) / scale
    ch_t   = torch.from_numpy(ch_arr.astype(np.float32)) / scale
    rss_t  = torch.from_numpy(rss_cache.astype(np.float32))

    n      = len(ls_arr)
    err_sq = np.empty(n, dtype=np.float64)
    ch_pw  = np.empty(n, dtype=np.float64)

    loader = DataLoader(TensorDataset(ls_t, ch_t, rss_t),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    start = 0
    with torch.no_grad():
        for smomp, accurate, rss in loader:
            pred     = model(smomp.to(device), rss.to(device))
            accurate = accurate.to(device)
            e = torch.sum((pred - accurate) ** 2, dim=(1, 2, 3)).cpu().numpy()
            p = torch.sum(accurate ** 2,           dim=(1, 2, 3)).cpu().numpy()
            err_sq[start:start + len(e)] = e.astype(np.float64) * scale ** 2
            ch_pw [start:start + len(e)] = p.astype(np.float64) * scale ** 2
            start += len(e)
    return err_sq, ch_pw


def ls_err_pw(ls_arr, ch_arr):
    ls  = ls_arr.astype(np.float64)
    ch  = ch_arr.astype(np.float64)
    return np.sum((ls - ch) ** 2, axis=(1, 2, 3)), np.sum(ch ** 2, axis=(1, 2, 3))


def atten_vs_run0(ch0_arr, ch_arr):
    pw0 = np.sum(ch0_arr.astype(np.float64) ** 2, axis=(1, 2, 3))
    pw  = np.sum(ch_arr .astype(np.float64) ** 2, axis=(1, 2, 3))
    return 10.0 * np.log10((pw0 + 1e-300) / (pw + 1e-300))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir",  default="data")
    parser.add_argument("--model",     default="models/snr0/random_3.0/simple_ls_val.pth",
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--ls-file",   default="ls_snr+0.npy",
                        help="LS filename to use inside each run_XXXX/ directory")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run inference even if CSV already exists.")
    args = parser.parse_args()

    # derive output tag from ls filename, e.g. ls_snr+0.npy → snr+0
    ls_tag = args.ls_file.replace("ls_", "").replace(".npy", "")

    device     = torch.device(args.device)
    global_max = global_max_from_run0(args.data_dir, args.ls_file, SPLIT_SEED)
    print(f"global_max = {global_max:.4e}  (from run_0000/{args.ls_file})")

    model = load_model(args.model, device)
    print(f"Loaded model : {args.model}")
    print(f"LS file      : {args.ls_file}  (tag: {ls_tag})\n")

    ch0_arr = np.load(os.path.join(args.data_dir, "run_0000", "channels.npy"), mmap_mode="r")

    for run_dir in sorted(glob.glob(os.path.join(args.data_dir, "run_*"))):
        if not os.path.isdir(run_dir):
            continue

        ch_path    = os.path.join(run_dir, "channels.npy")
        cache_path = os.path.join(run_dir, "rss_cache.npy")
        ls_path    = os.path.join(run_dir, args.ls_file)
        out_csv    = os.path.join(run_dir, f"seed_predictions_{ls_tag}.csv")

        if not os.path.exists(ch_path):
            print(f"[{os.path.basename(run_dir)}] skip — no channels.npy")
            continue
        if not os.path.exists(cache_path):
            print(f"[{os.path.basename(run_dir)}] skip — no rss_cache.npy")
            continue
        if not os.path.exists(ls_path):
            print(f"[{os.path.basename(run_dir)}] skip — no {args.ls_file}")
            continue
        if os.path.exists(out_csv) and not args.overwrite:
            print(f"[{os.path.basename(run_dir)}] skip — CSV exists")
            continue

        ch_arr    = np.load(ch_path,    mmap_mode="r")
        rss_cache = np.load(cache_path, mmap_mode="r")
        ls_arr    = np.load(ls_path,    mmap_mode="r")
        atten_db  = atten_vs_run0(ch0_arr, ch_arr)

        print(f"[{os.path.basename(run_dir)}] inferring {len(ls_arr)} users ...", flush=True)
        seed_err, seed_pw = infer_all_users(model, ls_arr, ch_arr, rss_cache, global_max, device)
        ls_err, _         = ls_err_pw(ls_arr, ch_arr)

        safe_pw = np.maximum(seed_pw, 1e-300)
        pd.DataFrame({
            "user_idx":     np.arange(len(ls_arr)),
            "seed_err_sq":  seed_err,
            "ch_pw":        seed_pw,
            "ls_err_sq":    ls_err,
            "atten_db":     atten_db,
            "seed_nmse_db": 10.0 * np.log10(seed_err / safe_pw),
            "ls_nmse_db":   10.0 * np.log10(ls_err   / safe_pw),
        }).to_csv(out_csv, index=False)
        print(f"  saved → {out_csv}")


if __name__ == "__main__":
    main()
