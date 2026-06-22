#!/usr/bin/env python3
"""
make_augmented_channels.py
--------------------------
Multi-blocker sweep augmentation.  Generates (n_runs + 1) run directories
under out_dir, one per blocker count:

    run_0000  →  0 trucks   (original seed data, no blockage)
    run_0001  →  step trucks
    run_0002  →  2*step trucks
    ...
    run_N     →  N*step trucks

Default: n_runs=10, step=2  →  11 runs, 0/2/4/.../20 trucks.

Truck positions are PRE-GENERATED ONCE and grown cumulatively — run_k uses
exactly the same first k*step trucks as run_{k-1}, plus step new ones.

Each run is saved independently:
    {out_dir}/run_XXXX/
        channels.npy            — real-stacked float32 (N, 2D, Nr, Nt)
        locations.txt           — exact UE positions from inter_locs
        locations_noisy.txt     — UE positions + Gaussian GPS noise
        ls_snr+{SNR}.npy        — LS OFDM estimates (real-stacked float32)
        blocked_mask.npy        — bool (N,) from actual channel-tensor power drop
        blocked_summary.json
        truck_params.json
    {out_dir}/plots/aug_gen/run_XXXX.png

Usage
    docker exec dpinn0 python3 make_augmented_channels.py \\
        --csv Dataset/15GHz_concatenated_data.csv

Parallel generation (split by run range, same seed → consistent trucks):
    docker exec dpinn0 python3 make_augmented_channels.py \\
        --start-run 0 --end-run 5
    docker exec dpinn0 python3 make_augmented_channels.py \\
        --start-run 6 --end-run 10
"""

import argparse
import glob
import json
import os
import re as _re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from tqdm import tqdm

from make_correct_channels import safe_parse_list, build_channel_tensor


# ── Truck OBB geometry ────────────────────────────────────────────────────────

class Truck:
    """Oriented Bounding Box truck model."""

    def __init__(self, center_x, center_y, heading_deg=0.0,
                 width=2.5, length=12.0, height=4.0):
        self.cx      = center_x
        self.cy      = center_y
        self.heading = heading_deg
        self.w       = width
        self.l       = length
        self.h       = height

        hrad = np.deg2rad(heading_deg)
        self.cos_h = np.cos(hrad)
        self.sin_h = np.sin(hrad)

        hw = width  / 2.0
        hl = length / 2.0
        dx = abs(hl * self.cos_h) + abs(hw * self.sin_h)
        dy = abs(hl * self.sin_h) + abs(hw * self.cos_h)
        self.box_min = np.array([center_x - dx, center_y - dy, 0.0])
        self.box_max = np.array([center_x + dx, center_y + dy, height])

    def to_dict(self):
        return dict(
            center_x=self.cx, center_y=self.cy, heading_deg=self.heading,
            width=self.w, length=self.l, height=self.h,
            box_min=self.box_min.tolist(), box_max=self.box_max.tolist(),
        )


def _ray_intersects_obb(p0, p1, truck):
    """
    3-D slab test in the truck's local frame.
    XY slab determines the parametric interval [t_min, t_max] where the
    segment is inside the truck footprint.  A Z-height check then confirms
    the segment actually passes through the physical truck volume [0, truck.h].
    Paths that pass entirely above the truck roof are not attenuated.
    """
    dx0, dy0 = p0[0] - truck.cx, p0[1] - truck.cy
    dx1, dy1 = p1[0] - truck.cx, p1[1] - truck.cy
    c, s = truck.cos_h, truck.sin_h
    lx0 =  c * dx0 + s * dy0;  ly0 = -s * dx0 + c * dy0
    lx1 =  c * dx1 + s * dy1;  ly1 = -s * dx1 + c * dy1
    hl, hw = truck.l / 2.0, truck.w / 2.0
    t_min, t_max = 0.0, 1.0
    for p_lo, p_hi, lo, hi in ((lx0, lx1, -hl, hl), (ly0, ly1, -hw, hw)):
        d = p_hi - p_lo
        if abs(d) < 1e-9:
            if p_lo < lo or p_lo > hi:
                return False
        else:
            t1 = (lo - p_lo) / d
            t2 = (hi - p_lo) / d
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
    if t_min > t_max:
        return False
    # Z check: segment Z at entry and exit of the XY slab
    z0, z1 = float(p0[2]), float(p1[2])
    z_entry = z0 + t_min * (z1 - z0)
    z_exit  = z0 + t_max * (z1 - z0)
    z_seg_min = min(z_entry, z_exit)
    z_seg_max = max(z_entry, z_exit)
    # intersects truck volume only if the Z range overlaps [0, truck.h]
    return z_seg_min <= truck.h and z_seg_max >= 0.0


def path_intersects_truck(inter_locs, truck):
    """True if any MPC bounce segment intersects the truck's 3D OBB volume."""
    if len(inter_locs) < 2:
        return False
    pts = [np.array([p[0], p[1], p[2]], dtype=float) for p in inter_locs]
    return any(_ray_intersects_obb(pts[i], pts[i + 1], truck)
               for i in range(len(pts) - 1))


# ── inter_locs parser ─────────────────────────────────────────────────────────

def _parse_inter_locs(raw):
    """Parse numpy-print-style inter_locs CSV field into list-of-paths."""
    parts = _re.split(r'\]\]\s*,\s*', raw.strip())
    paths = []
    for i, part in enumerate(parts):
        part = part.strip()
        if i < len(parts) - 1:
            part = part + ']'
        else:
            if part.endswith(']]'):
                part = part[:-1]
        rows = _re.findall(
            r'\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]', part
        )
        if rows:
            paths.append([[float(x), float(y), float(z)] for x, y, z in rows])
    return paths


# ── truck placement ───────────────────────────────────────────────────────────

def _place_truck(anchor_xy, bs_xy, bbox_half, truck_length, truck_width, rng):
    """
    Place one truck between the anchor user and the BS.
    Long axis perpendicular to BS direction so broadside faces all BS→UE paths.
    Truck sits close to the user cluster (standoff = bbox_half + width/2 + 1 m).
    """
    bs_vec  = bs_xy - anchor_xy
    bs_norm = np.linalg.norm(bs_vec)
    if bs_norm < 1e-6:
        angle  = rng.uniform(0, 2 * np.pi)
        bs_dir = np.array([np.cos(angle), np.sin(angle)])
    else:
        bs_dir = bs_vec / bs_norm

    perp_dir    = np.array([-bs_dir[1], bs_dir[0]])
    heading_deg = np.degrees(np.arctan2(perp_dir[1], perp_dir[0]))
    standoff    = bbox_half + truck_width / 2.0 + 1.0
    center      = anchor_xy + bs_dir * standoff

    return Truck(center_x=float(center[0]), center_y=float(center[1]),
                 heading_deg=float(heading_deg),
                 width=truck_width, length=truck_length)


def _estimate_local_heading_deg(anchor_xy, valid_xy, k_neighbors=40):
    """Estimate local road tangent heading (deg) from nearby UE points via PCA."""
    n = len(valid_xy)
    if n < 2:
        return 0.0
    k = int(max(2, min(k_neighbors, n)))
    d2 = np.sum((valid_xy - anchor_xy) ** 2, axis=1)
    idx = np.argpartition(d2, k - 1)[:k]
    neigh = valid_xy[idx]
    centered = neigh - np.mean(neigh, axis=0, keepdims=True)
    cov = centered.T @ centered
    vals, vecs = np.linalg.eigh(cov)
    tangent = vecs[:, int(np.argmax(vals))]
    return float(np.degrees(np.arctan2(tangent[1], tangent[0])))


def _project_polygon(points, axis):
    dots = points @ axis
    return float(np.min(dots)), float(np.max(dots))


def _polygons_overlap_sat(poly_a, poly_b):
    """2D convex polygon overlap using SAT (returns True for touch/overlap)."""
    for poly in (poly_a, poly_b):
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i + 1) % len(poly)]
            edge = p1 - p0
            nrm = np.array([-edge[1], edge[0]], dtype=float)
            nn = np.linalg.norm(nrm)
            if nn < 1e-12:
                continue
            axis = nrm / nn
            a0, a1 = _project_polygon(poly_a, axis)
            b0, b1 = _project_polygon(poly_b, axis)
            if a1 < b0 or b1 < a0:
                return False
    return True


def _trucks_overlap(t1, t2, min_gap=0.0):
    """True if two truck footprints overlap (with optional clearance margin)."""
    if min_gap > 0:
        w1 = t1.w + 2.0 * min_gap
        l1 = t1.l + 2.0 * min_gap
        w2 = t2.w + 2.0 * min_gap
        l2 = t2.l + 2.0 * min_gap
        a = Truck(t1.cx, t1.cy, t1.heading, width=w1, length=l1, height=t1.h)
        b = Truck(t2.cx, t2.cy, t2.heading, width=w2, length=l2, height=t2.h)
        c1 = _truck_corners(a)
        c2 = _truck_corners(b)
    else:
        c1 = _truck_corners(t1)
        c2 = _truck_corners(t2)
    return _polygons_overlap_sat(c1, c2)


def _place_all_trucks(n_total, valid_xy, bs_xy, truck_length, truck_width,
                      truck_height, bbox_half, rng,
                      placement_mode="bs_perp", road_half_width=0.0,
                      min_gap=0.0, max_tries_per_truck=500, heading_mode="tangent",
                      max_overlap_trucks=0):
    """Pre-generate n_total trucks sequentially from rng."""
    trucks = []
    n_valid = len(valid_xy)
    if n_valid == 0:
        return trucks

    if placement_mode == "road" and max_overlap_trucks == 0:
        available = np.ones(n_valid, dtype=bool)
        # Conservative center-distance exclusion radius to prevent OBB overlap.
        circ_radius = 0.5 * float(np.hypot(truck_length, truck_width)) + float(min_gap)
        excl_radius = 2.0 * circ_radius + 2.0 * float(max(0.0, road_half_width))
        excl_r2 = excl_radius * excl_radius

        for _ in range(n_total):
            avail_idx = np.flatnonzero(available)
            if len(avail_idx) == 0:
                print(f"WARNING: no available road anchors left; placed {len(trucks)}/{n_total}")
                break

            pick = int(avail_idx[int(rng.integers(0, len(avail_idx)))])
            anchor_xy = valid_xy[pick]

            center = np.array(anchor_xy, dtype=float)
            heading_deg = _estimate_local_heading_deg(anchor_xy, valid_xy)

            if road_half_width > 0:
                tangent = np.array([np.cos(np.deg2rad(heading_deg)), np.sin(np.deg2rad(heading_deg))])
                normal = np.array([-tangent[1], tangent[0]])
                lateral = float(rng.uniform(-road_half_width, road_half_width))
                center = center + lateral * normal

            if heading_mode == "bs_perp":
                bs_vec = bs_xy - center
                bs_norm = np.linalg.norm(bs_vec)
                if bs_norm < 1e-6:
                    heading_deg = float(rng.uniform(0.0, 360.0))
                else:
                    bs_dir = bs_vec / bs_norm
                    perp_dir = np.array([-bs_dir[1], bs_dir[0]])
                    heading_deg = float(np.degrees(np.arctan2(perp_dir[1], perp_dir[0])))

            t = Truck(center_x=float(center[0]), center_y=float(center[1]),
                      heading_deg=heading_deg,
                      width=truck_width, length=truck_length, height=truck_height)
            trucks.append(t)

            # Remove nearby anchors (+ buffer), ensuring future trucks cannot overlap.
            d2 = np.sum((valid_xy - center[np.newaxis, :]) ** 2, axis=1)
            available[d2 <= excl_r2] = False
    else:
        overlap_degree = []
        for _ in range(n_total):
            placed = False
            for _try in range(max_tries_per_truck):
                anchor_xy = valid_xy[int(rng.integers(0, n_valid))]
                if placement_mode == "road":
                    center = np.array(anchor_xy, dtype=float)
                    heading_deg = _estimate_local_heading_deg(anchor_xy, valid_xy)
                    if road_half_width > 0:
                        tangent = np.array([np.cos(np.deg2rad(heading_deg)), np.sin(np.deg2rad(heading_deg))])
                        normal = np.array([-tangent[1], tangent[0]])
                        lateral = float(rng.uniform(-road_half_width, road_half_width))
                        center = center + lateral * normal
                    if heading_mode == "bs_perp":
                        bs_vec = bs_xy - center
                        bs_norm = np.linalg.norm(bs_vec)
                        if bs_norm < 1e-6:
                            heading_deg = float(rng.uniform(0.0, 360.0))
                        else:
                            bs_dir = bs_vec / bs_norm
                            perp_dir = np.array([-bs_dir[1], bs_dir[0]])
                            heading_deg = float(np.degrees(np.arctan2(perp_dir[1], perp_dir[0])))
                    t = Truck(center_x=float(center[0]), center_y=float(center[1]),
                              heading_deg=heading_deg,
                              width=truck_width, length=truck_length, height=truck_height)
                else:
                    t = _place_truck(anchor_xy, bs_xy, bbox_half, truck_length, truck_width, rng)
                    t.h = truck_height
                    t.box_max[2] = truck_height

                overlap_idx = [i for i, ex in enumerate(trucks)
                               if _trucks_overlap(t, ex, min_gap=min_gap)]
                # max_overlap_trucks=0 -> no overlaps allowed.
                # max_overlap_trucks=1 -> one pair overlap allowed (no triple stacking).
                if len(overlap_idx) > max_overlap_trucks:
                    continue
                if any(overlap_degree[i] >= max_overlap_trucks for i in overlap_idx):
                    continue

                trucks.append(t)
                overlap_degree.append(len(overlap_idx))
                for i in overlap_idx:
                    overlap_degree[i] += 1
                placed = True
                break
            if not placed:
                print(f"WARNING: could not place all trucks without overlap; placed {len(trucks)}/{n_total}")
                break
    return trucks


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_real_stacked(arr):
    return np.concatenate([np.real(arr), np.imag(arr)], axis=1).astype(np.float32)


def _to_complex(arr):
    n = arr.shape[1] // 2
    return arr[:, :n].astype(np.complex128) + 1j * arr[:, n:].astype(np.complex128)


def _write_positions(xyz, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("x y z\n")
        for row in xyz:
            f.write(f"  {row[0]:.4f}  {row[1]:.4f}  {row[2]:.4f}\n")


def _add_noise(xyz, std, rng):
    noisy = xyz.copy()
    noisy[:, 0] += rng.normal(0.0, std, size=len(xyz))
    noisy[:, 1] += rng.normal(0.0, std, size=len(xyz))
    return noisy


def _truck_corners(truck):
    hrad = np.deg2rad(truck.heading)
    c, s = np.cos(hrad), np.sin(hrad)
    hl, hw = truck.l / 2, truck.w / 2
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([truck.cx, truck.cy])


# ── UE position extraction ────────────────────────────────────────────────────

def _extract_ue_positions(parsed_paths):
    xyz = []
    for paths in parsed_paths:
        if paths and len(paths[0]) >= 1:
            xyz.append(paths[0][-1])
        else:
            xyz.append([float("nan")] * 3)
    xyz_all = np.array(xyz, dtype=float)
    return xyz_all[:, :2], xyz_all


def _find_valid_indices(df):
    valid = []
    for i in range(len(df)):
        arrs = [safe_parse_list(df.loc[i, col])
                for col in ("AOD_PHI", "AOD_THETA", "AOA_PHI", "AOA_THETA",
                            "Pathgain", "ToA", "PHASE")]
        if any(len(a) == 0 for a in arrs):
            continue
        if len(set(len(a) for a in arrs)) != 1:
            continue
        valid.append(i)
    return valid


# ── blockage application ──────────────────────────────────────────────────────

def _apply_blockage(df, valid_indices, trucks, losses_db,
                    bw, n_tap, parsed_paths, parsed_pg, parsed_toa,
                    hard_block=False):
    """
    Apply NLOSv blockage from multiple trucks.
    losses_db : (n_valid, len(trucks)) pre-generated per-user-per-truck attenuation.
    hard_block: if True, any intercepted path is completely zeroed (-300 dB) instead
                of being attenuated by losses_db.
    Returns (df_mod, power_loss_db_raw) — blocked_mask computed later from tensors.
    """
    df_mod        = df.copy()
    n_valid       = len(valid_indices)
    power_loss_db = np.zeros(n_valid, dtype=float)

    for vi, user_idx in enumerate(valid_indices):
        paths   = parsed_paths[user_idx]
        pg_raw  = parsed_pg[user_idx]
        toa_raw = parsed_toa[user_idx]

        if not paths or len(pg_raw) == 0:
            continue

        min_toa     = float(np.min(toa_raw)) if len(toa_raw) > 0 else 0.0
        pg_modified = pg_raw.astype(float).copy()

        for mpc_idx, bounce_path in enumerate(paths):
            if mpc_idx >= len(pg_raw):
                break
            if hard_block:
                if any(path_intersects_truck(bounce_path, t) for t in trucks):
                    pg_modified[mpc_idx] = -300.0
            else:
                intercepting = [losses_db[vi, j]
                                for j, truck in enumerate(trucks)
                                if path_intersects_truck(bounce_path, truck)]
                if intercepting:
                    pg_modified[mpc_idx] -= max(intercepting)

        df_mod.at[user_idx, "Pathgain"] = str(pg_modified.tolist())

        orig_lin = np.sum(10 ** (pg_raw.astype(float) / 10))
        mod_lin  = np.sum(10 ** (pg_modified / 10))
        if orig_lin > 0 and mod_lin > 0:
            power_loss_db[vi] = 10.0 * np.log10(orig_lin / mod_lin)

    return df_mod, power_loss_db


# ── LS estimate generation ────────────────────────────────────────────────────

def _generate_ls(ch_real, snr_db, ref_noise_per_sample=None):
    """
    Generate real-stacked LS estimates from channels.
    Accepts either real-stacked float32 (N,32,4,576) or complex128 (N,16,4,576).
    Uses per-sample ref_noise_per_sample if provided (fixed noise floor),
    otherwise uses SNR-adaptive noise per sample.
    """
    from init_estimation import create_ls_ofdm_estimates
    import tempfile

    ch_cplx = _to_complex(ch_real)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_ch = f.name
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_ls = f.name

    try:
        np.save(tmp_ch, ch_cplx)
        create_ls_ofdm_estimates(
            tmp_ch, tmp_ls,
            SNR_dB=snr_db,
            method="basic",
            fixed_noise_power=ref_noise_per_sample,
        )
        ls_cplx = np.load(tmp_ls)
        return _to_real_stacked(ls_cplx)
    finally:
        for p in (tmp_ch, tmp_ls):
            if os.path.exists(p):
                os.remove(p)


# ── plot ──────────────────────────────────────────────────────────────────────

def _save_plot(run_id, n_trucks, trucks, valid_xy, blocked_mask,
               power_loss_db, bs_xy, out_path):
    fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(16, 7))

    ax_map.scatter(valid_xy[:, 0], valid_xy[:, 1],
                   c="lightgray", s=4, zorder=2, label="unblocked")
    if blocked_mask.any():
        ax_map.scatter(valid_xy[blocked_mask, 0], valid_xy[blocked_mask, 1],
                       c="red", s=10, zorder=3,
                       label=f"blocked ({blocked_mask.sum()})")

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(trucks), 1)))
    for t_idx, truck in enumerate(trucks):
        poly = MplPolygon(_truck_corners(truck), closed=True,
                          facecolor=colors[t_idx % len(colors)],
                          edgecolor="black", linewidth=0.8, alpha=0.7, zorder=4)
        ax_map.add_patch(poly)

    ax_map.scatter(*bs_xy, marker="*", s=200, c="blue", zorder=5, label="BS")
    ax_map.set_xlabel("x (m)"); ax_map.set_ylabel("y (m)")
    ax_map.set_title(f"Run {run_id:04d} — {n_trucks} trucks  |  "
                     f"{int(blocked_mask.sum())}/{len(valid_xy)} blocked")
    ax_map.legend(fontsize=7, loc="upper right")
    ax_map.set_aspect("equal", "datalim"); ax_map.grid(True, alpha=0.3)

    losses = power_loss_db[blocked_mask]
    if len(losses) > 0:
        ax_hist.hist(losses, bins=30, color="tomato", edgecolor="black", linewidth=0.5)
        ax_hist.axvline(float(np.mean(losses)), color="darkred", linestyle="--",
                        linewidth=1.5, label=f"mean {np.mean(losses):.1f} dB")
        ax_hist.legend(fontsize=9)
    else:
        ax_hist.text(0.5, 0.5, "no blocked users", ha="center", va="center",
                     transform=ax_hist.transAxes)

    ax_hist.set_xlabel("Channel power loss (dB)"); ax_hist.set_ylabel("Users")
    ax_hist.set_title("Power-loss distribution (blocked users)")
    ax_hist.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)


# ── per-run processing ────────────────────────────────────────────────────────

_BLOCK_THRESH_DB = 1e-6  # flag any user whose channel changed at all (any intercepted path)


def _run_one(run_id, n_trucks, df, xyz_all, valid_indices,
             trucks, losses_db, noise_seed, ch_unblocked,
             run_dir, plot_path, cfg,
             parsed_paths, parsed_pg, parsed_toa,
             ls_modes=("adaptive",),
             snr_info=None,
             overwrite=False,
             hard_block=False):
    """
    snr_info : dict  snr_value → (snr_tag, fixed_noise_scalar, ref_noise_per_sample)
    noise_seed: int  seed for this run's independent GPS noise realization
    overwrite : bool — delete channel/LS/plot files before processing so they are rebuilt
    """
    if snr_info is None:
        snr_info = {}

    os.makedirs(run_dir, exist_ok=True)

    if overwrite:
        targets = (
            [os.path.join(run_dir, f) for f in ("channels.npy", "blocked_mask.npy",
                                                  "blocked_summary.json")]
            + glob.glob(os.path.join(run_dir, "ls_snr*.npy"))
            + ([plot_path] if plot_path and os.path.exists(plot_path) else [])
        )
        for p in targets:
            if os.path.exists(p):
                os.remove(p)
                print(f"  [run {run_id:04d}]  removed {os.path.basename(p)}")
    bs_xy     = np.array([cfg.bs_x, cfg.bs_y])
    valid_xyz = xyz_all[valid_indices]

    print(f"  [run {run_id:04d}]  {n_trucks} trucks", end="")
    if trucks:
        centres = ", ".join(f"({t.cx:.0f},{t.cy:.0f})" for t in trucks[:4])
        if len(trucks) > 4:
            centres += f" … +{len(trucks)-4}"
        print(f"  centres: {centres}", end="")
    print()

    if getattr(cfg, "plots_only", False):
        df_mod, pg_loss_db = _apply_blockage(
            df, valid_indices, trucks, losses_db,
            cfg.bw, cfg.n_tap, parsed_paths, parsed_pg, parsed_toa,
            hard_block=hard_block,
        )
        _ = df_mod  # parsed for consistency; not used further in plot-only mode
        blocked_mask = pg_loss_db > 0.0
        tensor_loss_db = pg_loss_db
        with open(os.path.join(run_dir, "truck_params.json"), "w") as fh:
            json.dump({"bs_xy": bs_xy.tolist(),
                       "trucks": [t.to_dict() for t in trucks]}, fh, indent=2)
        _save_plot(run_id, n_trucks, trucks, valid_xyz[:, :2],
                   blocked_mask, tensor_loss_db, bs_xy, plot_path)
        print(f"  [run {run_id:04d}]  plot-only done → {plot_path}")
        return []

    ch_path   = os.path.join(run_dir, "channels.npy")
    mask_path = os.path.join(run_dir, "blocked_mask.npy")

    mask_only = False
    if os.path.exists(ch_path) and os.path.exists(mask_path):
        ch_real      = np.load(ch_path)
        blocked_mask = np.load(mask_path)
        # recompute tensor power drop so the plot is correct
        power_orig = np.sum(np.abs(ch_unblocked) ** 2, axis=(1, 2, 3))
        power_mod  = np.sum(np.abs(_to_complex(ch_real)) ** 2, axis=(1, 2, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            tensor_loss_db = np.where(
                power_orig > 0,
                10.0 * np.log10(np.maximum(power_orig, 1e-300) /
                                np.maximum(power_mod,  1e-300)),
                0.0,
            )
        print(f"  [run {run_id:04d}]  channels.npy exists — skipping channel build")
    elif os.path.exists(ch_path) and not os.path.exists(mask_path):
        # channels exist but mask was deleted — recompute mask only, skip LS entirely
        mask_only = True
        ch_real   = np.load(ch_path)
        n_valid   = len(ch_real)
        power_orig = np.sum(np.abs(ch_unblocked) ** 2, axis=(1, 2, 3))
        power_mod  = np.sum(np.abs(_to_complex(ch_real)) ** 2, axis=(1, 2, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            tensor_loss_db = np.where(
                power_orig > 0,
                10.0 * np.log10(np.maximum(power_orig, 1e-300) /
                                np.maximum(power_mod,  1e-300)),
                0.0,
            )
        blocked_mask = tensor_loss_db > _BLOCK_THRESH_DB
        n_blocked    = int(blocked_mask.sum())
        np.save(mask_path, blocked_mask)
        with open(os.path.join(run_dir, "blocked_summary.json"), "w") as fh:
            json.dump({
                "run_id": run_id, "n_trucks": n_trucks,
                "n_valid_channels": n_valid, "n_blocked_users": n_blocked,
                "channel_shape": list(ch_real.shape),
                "blocked_mask_method": "tensor_power_drop",
                "blocked_mask_thresh_db": _BLOCK_THRESH_DB,
            }, fh, indent=2)
        print(f"  [run {run_id:04d}]  mask recomputed from existing channels  "
              f"({n_blocked} blocked, thresh={_BLOCK_THRESH_DB} dB)")
        if plot_path and os.path.exists(plot_path):
            os.remove(plot_path)   # force plot to regenerate with updated mask
    else:
        # 1. apply path-gain blockage
        df_mod, pg_loss_db = _apply_blockage(
            df, valid_indices, trucks, losses_db,
            cfg.bw, cfg.n_tap, parsed_paths, parsed_pg, parsed_toa,
            hard_block=hard_block,
        )

        # 2. build channel tensor
        df_valid = df_mod.iloc[valid_indices].reset_index(drop=True)
        channel_matrices = build_channel_tensor(
            df_valid,
            N_tx_x=cfg.n_tx_x, N_tx_y=cfg.n_tx_y,
            N_rx_x=cfg.n_rx_x, N_rx_y=cfg.n_rx_y,
            N_tap=cfg.n_tap, Bw=cfg.bw, Pt=cfg.pt,
        )
        if len(channel_matrices) == 0:
            print(f"  [run {run_id:04d}] WARNING: no valid channels — skipping.")
            return []

        # 3. blocked mask from actual tensor power drop vs unblocked reference
        power_orig = np.sum(np.abs(ch_unblocked) ** 2, axis=(1, 2, 3))
        power_mod  = np.sum(np.abs(channel_matrices) ** 2, axis=(1, 2, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            tensor_loss_db = np.where(
                power_orig > 0,
                10.0 * np.log10(np.maximum(power_orig, 1e-300) /
                                np.maximum(power_mod,  1e-300)),
                0.0,
            )
        blocked_mask = tensor_loss_db > _BLOCK_THRESH_DB
        n_blocked    = int(blocked_mask.sum())
        n_valid      = len(channel_matrices)
        print(f"  [run {run_id:04d}]  {n_valid} channels  "
              f"{n_blocked} blocked (>{_BLOCK_THRESH_DB} dB tensor-power drop)")

        # 4. save channels + exact positions
        ch_real = _to_real_stacked(channel_matrices)
        np.save(ch_path,   ch_real)
        np.save(mask_path, blocked_mask)
        _write_positions(valid_xyz, os.path.join(run_dir, "locations.txt"))

        # JSON metadata
        with open(os.path.join(run_dir, "blocked_summary.json"), "w") as fh:
            json.dump({
                "run_id": run_id, "n_trucks": n_trucks,
                "n_valid_channels": n_valid, "n_blocked_users": n_blocked,
                "channel_shape": list(ch_real.shape),
                "blocked_mask_method": "tensor_power_drop",
                "blocked_mask_thresh_db": _BLOCK_THRESH_DB,
            }, fh, indent=2)
        with open(os.path.join(run_dir, "truck_params.json"), "w") as fh:
            json.dump({"bs_xy": bs_xy.tolist(),
                       "trucks": [t.to_dict() for t in trucks]}, fh, indent=2)

    # plot — always generated if missing (works for both fresh and skip paths)
    if not os.path.exists(plot_path):
        _save_plot(run_id, n_trucks, trucks, valid_xyz[:, :2],
                   blocked_mask, tensor_loss_db, bs_xy, plot_path)

    # independent GPS noise per run
    noisy_path = os.path.join(run_dir, "locations_noisy.txt")
    if not os.path.exists(noisy_path):
        run_noise_rng = np.random.default_rng(noise_seed)
        noisy_xyz = _add_noise(valid_xyz, cfg.user_noise, run_noise_rng)
        _write_positions(noisy_xyz, noisy_path)

    # 5. generate LS estimates (one file per SNR × mode) and compute NMSE — skip if mask-only
    if mask_only:
        return []
    ch_f32 = ch_real.astype(np.float32)
    ch_pow = float(np.sum(ch_f32 ** 2))
    nmse_rows = []

    for snr, (snr_tag, fixed_s, ref_arr) in snr_info.items():
        _ls_mode_cfgs = {
            "adaptive":  (f"ls_snr{snr_tag}.npy",          None),
            "fixed":     (f"ls_snr{snr_tag}_fixed.npy",     fixed_s),
            "refnoise":  (f"ls_snr{snr_tag}_refnoise.npy",  ref_arr),
        }
        for mode in ls_modes:
            fname, noise_param = _ls_mode_cfgs[mode]
            ls_path = os.path.join(run_dir, fname)
            if not os.path.exists(ls_path):
                print(f"  [run {run_id:04d}]  LS SNR={snr:+.0f} dB ({mode}) ...")
                np.save(ls_path, _generate_ls(ch_real, snr, noise_param))
            else:
                print(f"  [run {run_id:04d}]  LS SNR={snr:+.0f} dB ({mode}) exists — skipping")

            ls_f32 = np.load(ls_path, mmap_mode='r').astype(np.float32)
            nmse   = 10.0 * np.log10(np.sum((ls_f32 - ch_f32) ** 2) / ch_pow)
            nmse_rows.append({
                "run_id":  run_id,
                "snr":     int(snr),
                "ls_mode": mode,
                "nmse_db": round(float(nmse), 4),
            })
            print(f"  [run {run_id:04d}]  LS NMSE SNR={snr:+.0f} dB ({mode}): {nmse:.2f} dB")

    print(f"  [run {run_id:04d}]  done → {run_dir}")
    return nmse_rows


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Multi-blocker sweep augmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",      default="Dataset/15GHz_concatenated_data.csv")
    p.add_argument("--out-dir",  default="data")
    p.add_argument("--n-runs",   type=int, default=10,
                   help="Number of non-zero-truck runs. Total runs = n_runs+1 "
                        "(including run_0000 with 0 trucks).")
    p.add_argument("--step",     type=int, default=5,
                   help="Trucks added per run. run_k has k*step trucks.")
    p.add_argument("--start-run", type=int, default=None,
                   help="First run_id to generate (inclusive). Default: 0.")
    p.add_argument("--end-run",   type=int, default=None,
                   help="Last run_id to generate (inclusive). Default: n_runs.")
    p.add_argument("--snr-list",  nargs="+", type=float, default=[0.0],
                   help="SNR(s) in dB for LS generation. "
                        "One set of LS files is written per SNR value.")
    p.add_argument("--ls-modes", nargs="+", default=["adaptive"],
                   choices=["adaptive", "fixed", "refnoise"],
                   help="LS noise modes to generate per run.  "
                        "adaptive=SNR-adaptive per sample (default), "
                        "fixed=global scalar noise from run_0000 avg power, "
                        "refnoise=per-sample noise from run_0000 per-user power.")
    # channel physics
    p.add_argument("--pt",      type=float, default=50.0)
    p.add_argument("--bw",      type=float, default=4e8)
    p.add_argument("--n-tx-x",  type=int,   default=24)
    p.add_argument("--n-tx-y",  type=int,   default=24)
    p.add_argument("--n-rx-x",  type=int,   default=2)
    p.add_argument("--n-rx-y",  type=int,   default=2)
    p.add_argument("--n-tap",   type=int,   default=16)
    # BS position
    p.add_argument("--bs-x",    type=float, default=71.06)
    p.add_argument("--bs-y",    type=float, default=246.29)
    # truck geometry
    p.add_argument("--truck-width",  type=float, default=2.5)
    p.add_argument("--truck-length", type=float, default=12.0)
    p.add_argument("--truck-height", type=float, default=4.0)
    p.add_argument("--bbox-half",    type=float, default=2.5)
    p.add_argument("--placement-mode", choices=["bs_perp", "road"], default="road",
                   help="Truck placement strategy: bs_perp (legacy) or road-constrained.")
    p.add_argument("--heading-mode", choices=["tangent", "bs_perp"], default="tangent",
                   help="Heading strategy for road mode.")
    p.add_argument("--road-half-width", type=float, default=0.0,
                   help="Optional lateral jitter (m) around sampled road center points.")
    p.add_argument("--min-truck-gap", type=float, default=0.5,
                   help="Minimum clearance (m) between truck footprints.")
    p.add_argument("--max-place-tries", type=int, default=500,
                   help="Max placement attempts per truck before giving up.")
    p.add_argument("--max-overlap-trucks", type=int, default=0,
                   help="Allowed number of overlaps per truck footprint: 0=no overlap, 1=allow pair overlap.")
    # attenuation
    p.add_argument("--atten-min",  type=float, default=2.0)
    p.add_argument("--atten-max",  type=float, default=10.0)
    p.add_argument("--hard-block", action="store_true",
                   help="Completely zero any MPC that passes through a truck "
                        "(−300 dB), ignoring atten-min/max.")
    # GPS noise
    p.add_argument("--user-noise", type=float, default=0.5)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--overwrite",  action="store_true",
                   help="Delete channel/LS/plot files before regenerating each run. "
                        "Preserves truck_params.json, locations*.txt, and rss_cache.npy.")
    p.add_argument("--run-id-offset", type=int, default=0,
                   help="Add this offset to output run directory indices. "
                        "E.g. --run-id-offset 10 writes internal run_1…10 as run_0011…0020. "
                        "Truck counts are still computed from the internal run index × step.")
    p.add_argument("--skip-summary", action="store_true",
                   help="Skip writing blockage_summary.csv and ls_nmse.csv. "
                        "Use when running many parallel instances to avoid write conflicts.")
    p.add_argument("--dir-trucks", action="store_true",
                   help="Name each output directory after its truck count instead of its "
                        "run index.  E.g. --step 10 --start-run 3 --end-run 10 writes "
                        "run_0030/run_0040/.../run_0100 with 30/40/.../100 trucks. "
                        "Overrides --run-id-offset.")
    p.add_argument("--plots-only", action="store_true",
                   help="Fast mode: only generate blocker layout plots; skip channel/LS files.")
    return p.parse_args()


def main():
    args   = _parse_args()
    n_runs = args.n_runs
    step   = args.step
    total_trucks = n_runs * step        # e.g. 10 * 2 = 20

    start_run = args.start_run if args.start_run is not None else 0
    end_run   = args.end_run   if args.end_run   is not None else n_runs
    run_ids   = range(start_run, end_run + 1)

    plots_dir = os.path.join(args.out_dir, "plots", "aug_gen")
    os.makedirs(plots_dir, exist_ok=True)

    bs_xy = np.array([args.bs_x, args.bs_y])

    # ── load CSV once ──────────────────────────────────────────────────────────
    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    df.reset_index(drop=True, inplace=True)
    n_users = len(df)

    # ── pre-parse inter_locs / Pathgain / ToA ──────────────────────────────────
    print(f"\nPre-parsing {n_users} rows ...")
    parsed_paths, parsed_pg, parsed_toa = [], [], []
    for i in tqdm(range(n_users), desc="  parsing"):
        raw = df.loc[i, "inter_locs"]
        parsed_paths.append(_parse_inter_locs(str(raw)) if isinstance(raw, str) else [])
        parsed_pg.append(safe_parse_list(df.loc[i, "Pathgain"]))
        parsed_toa.append(safe_parse_list(df.loc[i, "ToA"]))

    # ── extract UE positions ───────────────────────────────────────────────────
    xy_all, xyz_all = _extract_ue_positions(parsed_paths)
    valid_xy = xy_all[~np.isnan(xy_all).any(axis=1)]
    print(f"UE positions: {n_users} rows  "
          f"x=[{valid_xy[:,0].min():.1f},{valid_xy[:,0].max():.1f}]  "
          f"y=[{valid_xy[:,1].min():.1f},{valid_xy[:,1].max():.1f}]")

    # ── find valid row indices ─────────────────────────────────────────────────
    print("\nFinding valid rows ...")
    valid_indices = _find_valid_indices(df)
    print(f"  {len(valid_indices)} / {n_users} rows valid")

    if args.plots_only:
        ch_unblocked = None
        snr_info = {}
        print("\nplots-only mode: skipping channel tensor + LS noise references")
    else:
        # ── unblocked reference tensor — load from run_0000 if it exists ──────
        run0_ch = os.path.join(args.out_dir, "run_0000", "channels.npy")
        if os.path.exists(run0_ch):
            print(f"\nLoading unblocked reference from {run0_ch} ...")
            ch_unblocked = _to_complex(np.load(run0_ch))
        else:
            print("\nBuilding unblocked reference channel tensor (run_0000 not yet generated) ...")
            df_orig      = df.iloc[valid_indices].reset_index(drop=True)
            ch_unblocked = build_channel_tensor(
                df_orig,
                N_tx_x=args.n_tx_x, N_tx_y=args.n_tx_y,
                N_rx_x=args.n_rx_x, N_rx_y=args.n_rx_y,
                N_tap=args.n_tap, Bw=args.bw, Pt=args.pt,
            )
        print(f"  unblocked tensor: {ch_unblocked.shape}")

        # ── noise references per SNR (for fixed / refnoise LS modes) ──────────
        ch_ub_pow = np.abs(ch_unblocked) ** 2
        snr_info  = {}
        print(f"\nNoise references ({len(args.snr_list)} SNR value(s)):")
        for snr in args.snr_list:
            snr_int = int(snr)
            snr_tag = f"+{snr_int}" if snr_int >= 0 else str(snr_int)
            snr_lin = 10 ** (snr / 10.0)
            fixed_s = float(np.mean(ch_ub_pow)) / snr_lin
            ref_arr = np.mean(ch_ub_pow, axis=(1, 2, 3)) / snr_lin
            snr_info[snr] = (snr_tag, fixed_s, ref_arr)
            print(f"  SNR={snr:+.0f} dB  fixed_scalar={fixed_s:.3e}  ref_mean={ref_arr.mean():.3e}")

    # ── pre-generate all trucks + attenuation matrix ───────────────────────────
    master_rng = np.random.default_rng(args.seed)
    truck_rng  = np.random.default_rng(master_rng.integers(0, 2**31))
    atten_rng  = np.random.default_rng(master_rng.integers(0, 2**31))
    noise_rng  = np.random.default_rng(master_rng.integers(0, 2**31))

    n_valid = len(valid_indices)

    print(f"\nPre-placing {total_trucks} trucks ...")
    all_trucks = _place_all_trucks(
        total_trucks, xy_all[valid_indices], bs_xy,
        args.truck_length, args.truck_width, args.truck_height,
        args.bbox_half, truck_rng,
        placement_mode=args.placement_mode,
        road_half_width=args.road_half_width,
        min_gap=args.min_truck_gap,
        max_tries_per_truck=args.max_place_tries,
        heading_mode=args.heading_mode,
        max_overlap_trucks=args.max_overlap_trucks,
    )

    print(f"Pre-generating attenuation matrix ({n_valid} × {total_trucks}) ...")
    all_losses = np.empty((n_valid, total_trucks), dtype=np.float64)
    for t in range(total_trucks):
        all_losses[:, t] = atten_rng.uniform(args.atten_min, args.atten_max, size=n_valid)

    # per-run independent GPS noise seeds (deterministic from master seed)
    total_runs  = n_runs + 1
    noise_seeds = [int(noise_rng.integers(0, 2**31)) for _ in range(total_runs)]
    print(f"Per-run GPS noise: std={args.user_noise} m  (independent per run)")

    # ── sweep ─────────────────────────────────────────────────────────────────
    print(f"\nGenerating runs {start_run} … {end_run}  "
          f"(step={step} trucks/run, total pool={total_trucks})\n")

    all_nmse_rows = []
    for run_id in run_ids:
        n_trucks   = run_id * step
        trucks_k   = all_trucks[:n_trucks]
        losses_k   = all_losses[:, :n_trucks] if n_trucks > 0 else np.zeros((n_valid, 0))
        out_run_id = n_trucks if args.dir_trucks else run_id + args.run_id_offset
        run_dir    = os.path.join(args.out_dir, f"run_{out_run_id:04d}")
        plot_path  = os.path.join(plots_dir, f"run_{out_run_id:04d}.png")

        rows = _run_one(
            run_id=out_run_id, n_trucks=n_trucks,
            df=df, xyz_all=xyz_all, valid_indices=valid_indices,
            trucks=trucks_k, losses_db=losses_k,
            noise_seed=noise_seeds[run_id],
            ch_unblocked=ch_unblocked,
            run_dir=run_dir, plot_path=plot_path,
            cfg=args,
            parsed_paths=parsed_paths,
            parsed_pg=parsed_pg,
            parsed_toa=parsed_toa,
            ls_modes=args.ls_modes,
            snr_info=snr_info,
            overwrite=args.overwrite,
            hard_block=args.hard_block,
        )
        if rows:
            all_nmse_rows.extend(rows)

    print(f"\nDone. {len(run_ids)} runs saved under {args.out_dir}/run_XXXX/")

    if not args.skip_summary:
        # ── aggregate blockage_summary.csv from all existing runs ─────────────
        _write_blockage_summary(args.out_dir)

        # ── write LS NMSE summary CSV ──────────────────────────────────────────
        _write_ls_nmse_csv(all_nmse_rows)


def _write_blockage_summary(out_dir):
    """Scan all run_XXXX/blocked_summary.json and write a single CSV."""
    import csv, glob
    rows = []
    for jpath in sorted(glob.glob(os.path.join(out_dir, "run_*", "blocked_summary.json"))):
        with open(jpath) as f:
            d = json.load(f)
        n_valid   = d.get("n_valid_channels", 0)
        n_blocked = d.get("n_blocked_users",  0)
        rows.append({
            "run_id":      d.get("run_id",   ""),
            "n_trucks":    d.get("n_trucks",  ""),
            "n_valid":     n_valid,
            "n_blocked":   n_blocked,
            "pct_blocked": round(100.0 * n_blocked / n_valid, 2) if n_valid else 0.0,
        })
    if not rows:
        return
    csv_path = os.path.join(out_dir, "blockage_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "n_trucks", "n_valid", "n_blocked", "pct_blocked"])
        w.writeheader()
        w.writerows(rows)
    print(f"  blockage summary → {csv_path}  ({len(rows)} runs)")


def _write_ls_nmse_csv(new_rows, csv_path="models/ls_nmse.csv"):
    """
    Merge new_rows with any existing rows in csv_path (keyed by run_id+snr+ls_mode)
    and write the combined result.  Creates models/ if needed.
    """
    import csv as _csv

    if not new_rows:
        return

    # Load existing rows so partial runs accumulate correctly
    existing = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in _csv.DictReader(f):
                key = (int(row["run_id"]), int(row["snr"]), row["ls_mode"])
                existing[key] = row

    for row in new_rows:
        key = (row["run_id"], row["snr"], row["ls_mode"])
        existing[key] = row

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = ["run_id", "snr", "ls_mode", "nmse_db"]
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(existing.values(), key=lambda r: (int(r["run_id"]), int(r["snr"]), r["ls_mode"])))
    print(f"  LS NMSE summary  → {csv_path}  ({len(existing)} rows)")


if __name__ == "__main__":
    main()
