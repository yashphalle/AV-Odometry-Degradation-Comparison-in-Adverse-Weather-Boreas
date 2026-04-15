#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np


def load_tum(path):
    """Load TUM trajectory: timestamp tx ty tz qx qy qz qw"""
    data = np.loadtxt(path)
    timestamps = data[:, 0]
    poses_txyz = data[:, 1:4]
    poses_quat = data[:, 4:8]  # qx qy qz qw
    return timestamps, poses_txyz, poses_quat


def load_boreas_gt(csv_path):
    """
    Load Boreas lidar_poses.csv.
    Columns: timestamp(us), x, y, z, roll, pitch, yaw (+ possibly more)
    Returns timestamps in seconds and xyz positions.
    """
    import csv

    timestamps = []
    xyz = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["GPSTime"]) / 1e6
            x = float(row["easting"])
            y = float(row["northing"])
            z = float(row["altitude"])
            timestamps.append(ts)
            xyz.append([x, y, z])

    return np.array(timestamps), np.array(xyz)


def tum_quat_to_matrix(txyz, quat):
    """Convert TUM poses (N x 3 xyz, N x 4 qxqyqzqw) to N x 4x4 matrices."""
    from scipy.spatial.transform import Rotation
    N = len(txyz)
    matrices = np.eye(4)[None].repeat(N, axis=0)
    matrices[:, :3, 3] = txyz
    matrices[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    return matrices


def rpy_to_matrix(roll, pitch, yaw):
    """Convert RPY (radians) to 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def load_boreas_gt_as_matrices(csv_path):
    """Load GT as 4x4 matrices and timestamps."""
    import csv

    timestamps = []
    matrices = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["GPSTime"]) / 1e6
            x = float(row["easting"])
            y = float(row["northing"])
            z = float(row["altitude"])
            roll  = float(row["roll"])
            pitch = float(row["pitch"])
            yaw   = float(row["heading"])

            T = np.eye(4)
            T[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
            T[:3, 3]  = [x, y, z]

            timestamps.append(ts)
            matrices.append(T)

    timestamps = np.array(timestamps)
    matrices   = np.array(matrices)

    T0_inv = np.linalg.inv(matrices[0])
    for i in range(len(matrices)):
        matrices[i] = T0_inv @ matrices[i]

    return timestamps, matrices


def associate_timestamps(ts_est, ts_gt, max_diff=0.05):
    """
    Match estimated timestamps to GT timestamps.
    Returns (est_indices, gt_indices) of matched pairs.
    max_diff: max allowed time difference in seconds.
    """
    est_idx, gt_idx = [], []
    for i, ts in enumerate(ts_est):
        j = np.argmin(np.abs(ts_gt - ts))
        if np.abs(ts_gt[j] - ts) <= max_diff:
            est_idx.append(i)
            gt_idx.append(j)
    return np.array(est_idx), np.array(gt_idx)


def umeyama_alignment(src, dst):
    """
    Umeyama alignment: find scale s, rotation R, translation t such that
    dst ≈ s * R @ src + t.
    src, dst: (N, 3) arrays.
    Returns aligned src (N, 3).
    """
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = (src_c ** 2).sum() / n
    H = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(H)
    det_sign = np.linalg.det(U @ Vt)
    S = np.diag([1, 1, det_sign])

    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / sigma_src
    t = mu_dst - s * R @ mu_src

    return (s * (R @ src_c.T).T) + mu_dst


def compute_ate(est_xyz, gt_xyz):
    """
    Absolute Trajectory Error.
    Aligns est to gt via Umeyama, then computes per-frame translation error.
    Returns dict with rmse, mean, median, std, max.
    """
    aligned = umeyama_alignment(est_xyz, gt_xyz)
    errors = np.linalg.norm(aligned - gt_xyz, axis=1)
    return {
        "rmse":   float(np.sqrt((errors ** 2).mean())),
        "mean":   float(errors.mean()),
        "median": float(np.median(errors)),
        "std":    float(errors.std()),
        "max":    float(errors.max()),
    }, aligned


def compute_rpe(est_matrices, gt_matrices, delta=1):
    """
    Relative Pose Error over delta-frame pairs.
    Returns dict with rmse, mean, median, std, max for translation and rotation.
    """
    trans_errors = []
    rot_errors   = []
    from scipy.spatial.transform import Rotation

    n = min(len(est_matrices), len(gt_matrices))
    for i in range(n - delta):
        # Relative poses
        Q_est = np.linalg.inv(est_matrices[i]) @ est_matrices[i + delta]
        Q_gt  = np.linalg.inv(gt_matrices[i])  @ gt_matrices[i + delta]

        Q_err = np.linalg.inv(Q_gt) @ Q_est

        trans_errors.append(np.linalg.norm(Q_err[:3, 3]))

        # Rotation error in degrees
        R_err = Q_err[:3, :3]
        angle = Rotation.from_matrix(R_err).magnitude()
        rot_errors.append(np.degrees(angle))

    trans_errors = np.array(trans_errors)
    rot_errors   = np.array(rot_errors)

    def stats(arr):
        return {
            "rmse":   float(np.sqrt((arr ** 2).mean())),
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "std":    float(arr.std()),
            "max":    float(arr.max()),
        }

    return stats(trans_errors), stats(rot_errors)



def print_results(label, algo, ate, rpe_t, rpe_r):
    print(f"\n{'='*50}")
    print(f"  Results: {algo} | {label}")
    print(f"{'='*50}")
    print(f"  ATE (m)  — RMSE: {ate['rmse']:.4f}  Mean: {ate['mean']:.4f}  "
          f"Median: {ate['median']:.4f}  Max: {ate['max']:.4f}")
    print(f"  RPE-t(m) — RMSE: {rpe_t['rmse']:.4f}  Mean: {rpe_t['mean']:.4f}  "
          f"Median: {rpe_t['median']:.4f}  Max: {rpe_t['max']:.4f}")
    print(f"  RPE-r(°) — RMSE: {rpe_r['rmse']:.4f}  Mean: {rpe_r['mean']:.4f}  "
          f"Median: {rpe_r['median']:.4f}  Max: {rpe_r['max']:.4f}")
    print(f"{'='*50}\n")


def save_results(label, algo, ate, rpe_t, rpe_r, out_dir):
    """Save metrics to a CSV for later aggregation."""
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo}_{label}_metrics.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "type", "rmse", "mean", "median", "std", "max"])
        writer.writerow(["ATE",   "translation(m)", ate["rmse"],   ate["mean"],   ate["median"],   ate["std"],   ate["max"]])
        writer.writerow(["RPE-t", "translation(m)", rpe_t["rmse"], rpe_t["mean"], rpe_t["median"], rpe_t["std"], rpe_t["max"]])
        writer.writerow(["RPE-r", "rotation(deg)",  rpe_r["rmse"], rpe_r["mean"], rpe_r["median"], rpe_r["std"], rpe_r["max"]])

    print(f"[SAVED] Metrics -> {out_path}")
    return out_path


def save_plots(label, algo, est_xyz, gt_xyz, aligned_xyz, ate_errors, out_dir):
    """Save trajectory and ATE error plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Trajectory plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_xyz[:, 0],      gt_xyz[:, 1],      "g-",  linewidth=1.5, label="Ground Truth")
    ax.plot(aligned_xyz[:, 0], aligned_xyz[:, 1], "b--", linewidth=1,   label=f"{algo} (aligned)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Trajectory — {algo} | {label}")
    ax.legend()
    ax.set_aspect("equal")
    traj_path = out_dir / f"{algo}_{label}_trajectory.png"
    fig.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Trajectory plot -> {traj_path}")

    # ATE over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ate_errors, linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("ATE (m)")
    ax.set_title(f"ATE over time — {algo} | {label}")
    ate_path = out_dir / f"{algo}_{label}_ate.png"
    fig.savefig(ate_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] ATE plot -> {ate_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate odometry output against Boreas GT.")
    parser.add_argument("--poses",      required=True,  help="Path to estimated poses (TUM format)")
    parser.add_argument("--gt",         required=True,  help="Path to lidar_poses.csv (Boreas GT)")
    parser.add_argument("--label",      required=True,  help="Weather label (clear/snow/rain)")
    parser.add_argument("--algo",       default="unknown", help="Algorithm name (e.g. kiss_icp, loam)")
    parser.add_argument("--results",    default="results",  help="Root dir for output metrics/plots")
    parser.add_argument("--max-diff",   type=float, default=0.05,
                        help="Max timestamp diff (s) for GT association (default: 0.05)")
    parser.add_argument("--rpe-delta",  type=int,   default=1,
                        help="Frame delta for RPE (default: 1)")
    parser.add_argument("--save-plots", action="store_true", help="Save trajectory and ATE plots")
    return parser.parse_args()


def main():
    args = parse_args()
    poses_path = Path(args.poses)
    gt_path    = Path(args.gt)
    out_dir    = Path(args.results) / args.label

    if not poses_path.exists():
        print(f"[ERROR] Poses file not found: {poses_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.exists():
        print(f"[ERROR] GT file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[EVAL] {args.algo} | {args.label}")
    print(f"  Poses : {poses_path}")
    print(f"  GT    : {gt_path}")

    # Load
    est_ts, est_xyz, est_quat   = load_tum(poses_path)
    gt_ts,  gt_matrices         = load_boreas_gt_as_matrices(gt_path)
    gt_xyz  = gt_matrices[:, :3, 3]

    print(f"  Estimated poses : {len(est_ts)}")
    print(f"  GT poses        : {len(gt_ts)}")

    # Associate
    est_idx, gt_idx = associate_timestamps(est_ts, gt_ts, max_diff=args.max_diff)
    print(f"  Matched pairs   : {len(est_idx)} (max_diff={args.max_diff}s)")

    if len(est_idx) < 10:
        print("[ERROR] Too few matched pairs — check timestamp alignment.", file=sys.stderr)
        sys.exit(1)

    est_xyz_m = est_xyz[est_idx]
    gt_xyz_m  = gt_xyz[gt_idx]

    # Build 4x4 matrices for matched poses (for RPE)
    est_matrices_m = tum_quat_to_matrix(est_xyz_m, est_quat[est_idx])
    gt_matrices_m  = gt_matrices[gt_idx]

    # ATE
    ate, aligned_xyz = compute_ate(est_xyz_m, gt_xyz_m)

    # ATE per-frame errors (for plotting)
    aligned_full = umeyama_alignment(est_xyz_m, gt_xyz_m)
    ate_errors   = np.linalg.norm(aligned_full - gt_xyz_m, axis=1)

    # RPE
    rpe_t, rpe_r = compute_rpe(est_matrices_m, gt_matrices_m, delta=args.rpe_delta)

    # Print
    print_results(args.label, args.algo, ate, rpe_t, rpe_r)

    # Save metrics CSV
    save_results(args.label, args.algo, ate, rpe_t, rpe_r, out_dir)

    # Save plots
    if args.save_plots:
        save_plots(args.label, args.algo, est_xyz_m, gt_xyz_m, aligned_xyz, ate_errors, out_dir)

    print("=== Evaluation complete ===")


if __name__ == "__main__":
    main()