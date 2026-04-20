#!/usr/bin/env python3
import argparse
import csv
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


def tum_quat_to_matrix(txyz, quat):
    """Convert TUM poses (N x 3 xyz, N x 4 qxqyqzqw) to N x 4x4 matrices."""
    from scipy.spatial.transform import Rotation

    n = len(txyz)
    matrices = np.eye(4)[None].repeat(n, axis=0)
    matrices[:, :3, 3] = txyz
    matrices[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    return matrices


def rpy_to_matrix(roll, pitch, yaw):
    """Convert roll, pitch, yaw in radians to a 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def get_first_present(row, candidates, required=True):
    """Return the first matching key from candidates found in row."""
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    if required:
        raise KeyError(f"None of these columns were found: {candidates}")
    return None


def load_boreas_gt_as_matrices(csv_path):
    """
    Load Boreas GT as 4x4 matrices and timestamps.

    Works for both lidar_poses.csv and camera_poses.csv by trying a set of
    likely column names. The trajectory is normalized so the first pose is identity.
    """
    timestamps = []
    matrices = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ts = float(get_first_present(row, ["GPSTime", "timestamp", "time"])) / 1e6

            x = float(get_first_present(row, ["x", "X", "easting"]))
            y = float(get_first_present(row, ["y", "Y", "northing"]))
            z = float(get_first_present(row, ["z", "Z", "altitude"]))

            roll = float(get_first_present(row, ["roll", "Roll"]))
            pitch = float(get_first_present(row, ["pitch", "Pitch"]))
            yaw = float(get_first_present(row, ["yaw", "Yaw", "heading", "Heading"]))

            T = np.eye(4)
            T[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
            T[:3, 3] = [x, y, z]

            timestamps.append(ts)
            matrices.append(T)

    if not matrices:
        raise RuntimeError(f"No poses loaded from GT file: {csv_path}")

    timestamps = np.array(timestamps)
    matrices = np.array(matrices)

    # Save initial rotation before normalizing (needed to convert back to ENU for plots)
    R0 = matrices[0][:3, :3].copy()

    # Normalize GT to start at identity
    T0_inv = np.linalg.inv(matrices[0])
    matrices = np.array([T0_inv @ T for T in matrices])

    return timestamps, matrices, R0


def associate_timestamps(ts_est, ts_gt, max_diff=0.05):
    """
    Match estimated timestamps to GT timestamps.
    Returns (est_indices, gt_indices) of matched pairs.
    """
    est_idx, gt_idx = [], []

    for i, ts in enumerate(ts_est):
        j = np.argmin(np.abs(ts_gt - ts))
        if np.abs(ts_gt[j] - ts) <= max_diff:
            est_idx.append(i)
            gt_idx.append(j)

    return np.array(est_idx), np.array(gt_idx)


def umeyama_alignment(src, dst, with_scale=True):
    """
    Find alignment such that:
        dst ≈ s * R @ src + t

    src, dst: (N, 3) arrays
    with_scale=True  -> Sim(3)
    with_scale=False -> SE(3)

    Returns:
        aligned_src, s, R, t
    """
    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for alignment.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = (src_c ** 2).sum() / n
    H = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(H)
    det_sign = np.linalg.det(U @ Vt)
    S = np.diag([1.0, 1.0, det_sign])

    R = U @ S @ Vt

    if with_scale:
        s = (D * np.diag(S)).sum() / sigma_src
    else:
        s = 1.0

    t = mu_dst - s * R @ mu_src
    aligned = (s * (R @ src.T)).T + t

    return aligned, s, R, t


def apply_transform_to_matrices(matrices, s, R, t):
    """
    Apply global alignment to estimated pose matrices.

    For Sim(3), only translation is scaled. Rotation is left-multiplied by R.
    """
    out = np.copy(matrices)

    for i in range(len(out)):
        out[i, :3, :3] = R @ out[i, :3, :3]
        out[i, :3, 3] = s * (R @ out[i, :3, 3]) + t

    return out


def compute_ate(est_xyz, gt_xyz, alignment="sim3"):
    """
    Absolute Trajectory Error after alignment.
    """
    with_scale = alignment.lower() == "sim3"
    aligned, s, R, t = umeyama_alignment(est_xyz, gt_xyz, with_scale=with_scale)
    errors = np.linalg.norm(aligned - gt_xyz, axis=1)

    stats = {
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
        "max": float(errors.max()),
    }

    return stats, aligned, errors, s, R, t


def compute_rpe(est_matrices, gt_matrices, delta=1):
    """
    Relative Pose Error over delta-frame pairs.
    Returns translation and rotation stats.
    """
    from scipy.spatial.transform import Rotation

    trans_errors = []
    rot_errors = []

    n = min(len(est_matrices), len(gt_matrices))
    for i in range(n - delta):
        Q_est = np.linalg.inv(est_matrices[i]) @ est_matrices[i + delta]
        Q_gt = np.linalg.inv(gt_matrices[i]) @ gt_matrices[i + delta]

        Q_err = np.linalg.inv(Q_gt) @ Q_est

        trans_errors.append(np.linalg.norm(Q_err[:3, 3]))

        R_err = Q_err[:3, :3]
        angle = Rotation.from_matrix(R_err).magnitude()
        rot_errors.append(np.degrees(angle))

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    def stats(arr):
        return {
            "rmse": float(np.sqrt((arr ** 2).mean())),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "max": float(arr.max()),
        }

    return stats(trans_errors), stats(rot_errors)


def print_results(label, algo, alignment, ate, rpe_t, rpe_r):
    print(f"\n{'=' * 60}")
    print(f"Results: {algo} | {label} | alignment={alignment}")
    print(f"{'=' * 60}")
    print(
        f"ATE (m)   RMSE: {ate['rmse']:.4f}  Mean: {ate['mean']:.4f}  "
        f"Median: {ate['median']:.4f}  Std: {ate['std']:.4f}  Max: {ate['max']:.4f}"
    )
    print(
        f"RPE-t (m) RMSE: {rpe_t['rmse']:.4f}  Mean: {rpe_t['mean']:.4f}  "
        f"Median: {rpe_t['median']:.4f}  Std: {rpe_t['std']:.4f}  Max: {rpe_t['max']:.4f}"
    )
    print(
        f"RPE-r (°) RMSE: {rpe_r['rmse']:.4f}  Mean: {rpe_r['mean']:.4f}  "
        f"Median: {rpe_r['median']:.4f}  Std: {rpe_r['std']:.4f}  Max: {rpe_r['max']:.4f}"
    )
    print(f"{'=' * 60}\n")


def save_results(label, algo, alignment, ate, rpe_t, rpe_r, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo}_{label}_metrics.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "type", "alignment", "rmse", "mean", "median", "std", "max"])
        writer.writerow(["ATE", "translation(m)", alignment, ate["rmse"], ate["mean"], ate["median"], ate["std"], ate["max"]])
        writer.writerow(["RPE-t", "translation(m)", alignment, rpe_t["rmse"], rpe_t["mean"], rpe_t["median"], rpe_t["std"], rpe_t["max"]])
        writer.writerow(["RPE-r", "rotation(deg)", alignment, rpe_r["rmse"], rpe_r["mean"], rpe_r["median"], rpe_r["std"], rpe_r["max"]])

    print(f"[SAVED] Metrics -> {out_path}")
    return out_path


def save_plots(label, algo, gt_xyz_full, gt_xyz_matched, aligned_xyz, ate_errors, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        gt_xyz_full[:, 0], gt_xyz_full[:, 1],
        color="lightgreen", linewidth=1.0, label="Ground Truth (full)"
    )
    ax.plot(
        gt_xyz_matched[:, 0], gt_xyz_matched[:, 1],
        "g-", linewidth=1.5, label="Ground Truth (matched)"
    )
    ax.plot(
        aligned_xyz[:, 0], aligned_xyz[:, 1],
        "b--", linewidth=1.0, label=f"{algo} (aligned)"
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Trajectory — {algo} | {label}")
    ax.legend()
    ax.set_aspect("equal")
    traj_path = out_dir / f"{algo}_{label}_trajectory.png"
    fig.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ate_errors, linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("ATE (m)")
    ax.set_title(f"ATE over time — {algo} | {label}")
    ate_path = out_dir / f"{algo}_{label}_ate.png"
    fig.savefig(ate_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate odometry output against Boreas GT.")
    parser.add_argument("--poses", required=True, help="Path to estimated poses (TUM format)")
    parser.add_argument("--gt", required=True, help="Path to Boreas GT CSV (camera_poses.csv or lidar_poses.csv)")
    parser.add_argument("--label", required=True, help="Run label, e.g. clear_baseline")
    parser.add_argument("--algo", default="unknown", help="Algorithm name, e.g. orbslam3")
    parser.add_argument("--results", default="results", help="Root dir for output metrics/plots")
    parser.add_argument("--max-diff", type=float, default=0.05, help="Max timestamp diff in seconds")
    parser.add_argument("--rpe-delta", type=int, default=1, help="Frame delta for RPE")
    parser.add_argument("--alignment", choices=["sim3", "se3"], default="sim3", help="Alignment model")
    parser.add_argument("--save-plots", action="store_true", help="Save trajectory and ATE plots")
    return parser.parse_args()


def main():
    args = parse_args()
    poses_path = Path(args.poses)
    gt_path = Path(args.gt)
    out_dir = Path(args.results) / args.label

    if not poses_path.exists():
        print(f"[ERROR] Poses file not found: {poses_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.exists():
        print(f"[ERROR] GT file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[EVAL] {args.algo} | {args.label}")
    print(f"  Poses     : {poses_path}")
    print(f"  GT        : {gt_path}")
    print(f"  Alignment : {args.alignment}")

    est_ts, est_xyz, est_quat = load_tum(poses_path)
    gt_ts, gt_matrices, R0 = load_boreas_gt_as_matrices(gt_path)
    gt_xyz = gt_matrices[:, :3, 3]

    print(f"  Estimated poses : {len(est_ts)}")
    print(f"  GT poses        : {len(gt_ts)}")

    est_idx, gt_idx = associate_timestamps(est_ts, gt_ts, max_diff=args.max_diff)
    print(f"  Matched pairs   : {len(est_idx)} (max_diff={args.max_diff}s)")

    if len(est_idx) < 10:
        print("[ERROR] Too few matched pairs — check timestamp alignment.", file=sys.stderr)
        sys.exit(1)

    est_xyz_m = est_xyz[est_idx]
    gt_xyz_m = gt_xyz[gt_idx]

    est_matrices_m = tum_quat_to_matrix(est_xyz_m, est_quat[est_idx])
    gt_matrices_m = gt_matrices[gt_idx]

    ate, aligned_xyz, ate_errors, s, R, t = compute_ate(est_xyz_m, gt_xyz_m, alignment=args.alignment)

    aligned_est_matrices = apply_transform_to_matrices(est_matrices_m, s, R, t)
    rpe_t, rpe_r = compute_rpe(aligned_est_matrices, gt_matrices_m, delta=args.rpe_delta)

    print_results(args.label, args.algo, args.alignment, ate, rpe_t, rpe_r)
    save_results(args.label, args.algo, args.alignment, ate, rpe_t, rpe_r, out_dir)

    if args.save_plots:
        # R0 @ camera_frame_pos recovers ENU displacement (east, north, alt)
        gt_enu = (R0 @ gt_xyz.T).T
        gt_m_enu = (R0 @ gt_xyz_m.T).T
        aligned_enu = (R0 @ aligned_xyz.T).T
        save_plots(args.label, args.algo, gt_enu, gt_m_enu, aligned_enu, ate_errors, out_dir)

    print("=== Evaluation complete ===")


if __name__ == "__main__":
    main()