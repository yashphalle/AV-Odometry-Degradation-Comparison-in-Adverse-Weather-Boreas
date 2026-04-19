#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

S3_BUCKET = "boreas"
SEQUENCES = {
    "clear": "boreas-2021-04-08-12-44",
    "snow":  "boreas-2021-01-26-11-22",
    "rain":  "boreas-2021-07-20-17-33",
}


def download_sequence(seq_name, output_dir, max_scans=None):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    seq_dir = output_dir / seq_name

    lidar_dir = seq_dir / "lidar"
    lidar_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] Listing lidar scans ...", flush=True)

    paginator = s3.get_paginator("list_objects_v2")
    all_keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{seq_name}/lidar/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".bin"):
                all_keys.append((obj["Key"], obj["Size"]))

    all_keys.sort(key=lambda x: x[0])
    if max_scans:
        all_keys = all_keys[:max_scans]
    print(f"  Downloading {len(all_keys)} scans ...", flush=True)

    for n, (key, s3_size) in enumerate(all_keys):
        local = lidar_dir / Path(key).name
        if local.exists() and local.stat().st_size == s3_size:
            continue
        s3.download_file(S3_BUCKET, key, str(local))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(all_keys)} scans downloaded ...", flush=True)
    print(f"  Lidar done.", flush=True)

    for modality in ["applanix", "calib"]:
        mod_dir = seq_dir / modality
        mod_dir.mkdir(parents=True, exist_ok=True)
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{seq_name}/{modality}/"):
            for obj in page.get("Contents", []):
                key     = obj["Key"]
                s3_size = obj["Size"]
                local   = mod_dir / Path(key).name
                if local.exists() and local.stat().st_size == s3_size:
                    continue
                s3.download_file(S3_BUCKET, key, str(local))
        print(f"  {modality} done.", flush=True)

    return seq_dir


def load_bin(bin_file, min_range=0.5, max_range=100.0):
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 6)
    xyz    = points[:, :3].astype(np.float64)
    rings  = points[:, 4].astype(np.int32)

    ranges = np.linalg.norm(xyz, axis=1)
    valid  = (
        np.isfinite(xyz).all(axis=1) &
        (ranges > min_range) &
        (ranges < max_range)
    )
    return xyz[valid], rings[valid]


def extract_features(xyz, rings, n_edge=20, n_planar=80, edge_thresh=0.1):
    half = 5
    curvature = np.zeros(len(xyz))
    edge_idx, planar_idx = [], []

    for ring_id in np.unique(rings):
        idx = np.where(rings == ring_id)[0]
        if len(idx) < 2 * half + 1:
            continue

        pts  = xyz[idx]
        M    = len(pts)
        cs   = np.cumsum(pts, axis=0)

        i_vals = np.arange(half, M - half)
        r_vals = i_vals + half
        l_vals = i_vals - half - 1

        win = cs[r_vals].copy()
        win[l_vals >= 0] -= cs[l_vals[l_vals >= 0]]

        diff = win - (2 * half + 1) * pts[i_vals]
        c    = np.sum(diff ** 2, axis=1)

        curvature[idx[i_vals]] = c

        order = np.argsort(c)
        planar_idx.extend(idx[i_vals[order[:n_planar]]])
        edge_idx.extend(idx[i_vals[order[-(n_edge):]]][c[order[-n_edge:]] > edge_thresh])

    edge_pts   = xyz[edge_idx]   if edge_idx   else np.empty((0, 3))
    planar_pts = xyz[planar_idx] if planar_idx else np.empty((0, 3))
    return edge_pts, planar_pts


def build_kdtree(pts):
    from scipy.spatial import KDTree
    return KDTree(pts), pts


def knn(tree, pts_ref, query, k):
    _, idx = tree.query(query, k=k)
    return pts_ref[np.asarray(idx)]


def point_to_line_residuals(pose_vec, src, line_a, line_b):
    R = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
    t = pose_vec[3:]
    transformed = (R @ src.T).T + t

    ab      = line_b - line_a
    ap      = transformed - line_a
    cross   = np.cross(ap, ab)
    ab_norm = np.linalg.norm(ab, axis=1, keepdims=True) + 1e-9
    return np.linalg.norm(cross, axis=1) / ab_norm.squeeze()


def point_to_plane_residuals(pose_vec, src, plane_pts):
    R = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
    t = pose_vec[3:]
    transformed = (R @ src.T).T + t

    v1      = plane_pts[:, 1] - plane_pts[:, 0]
    v2      = plane_pts[:, 2] - plane_pts[:, 0]
    normals = np.cross(v1, v2)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    normals = normals / norms

    d = plane_pts[:, 0]
    return np.einsum('ij,ij->i', transformed - d, normals)


def optimise_pose(pose_vec,
                  edge_src, edge_tree, edge_ref,
                  planar_src, planar_tree, planar_ref,
                  max_correspondence=2.0):
    R0 = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
    t0 = pose_vec[3:]

    edge_src_fixed, la_fixed, lb_fixed = [], [], []
    if len(edge_src) > 0 and len(edge_ref) > 5:
        tf = (R0 @ edge_src.T).T + t0
        for i, p in enumerate(tf):
            nbrs = knn(edge_tree, edge_ref, p, 2)
            if np.linalg.norm(p - nbrs[0]) < max_correspondence:
                edge_src_fixed.append(edge_src[i])
                la_fixed.append(nbrs[0])
                lb_fixed.append(nbrs[1])

    planar_src_fixed, triplets_fixed = [], []
    if len(planar_src) > 0 and len(planar_ref) > 5:
        tf = (R0 @ planar_src.T).T + t0
        for i, p in enumerate(tf):
            nbrs = knn(planar_tree, planar_ref, p, 3)
            if np.linalg.norm(p - nbrs[0]) < max_correspondence:
                planar_src_fixed.append(planar_src[i])
                triplets_fixed.append(nbrs)

    if len(edge_src_fixed) + len(planar_src_fixed) < 6:
        return pose_vec

    has_edge   = len(edge_src_fixed)   > 0
    has_planar = len(planar_src_fixed) > 0

    e_src = np.array(edge_src_fixed)   if has_edge   else None
    la    = np.array(la_fixed)         if has_edge   else None
    lb    = np.array(lb_fixed)         if has_edge   else None
    p_src = np.array(planar_src_fixed) if has_planar else None
    trips = np.array(triplets_fixed)   if has_planar else None

    def residuals(pv):
        res = []
        if has_edge:
            res.append(point_to_line_residuals(pv, e_src, la, lb))
        if has_planar:
            res.append(point_to_plane_residuals(pv, p_src, trips))
        return np.concatenate(res)

    result = least_squares(residuals, pose_vec, method='lm', max_nfev=20)
    return result.x


class VoxelMap:
    def __init__(self, voxel_size=2.0, max_pts_per_voxel=20):
        self.voxel_size = voxel_size
        self.max_pts    = max_pts_per_voxel
        self._map       = {}

    def insert(self, pts):
        keys = np.floor(pts / self.voxel_size).astype(int)
        for k, p in zip(map(tuple, keys), pts):
            bucket = self._map.setdefault(k, [])
            if len(bucket) < self.max_pts:
                bucket.append(p)

    def to_array(self):
        if not self._map:
            return np.empty((0, 3))
        return np.vstack([np.array(v) for v in self._map.values()])


def save_tum(poses, timestamps, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ts, pose in zip(timestamps, poses):
            t = pose[:3, 3]
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    print(f"\n[SAVED] TUM poses -> {out_path}")


def run_loam(bin_files, out_path, map_update_every=5):
    print(f"\n[LOAM] Processing {len(bin_files)} scans", flush=True)

    poses        = []
    timestamps   = []
    current_pose = np.eye(4)
    pose_vec     = np.zeros(6)

    prev_edge_tree   = None
    prev_edge_ref    = None
    prev_planar_tree = None
    prev_planar_ref  = None

    edge_map   = VoxelMap(voxel_size=2.0)
    planar_map = VoxelMap(voxel_size=2.0)

    for i, bin_file in enumerate(bin_files):
        if i == 0:
            print(f"[SCAN 0] Loading first scan: {bin_file.name} ...", flush=True)

        xyz, rings = load_bin(bin_file)

        if i == 0:
            print(f"  Points loaded: {len(xyz)}  Rings: {np.unique(rings).size}", flush=True)

        if len(xyz) < 50:
            print(f"  [WARN] Sparse scan at {bin_file.name}, skipping", flush=True)
            continue

        ts_sec = int(bin_file.stem) / 1e6

        if i == 0:
            print(f"  Extracting features ...", flush=True)

        edge_pts, planar_pts = extract_features(xyz, rings)

        if i == 0:
            print(f"  Edge pts: {len(edge_pts)}  Planar pts: {len(planar_pts)}", flush=True)

        if i == 0:
            if len(edge_pts)   > 0: edge_map.insert(edge_pts)
            if len(planar_pts) > 0: planar_map.insert(planar_pts)

            prev_edge_tree, prev_edge_ref     = (build_kdtree(edge_pts)
                                                  if len(edge_pts) > 0
                                                  else (None, np.empty((0, 3))))
            prev_planar_tree, prev_planar_ref = (build_kdtree(planar_pts)
                                                  if len(planar_pts) > 0
                                                  else (None, np.empty((0, 3))))

            poses.append(current_pose.copy())
            timestamps.append(ts_sec)
            print(f"  Frame 0 done — entering main loop ...", flush=True)
            continue

        if prev_edge_tree is not None and prev_planar_tree is not None:
            pose_vec = optimise_pose(
                pose_vec,
                edge_pts,   prev_edge_tree,   prev_edge_ref,
                planar_pts, prev_planar_tree, prev_planar_ref,
            )

        T_rel         = np.eye(4)
        T_rel[:3, :3] = Rotation.from_rotvec(pose_vec[:3]).as_matrix()
        T_rel[:3,  3] = pose_vec[3:]

        if i % map_update_every == 0:
            map_edge_arr   = edge_map.to_array()
            map_planar_arr = planar_map.to_array()

            if len(map_edge_arr) > 5 and len(map_planar_arr) > 5:
                map_edge_tree,   _ = build_kdtree(map_edge_arr)
                map_planar_tree, _ = build_kdtree(map_planar_arr)

                T_world_guess = current_pose @ T_rel
                R_cur = T_world_guess[:3, :3]
                t_cur = T_world_guess[:3, 3]

                edge_world   = (R_cur @ edge_pts.T).T   + t_cur if len(edge_pts)   > 0 else edge_pts
                planar_world = (R_cur @ planar_pts.T).T + t_cur if len(planar_pts) > 0 else planar_pts

                refined_vec = optimise_pose(
                    np.concatenate([Rotation.from_matrix(R_cur).as_rotvec(), t_cur]),
                    edge_world,   map_edge_tree,   map_edge_arr,
                    planar_world, map_planar_tree, map_planar_arr,
                )

                T_world_refined         = np.eye(4)
                T_world_refined[:3, :3] = Rotation.from_rotvec(refined_vec[:3]).as_matrix()
                T_world_refined[:3,  3] = refined_vec[3:]
                T_rel = np.linalg.inv(current_pose) @ T_world_refined

        current_pose = current_pose @ T_rel

        R_g = current_pose[:3, :3]
        t_g = current_pose[:3, 3]
        if len(edge_pts)   > 0: edge_map.insert((R_g @ edge_pts.T).T + t_g)
        if len(planar_pts) > 0: planar_map.insert((R_g @ planar_pts.T).T + t_g)

        prev_edge_tree, prev_edge_ref     = (build_kdtree(edge_pts)
                                              if len(edge_pts) > 0
                                              else (None, np.empty((0, 3))))
        prev_planar_tree, prev_planar_ref = (build_kdtree(planar_pts)
                                              if len(planar_pts) > 0
                                              else (None, np.empty((0, 3))))

        pose_vec = np.concatenate([
            Rotation.from_matrix(T_rel[:3, :3]).as_rotvec(),
            T_rel[:3, 3]
        ])

        poses.append(current_pose.copy())
        timestamps.append(ts_sec)

        interval = 10 if i < 100 else 100
        if (i + 1) % interval == 0:
            print(f"  Processed {i + 1}/{len(bin_files)} scans", flush=True)

    print(f"  Done. {len(poses)} poses estimated.", flush=True)
    save_tum(poses, timestamps, out_path)


def validate_sequence(seq_path):
    print(f"\n[VALIDATE] {seq_path}")

    for sub in ["lidar", "applanix", "calib"]:
        if not (seq_path / sub).exists():
            print(f"[ERROR] Missing folder: {seq_path / sub}", file=sys.stderr)
            sys.exit(1)
    print("  [OK] lidar/, applanix/, calib/ present")

    gt_path = seq_path / "applanix" / "lidar_poses.csv"
    if not gt_path.exists():
        print("[ERROR] lidar_poses.csv not found.", file=sys.stderr)
        sys.exit(1)
    print("  [OK] lidar_poses.csv present")

    bin_files = sorted((seq_path / "lidar").glob("*.bin"))
    print(f"  LiDAR scans: {len(bin_files)}")

    if not bin_files:
        print("[ERROR] No .bin files in lidar/", file=sys.stderr)
        sys.exit(1)

    return bin_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LOAM on a Boreas sequence."
    )
    parser.add_argument("--sequence",  default=None,
                        help="Weather label (clear/snow/rain) — used with --download")
    parser.add_argument("--data",      default=None,
                        help="Path to already-downloaded sequence folder")
    parser.add_argument("--download",  action="store_true",
                        help="Download data from S3 before running (requires --sequence)")
    parser.add_argument("--output",    default="boreas_data",
                        help="Root dir for downloaded data (default: boreas_data/)")
    parser.add_argument("--label",     default=None,
                        help="Weather label for output folder (clear/snow/rain)")
    parser.add_argument("--results",   default="results",
                        help="Root directory for output poses (default: results/)")
    parser.add_argument("--map-update-every", type=int, default=5,
                        help="Run scan-to-map refinement every N frames (default: 5)")
    parser.add_argument("--max-scans", type=int, default=None,
                        help="Process only first N scans (for quick testing)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.download:
        if not args.sequence:
            print("[ERROR] --download requires --sequence (clear/snow/rain)", file=sys.stderr)
            sys.exit(1)
        seq_name   = SEQUENCES.get(args.sequence.lower(), args.sequence)
        label      = args.label or args.sequence.lower()
        output_dir = Path(args.output).expanduser()
        print(f"[DOWNLOAD] Fetching {seq_name} (max_scans={args.max_scans}) ...", flush=True)
        seq_path = download_sequence(seq_name, output_dir, max_scans=args.max_scans)
    elif args.data:
        seq_path = Path(args.data).expanduser()
        label    = args.label or seq_path.name
    else:
        print("[ERROR] Provide --data <path> or use --download --sequence <label>", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.results) / label / "loam_poses.txt"

    print("=== LOAM Pipeline ===", flush=True)
    print(f"Sequence : {seq_path}", flush=True)
    print(f"Label    : {label}", flush=True)
    print(f"Output   : {out_path}", flush=True)

    bin_files = validate_sequence(seq_path)
    if args.max_scans:
        bin_files = bin_files[:args.max_scans]
        print(f"[TEST] Limited to first {args.max_scans} scans", flush=True)
    run_loam(bin_files, out_path, map_update_every=args.map_update_every)

    print("\n=== Done ===")
    print(f"Poses saved to: {out_path}")
    print(f"Next: python evaluator.py --poses {out_path} "
          f"--gt {seq_path}/applanix/lidar_poses.csv "
          f"--label {label} --algo loam --save-plots")


if __name__ == "__main__":
    main()
