#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SEQUENCES = {
    "clear": "boreas-2021-04-08-12-44",
    "snow":  "boreas-2021-01-26-11-22",
    "rain":  "boreas-2021-07-20-17-33",
}

S3_BUCKET  = "boreas"
MODALITIES = ["lidar", "applanix", "calib"]


def download_sequence(seq_name, output_dir):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    for modality in MODALITIES:
        prefix     = f"{seq_name}/{modality}/"
        local_base = output_dir / seq_name / modality
        local_base.mkdir(parents=True, exist_ok=True)
        print(f"[DOWNLOAD] s3://{S3_BUCKET}/{prefix} -> {local_base}")

        paginator = s3.get_paginator("list_objects_v2")
        downloaded, skipped = 0, 0

        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key        = obj["Key"]
                s3_size    = obj["Size"]
                local_path = local_base / Path(key).name

                if local_path.exists() and local_path.stat().st_size == s3_size:
                    skipped += 1
                    continue

                s3.download_file(S3_BUCKET, key, str(local_path))
                downloaded += 1
                if downloaded % 100 == 0:
                    print(f"  Downloaded {downloaded} files...")

        print(f"  Done — {downloaded} new, {skipped} already present")


def validate_sequence(seq_path):
    print(f"\n[VALIDATE] {seq_path}")

    for modality in MODALITIES:
        if not (seq_path / modality).exists():
            print(f"[ERROR] Missing folder: {seq_path / modality}", file=sys.stderr)
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


def load_bin(bin_file, min_range=0.5, max_range=100.0):
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 6)
    xyz    = points[:, :3].astype(np.float64)

    ranges = np.linalg.norm(xyz, axis=1)
    valid  = (
        np.isfinite(xyz).all(axis=1) &
        (ranges > min_range) &
        (ranges < max_range)
    )
    return xyz[valid]


def save_tum(poses, timestamps, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ts, pose in zip(timestamps, poses):
            t = pose[:3, 3]
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    print(f"\n[SAVED] TUM poses -> {out_path}")


def run_gicp(bin_files, out_path):
    import small_gicp

    print(f"\n[GICP] Processing {len(bin_files)} scans", flush=True)

    poses        = []
    timestamps   = []
    current_pose = np.eye(4)
    prev_xyz     = None

    for i, bin_file in enumerate(bin_files):
        xyz = load_bin(bin_file)

        if len(xyz) == 0:
            print(f"  [WARN] Empty scan at {bin_file.name}, skipping", flush=True)
            continue

        ts_sec = int(bin_file.stem) / 1e6

        if prev_xyz is None:
            prev_xyz = xyz
            poses.append(current_pose.copy())
            timestamps.append(ts_sec)
            continue

        result = small_gicp.align(
            prev_xyz,
            xyz,
            downsampling_resolution=0.5,
            registration_type="GICP",
            max_correspondence_distance=2.0,
            num_threads=4,
        )

        T_rel        = result.T_target_source
        current_pose = current_pose @ T_rel

        poses.append(current_pose.copy())
        timestamps.append(ts_sec)

        prev_xyz = xyz

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(bin_files)} scans", flush=True)

    print(f"  Done. {len(poses)} poses estimated.", flush=True)
    save_tum(poses, timestamps, out_path)


def resolve_sequence(name):
    if name.lower() in SEQUENCES:
        return name.lower(), SEQUENCES[name.lower()]
    for label, seq in SEQUENCES.items():
        if seq == name:
            return label, seq
    return "unknown", name


def parse_args():
    parser = argparse.ArgumentParser(description="Run GICP on a Boreas sequence.")
    parser.add_argument("--sequence",      required=True,
                        help="Weather label (clear/snow/rain) or full sequence name")
    parser.add_argument("--output",        default="~/boreas_data",
                        help="Root directory for data (default: ~/boreas_data)")
    parser.add_argument("--results",       default="results",
                        help="Root directory for output poses (default: results/)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download — use if data already exists locally")
    parser.add_argument("--max-scans",     type=int, default=None,
                        help="Process only first N scans (for quick testing)")
    return parser.parse_args()


def main():
    args            = parse_args()
    label, seq_name = resolve_sequence(args.sequence)
    output_dir      = Path(args.output).expanduser()
    seq_path        = output_dir / seq_name
    out_path        = Path(args.results) / label / "gicp_poses.txt"

    print("=== GICP Pipeline ===")
    print(f"Sequence : {seq_name}")
    print(f"Label    : {label}")
    print(f"Data dir : {seq_path}")
    print(f"Output   : {out_path}")

    if not args.skip_download:
        download_sequence(seq_name, output_dir)
    else:
        print("\n[DOWNLOAD] Skipped.")

    bin_files = validate_sequence(seq_path)
    if args.max_scans:
        bin_files = bin_files[:args.max_scans]
        print(f"[TEST] Limited to first {args.max_scans} scans", flush=True)

    run_gicp(bin_files, out_path)

    print("\n=== Done ===")
    print(f"Poses saved to: {out_path}")
    print(f"Next: python evaluator.py --poses {out_path} "
          f"--gt {seq_path}/applanix/lidar_poses.csv "
          f"--label {label} --algo gicp --save-plots")


if __name__ == "__main__":
    main()
