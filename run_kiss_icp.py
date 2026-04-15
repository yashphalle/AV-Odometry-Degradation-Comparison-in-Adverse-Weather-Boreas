#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

SEQUENCES = {
    "clear": "boreas-2021-04-08-12-44",
    "snow":  "boreas-2021-01-26-11-22",
    "rain":  "boreas-2021-07-20-17-33",
}

S3_BUCKET = "boreas"
MODALITIES = ["lidar", "applanix", "calib"]


def download_sequence(seq_name, output_dir):
    """Download lidar, applanix, calib from S3 using boto3. Resume-safe."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    for modality in MODALITIES:
        prefix = f"{seq_name}/{modality}/"
        local_base = output_dir / seq_name / modality
        local_base.mkdir(parents=True, exist_ok=True)

        print(f"[DOWNLOAD] s3://{S3_BUCKET}/{prefix} -> {local_base}")

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)

        files_downloaded = 0
        files_skipped = 0

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                s3_size = obj["Size"]
                filename = Path(key).name
                local_path = local_base / filename

                if local_path.exists() and local_path.stat().st_size == s3_size:
                    files_skipped += 1
                    continue

                s3.download_file(S3_BUCKET, key, str(local_path))
                files_downloaded += 1

                if files_downloaded % 100 == 0:
                    print(f"  Downloaded {files_downloaded} files...")

        print(f"  Done — {files_downloaded} new, {files_skipped} already present")



def validate_sequence(seq_path):
    """Validate downloaded sequence. Returns bin file list. Exits on errors."""
    print(f"\n[VALIDATE] {seq_path}")

    for modality in MODALITIES:
        folder = seq_path / modality
        if not folder.exists():
            print(f"[ERROR] Missing folder: {folder}", file=sys.stderr)
            sys.exit(1)
    print("  [OK] lidar/, applanix/, calib/ present")

    gt_path = seq_path / "applanix" / "lidar_poses.csv"
    if not gt_path.exists():
        print("[ERROR] lidar_poses.csv not found — test-set sequence with withheld GT.", file=sys.stderr)
        sys.exit(1)
    print("  [OK] lidar_poses.csv present")

    bin_files = sorted((seq_path / "lidar").glob("*.bin"))
    bin_count = len(bin_files)

    with open(gt_path) as f:
        gt_rows = sum(1 for line in f) - 1  # subtract header

    print(f"  LiDAR scans : {bin_count}")
    print(f"  GT pose rows: {gt_rows}")

    if bin_count == 0:
        print("[ERROR] No .bin files found in lidar/", file=sys.stderr)
        sys.exit(1)

    if bin_count != gt_rows:
        print(f"  [WARN] Mismatch — will use first {min(bin_count, gt_rows)} frames")

    return bin_files



def load_bin(bin_file, min_range=0.5, max_range=100.0):
    """
    Load Boreas .bin file.
    Format: Nx6 float32 (x, y, z, intensity, ring, time)
    Returns xyz as float64 and yaw-based timestamps for deskewing.
    """
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 6)
    xyz = points[:, :3].astype(np.float64)

    # Filter invalid points
    ranges = np.linalg.norm(xyz, axis=1)
    valid = (
        np.isfinite(xyz).all(axis=1) &
        (ranges > min_range) &
        (ranges < max_range)
    )
    xyz = xyz[valid]

    # Yaw-based per-point timestamps for deskewing (from boreas.py dataloader)
    x = xyz[:, 0]
    y = xyz[:, 1]
    yaw = -np.arctan2(y, x)
    timestamps = 0.5 * (yaw / np.pi + 1.0)

    return xyz, timestamps


def run_kiss_icp(bin_files, out_path):
    """Run KISS-ICP scan-by-scan using KissICP class and save poses in TUM format."""
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    print(f"\n[KISS-ICP] Processing {len(bin_files)} scans")

    config = KISSConfig()
    config.mapping.voxel_size = 1.0  # meters — suitable for outdoor driving LiDAR
    odometry = KissICP(config=config)

    poses = []
    timestamps = []

    for i, bin_file in enumerate(bin_files):
        xyz, per_point_ts = load_bin(bin_file)

        if len(xyz) == 0:
            print(f"  [WARN] Empty scan at {bin_file.name}, skipping")
            continue

        # Scan timestamp from filename — Boreas uses microseconds
        ts_sec = int(bin_file.stem) / 1e6

        odometry.register_frame(xyz, per_point_ts)

        poses.append(odometry.last_pose.copy())
        timestamps.append(ts_sec)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(bin_files)} scans")

    print(f"  Done. {len(poses)} poses estimated.")
    save_tum(poses, timestamps, out_path)


def save_tum(poses, timestamps, out_path):
    """Save 4x4 pose matrices as TUM trajectory format."""
    from scipy.spatial.transform import Rotation

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for ts, pose in zip(timestamps, poses):
            t = pose[:3, 3]
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()  # [qx, qy, qz, qw]
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    print(f"\n[SAVED] TUM poses -> {out_path}")


def resolve_sequence(name):
    """Return (weather_label, sequence_name)."""
    if name.lower() in SEQUENCES:
        return name.lower(), SEQUENCES[name.lower()]
    for label, seq in SEQUENCES.items():
        if seq == name:
            return label, seq
    return "unknown", name


def parse_args():
    parser = argparse.ArgumentParser(description="Download, validate, and run KISS-ICP on a Boreas sequence.")
    parser.add_argument("--sequence", required=True,
                        help="Weather label (clear/snow/rain) or full sequence name")
    parser.add_argument("--output", type=str, default="~/boreas_data",
                        help="Root directory for downloaded data (default: ~/boreas_data)")
    parser.add_argument("--results", type=str, default="results",
                        help="Root directory for output poses (default: results/)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download — use if data already exists locally")
    return parser.parse_args()


def main():
    args = parse_args()
    label, seq_name = resolve_sequence(args.sequence)
    output_dir = Path(args.output).expanduser()
    seq_path = output_dir / seq_name
    out_path = Path(args.results) / label / "kiss_icp_poses.txt"

    print("=== KISS-ICP Pipeline ===")
    print(f"Sequence : {seq_name}")
    print(f"Label    : {label}")
    print(f"Data dir : {seq_path}")
    print(f"Output   : {out_path}")

    if not args.skip_download:
        download_sequence(seq_name, output_dir)
    else:
        print("\n[DOWNLOAD] Skipped.")

    bin_files = validate_sequence(seq_path)
    run_kiss_icp(bin_files, out_path)

    print("\n=== Done ===")
    print(f"Poses saved to: {out_path}")
    print(f"Next: run evaluate.py --sequence {label} --poses {out_path}")


if __name__ == "__main__":
    main()