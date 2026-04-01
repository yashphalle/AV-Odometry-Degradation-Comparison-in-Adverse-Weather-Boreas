#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

# Known sequences by weather label
SEQUENCES = {
    "clear": "boreas-2020-11-26-13-58",
    "snow":  "boreas-2021-01-26-10-59",
    "rain":  "boreas-2021-08-05-13-34",
}

S3_BASE = "s3://boreas"
DEFAULT_MODALITIES = ["lidar", "applanix", "calib"]


def parse_args():
    parser = argparse.ArgumentParser(description="Download Boreas dataset sequences")
    parser.add_argument(
        "--sequences", nargs="+", required=True,
        help="Weather labels (clear/snow/rain) or full sequence names"
    )
    parser.add_argument(
        "--output", type=str, default="~/boreas_data",
        help="Local output directory (default: ~/boreas_data)"
    )
    parser.add_argument(
        "--modalities", nargs="+", default=DEFAULT_MODALITIES,
        choices=["lidar", "applanix", "calib", "camera"],
        help="Which data folders to download (default: lidar applanix calib)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    return parser.parse_args()


def resolve_sequence(name: str) -> str:
    """Resolve label (clear/snow/rain) or pass through full sequence name."""
    return SEQUENCES.get(name.lower(), name)


def sync_s3(s3_path: str, local_path: Path, dry_run: bool):
    local_path.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", s3_path, str(local_path), "--no-sign-request"]
    print(f"[SYNC] {s3_path} -> {local_path}")
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()
    output_dir = Path(args.output).expanduser()

    for seq_input in args.sequences:
        seq_name = resolve_sequence(seq_input)
        print(f"\n=== Downloading: {seq_name} ===")
        for modality in args.modalities:
            s3_path = f"{S3_BASE}/{seq_name}/{modality}/"
            local_path = output_dir / seq_name / modality
            sync_s3(s3_path, local_path, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()