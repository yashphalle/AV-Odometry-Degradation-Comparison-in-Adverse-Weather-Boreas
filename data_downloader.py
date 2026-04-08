#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

SEQUENCES = {
    "clear": "boreas-2020-11-26-13-58",
    "snow":  "boreas-2021-01-26-11-22",
    "rain":  "boreas-2021-08-05-13-34",
}

S3_BUCKET = "boreas"
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
        help="Print what would be downloaded without downloading"
    )
    return parser.parse_args()


def resolve_sequence(name):
    return SEQUENCES.get(name.lower(), name)


def download_sequence(seq_name, output_dir, modalities=None, dry_run=False):
    if modalities is None:
        modalities = DEFAULT_MODALITIES

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    for modality in modalities:
        prefix = f"{seq_name}/{modality}/"
        local_base = output_dir / seq_name / modality
        local_base.mkdir(parents=True, exist_ok=True)

        print(f"[DOWNLOAD] s3://{S3_BUCKET}/{prefix} -> {local_base}")

        if dry_run:
            print(f"  [DRY RUN] Would download s3://{S3_BUCKET}/{prefix}")
            continue

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


def main():
    args = parse_args()
    output_dir = Path(args.output).expanduser()

    for seq_input in args.sequences:
        seq_name = resolve_sequence(seq_input)
        print(f"\n=== Downloading: {seq_name} ===")
        download_sequence(seq_name, output_dir, modalities=args.modalities, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()