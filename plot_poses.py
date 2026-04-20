#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_boreas_xy(csv_path):
    xs, ys = [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["easting"])
            y = float(row["northing"])
            xs.append(x)
            ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    # Normalize so trajectory starts at origin
    xs = xs - xs[0]
    ys = ys - ys[0]

    return xs, ys


def main():
    parser = argparse.ArgumentParser(description="Plot Boreas GT trajectory from camera_poses.csv or lidar_poses.csv")
    parser.add_argument("--gt", required=True, help="Path to Boreas GT CSV")
    parser.add_argument("--save", default=None, help="Optional output image path")
    parser.add_argument("--title", default=None, help="Optional plot title")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    xs, ys = load_boreas_xy(gt_path)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, "g-", linewidth=1.5, label=gt_path.name)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(args.title if args.title else f"Trajectory — {gt_path.name}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()