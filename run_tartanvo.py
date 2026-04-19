#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from data_downloader import download_sequence, SEQUENCES

TARGET_W, TARGET_H = 640, 448


def load_intrinsics(calib_dir):
    p_path = Path(calib_dir) / "P_camera.txt"
    y_path = Path(calib_dir) / "camera0_intrinsics.yaml"

    if p_path.exists():
        P = np.loadtxt(p_path)
        return float(P[0, 0]), float(P[1, 1]), float(P[0, 2]), float(P[1, 2])

    if y_path.exists():
        with open(y_path) as f:
            data = yaml.safe_load(f)
        for key in ("projection_matrix", "camera_matrix"):
            if key in data:
                d = data[key]["data"]
                if len(d) == 12:
                    return float(d[0]), float(d[5]), float(d[2]), float(d[6])
                else:
                    return float(d[0]), float(d[4]), float(d[2]), float(d[5])

    print("[ERROR] No intrinsics file found in calib/", file=sys.stderr)
    sys.exit(1)


def make_intrinsics_layer(w, h, fx, fy, cx, cy):
    xs = (np.arange(w, dtype=np.float32) - cx) / fx
    ys = (np.arange(h, dtype=np.float32) - cy) / fy
    xs, ys = np.meshgrid(xs, ys)
    return np.stack([xs, ys], axis=0)


def load_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.resize(img, (TARGET_W, TARGET_H))
    return img.astype(np.float32) / 255.0


def motion_to_matrix(motion):
    T         = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(motion[3:]).as_matrix()
    T[:3,  3] = motion[:3]
    return T


def save_tum(poses, timestamps, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ts, pose in zip(timestamps, poses):
            t = pose[:3, 3]
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    print(f"\n[SAVED] TUM poses -> {out_path}", flush=True)


def run_tartanvo(img_files, calib_dir, out_path, model_path, tartanvo_path):
    sys.path.insert(0, str(tartanvo_path))
    from TartanVO import TartanVO

    print(f"\n[TartanVO] Loading model: {model_path}", flush=True)
    orig_cwd = os.getcwd()
    os.chdir(str(tartanvo_path))
    vo = TartanVO(model_name=model_path.name)
    os.chdir(orig_cwd)

    fx, fy, cx, cy = load_intrinsics(calib_dir)
    print(f"  Intrinsics — fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}", flush=True)

    # PWC-Net outputs flow at 1/4 resolution; intrinsic layer must match
    flow_w = TARGET_W // 4
    flow_h = TARGET_H // 4
    intrinsic_layer = make_intrinsics_layer(flow_w, flow_h, fx / 4, fy / 4, cx / 4, cy / 4)

    print(f"\n[TartanVO] Processing {len(img_files)} frames", flush=True)

    def to_tensor(arr):
        return torch.from_numpy(arr.transpose(2, 0, 1)[np.newaxis]).float()

    poses        = []
    timestamps   = []
    current_pose = np.eye(4)
    prev_img     = None

    for i, img_path in enumerate(img_files):
        ts_sec = int(img_path.stem) / 1e6
        img    = load_image(img_path)

        if img is None:
            print(f"  [WARN] Could not load {img_path.name}, skipping", flush=True)
            continue

        if prev_img is None:
            prev_img = img
            poses.append(current_pose.copy())
            timestamps.append(ts_sec)
            continue

        sample = {
            "img1":      to_tensor(prev_img),
            "img2":      to_tensor(img),
            "intrinsic": torch.from_numpy(intrinsic_layer[np.newaxis]).float(),
        }

        motions, _   = vo.test_batch(sample)
        T_rel        = motion_to_matrix(motions[0])
        current_pose = current_pose @ T_rel

        poses.append(current_pose.copy())
        timestamps.append(ts_sec)
        prev_img = img

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(img_files)} frames", flush=True)

    print(f"  Done. {len(poses)} poses estimated.", flush=True)
    save_tum(poses, timestamps, out_path)


def validate_sequence(seq_path):
    print(f"\n[VALIDATE] {seq_path}", flush=True)

    for sub in ["camera", "applanix", "calib"]:
        if not (seq_path / sub).exists():
            print(f"[ERROR] Missing folder: {seq_path / sub}", file=sys.stderr)
            sys.exit(1)
    print("  [OK] camera/, applanix/, calib/ present", flush=True)

    gt_path = seq_path / "applanix" / "camera_poses.csv"
    if not gt_path.exists():
        print("[WARN] camera_poses.csv not found", flush=True)

    img_files = sorted((seq_path / "camera").glob("*.png"))
    print(f"  Camera frames: {len(img_files)}", flush=True)

    if not img_files:
        print("[ERROR] No .png files in camera/", file=sys.stderr)
        sys.exit(1)

    return img_files


def parse_args():
    parser = argparse.ArgumentParser(description="Run TartanVO on a Boreas camera sequence.")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--data",     help="Path to sequence folder")
    grp.add_argument("--sequence", help="Weather label (clear/snow/rain)")

    parser.add_argument("--download",      action="store_true",
                        help="Download data before running (requires --sequence)")
    parser.add_argument("--output",        default="boreas_data",
                        help="Root dir for downloaded data (default: boreas_data/)")
    parser.add_argument("--label",         default=None,
                        help="Weather label for output folder; inferred from --sequence if omitted")
    parser.add_argument("--tartanvo-path", required=True,
                        help="Path to cloned TartanVO repo")
    parser.add_argument("--model",         required=True,
                        help="Path to tartanvo_1914.pkl model file")
    parser.add_argument("--results",       default="results",
                        help="Root directory for output poses (default: results/)")
    parser.add_argument("--max-scans",     type=int, default=None,
                        help="Process only first N frames (for quick testing)")
    return parser.parse_args()


def main():
    args          = parse_args()
    tartanvo_path = Path(args.tartanvo_path).expanduser()
    model_path    = Path(args.model).expanduser()

    if args.sequence:
        seq_key    = args.sequence.lower()
        seq_name   = SEQUENCES.get(seq_key, args.sequence)
        label      = args.label or seq_key
        output_dir = Path(args.output).expanduser()
        seq_path   = output_dir / seq_name
        if args.download:
            print("[DOWNLOAD] Starting camera data download ...", flush=True)
            download_sequence(seq_name, output_dir,
                              modalities=["camera", "applanix", "calib"],
                              max_files=args.max_scans)
    else:
        seq_path = Path(args.data).expanduser()
        label    = args.label or seq_path.name

    calib_dir = seq_path / "calib"
    out_path  = Path(args.results) / label / "tartanvo_poses.txt"

    print("=== TartanVO Pipeline ===", flush=True)
    print(f"Sequence    : {seq_path}", flush=True)
    print(f"Label       : {label}", flush=True)
    print(f"TartanVO    : {tartanvo_path}", flush=True)
    print(f"Model       : {model_path}", flush=True)
    print(f"Output      : {out_path}", flush=True)

    if not tartanvo_path.exists():
        print(f"[ERROR] TartanVO path not found: {tartanvo_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    img_files = validate_sequence(seq_path)
    if args.max_scans:
        img_files = img_files[:args.max_scans]
        print(f"[TEST] Limited to first {args.max_scans} frames", flush=True)

    run_tartanvo(img_files, calib_dir, out_path, model_path, tartanvo_path)

    print("\n=== Done ===")
    print(f"Poses saved to: {out_path}")
    print(f"Next: python evaluator.py --poses {out_path} "
          f"--gt {seq_path}/applanix/camera_poses.csv "
          f"--label {label} --algo tartanvo --save-plots")


if __name__ == "__main__":
    main()
