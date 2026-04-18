#!/usr/bin/env python3
from pathlib import Path

BASE = Path(__file__).parent
SEQ = BASE / "boreas_data/boreas-2021-04-08-12-44"
CAM = SEQ / "camera"

OUT = BASE / "results/clear/orbslam3/image_list.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Recursive search (important)
imgs = sorted(
    list(CAM.rglob("*.png")) +
    list(CAM.rglob("*.jpg"))
)

assert len(imgs) > 0, "No images found"

with open(OUT, "w") as f:
    for img in imgs:
        try:
            ts = int(img.stem) / 1e6
        except:
            continue
        f.write(f"{ts:.6f} {img.resolve()}\n")

print(f"[DONE] Image list created: {OUT}")
print(f"[INFO] Frames: {len(imgs)}")
print(f"[INFO] Example: {imgs[0]}")