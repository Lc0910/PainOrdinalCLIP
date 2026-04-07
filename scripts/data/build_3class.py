"""Build 3-class BioVid data lists.

User-specified label scheme:
    {0, 1, 2} -> class 0  (no pain + low pain)
    {3}       -> class 1  (moderate pain)
    {4}       -> class 2  (severe pain)

This grouping is intentionally imbalanced — class 0 has roughly 3x the samples
of class 1 and class 2. The training pipeline should compensate via class
weights (see configs/base_cfgs/data_cfg/datasets/biovid/biovid_3class.yaml).

Output files (written next to the originals):
  data/biovid/train_3class.txt   label in {0, 1, 2}
  data/biovid/test_3class.txt    label in {0, 1, 2}

Usage:
    python scripts/data/build_3class.py
    python scripts/data/build_3class.py --min_frame 50  # default
    python scripts/data/build_3class.py --min_frame 0   # no frame skip
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional


# Original BioVid label -> 3-class label
LABEL_MAP: dict[int, Optional[int]] = {
    0: 0,    # BLN  -> class 0 (no pain)
    1: 0,    # PA1  -> class 0 (low pain)
    2: 0,    # PA2  -> class 0 (low pain)
    3: 1,    # PA3  -> class 1 (moderate)
    4: 2,    # PA4  -> class 2 (severe)
}


def build(src: Path, dst: Path, min_frame: int) -> tuple[int, int]:
    """Return (lines_read, lines_kept)."""
    lines = src.read_text().strip().splitlines()
    kept: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        img_path, orig_label = parts[0], int(parts[1])

        new_label = LABEL_MAP.get(orig_label)
        if new_label is None:
            continue

        # Frame index filter (skip first ~2s of clip)
        m = re.search(r"_(\d+)\.jpg", img_path)
        if m and int(m.group(1)) < min_frame:
            continue

        kept.append(f"{img_path} {new_label}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + "\n")
    return len(lines), len(kept)


def class_counts(path: Path) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        label = int(line.rsplit(" ", 1)[1])
        counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/biovid")
    parser.add_argument(
        "--min_frame",
        type=int,
        default=50,
        help="Skip frames with index < min_frame (default 50 ~= 2s at 25fps)",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"Building 3-class lists  [{{0,1,2}}=0  {{3}}=1  {{4}}=2]  min_frame={args.min_frame}")
    for split in args.splits:
        src = data_dir / f"{split}.txt"
        dst = data_dir / f"{split}_3class.txt"
        if not src.exists():
            print(f"  [SKIP] {src} not found")
            continue

        total, kept = build(src, dst, args.min_frame)
        counts = class_counts(dst)
        print(
            f"  [{split:5s}] {total:6d} -> {kept:5d} lines | "
            f"class0={counts[0]} class1={counts[1]} class2={counts[2]} -> {dst}"
        )

        # Suggest class weights (inverse frequency, normalized so sum == num_classes)
        total_kept = sum(counts.values())
        if total_kept > 0 and all(counts.values()):
            inv = [total_kept / (3 * counts[i]) for i in range(3)]
            print(f"           suggested class_weights: [{inv[0]:.3f}, {inv[1]:.3f}, {inv[2]:.3f}]")

    print("\nNext steps:")
    print("  1. Use config: configs/base_cfgs/data_cfg/datasets/biovid/biovid_3class.yaml")
    print("  2. Run experiments via scripts/experiments/run_fer_experiments_3class.sh")


if __name__ == "__main__":
    main()
