"""Filter BioVid data lists to skip early frames (before pain response onset).

Based on Werner et al. (ACIIW 2017): "facial activity starts about 2s after
the temperature plateau is reached."

At 25fps over a 5.5s clip (~137 frames):
  - start_frame = int(137 * 0.20) = 27  →  1.08s  (current)
  - paper onset  = 2.0s                 →  frame 50 = 36.5%

Frames 27 and 38 (1.08s and 1.52s) pre-date the pain response onset and
behave like baseline — averaging them into the video-level prediction
contaminates PA1-PA4 toward class 0.

This script generates *_skip2.txt variants that exclude frames with index < 50.

Usage:
    python scripts/data/filter_frames.py
    python scripts/data/filter_frames.py --min_frame 62  # skip first 3 frames
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def filter_datalist(
    src: Path,
    dst: Path,
    min_frame: int,
) -> tuple[int, int]:
    """Return (original_count, kept_count)."""
    lines = src.read_text().strip().splitlines()
    kept = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Extract the trailing frame index: "images/xxx_<idx>.jpg <label>"
        m = re.search(r"_(\d+)\.jpg", line.split()[0])
        if m is None:
            kept.append(line)          # no frame index → keep as-is
            continue
        if int(m.group(1)) >= min_frame:
            kept.append(line)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + "\n")
    return len(lines), len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter BioVid data lists to skip early frames")
    parser.add_argument(
        "--data_dir",
        default="data/biovid",
        help="Directory containing train.txt / test.txt",
    )
    parser.add_argument(
        "--min_frame",
        type=int,
        default=50,
        help="Keep only frames with index >= min_frame (default: 50 ≈ 2.0s at 25fps)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Which splits to filter (default: train test)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    suffix = f"_skip{args.min_frame}"

    for split in args.splits:
        src = data_dir / f"{split}.txt"
        if not src.exists():
            print(f"[SKIP] {src} not found")
            continue

        dst = data_dir / f"{split}{suffix}.txt"
        orig, kept = filter_datalist(src, dst, args.min_frame)
        dropped = orig - kept
        print(
            f"[{split:5s}] {orig:6d} → {kept:6d} lines  "
            f"(dropped {dropped} frames with idx < {args.min_frame})"
        )
        print(f"         Saved → {dst}")

    print("\nDone. To use the filtered lists, update your biovid.yaml:")
    print(f"  data_cfg:")
    print(f"    dataset_cfg:")
    print(f"      train_data_list: data/biovid/train{suffix}.txt")
    print(f"      test_data_list:  data/biovid/test{suffix}.txt")


if __name__ == "__main__":
    main()
