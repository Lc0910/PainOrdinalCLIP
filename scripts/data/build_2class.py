"""Build 2-class BioVid data lists: BLN (0) vs PA4 (4→1).

Applies two optimisations simultaneously:
  1. Only keep BLN and PA4 samples (discard PA1/PA2/PA3)
  2. Skip early frames (index < MIN_FRAME_IDX) — paper shows pain response
     starts ~2 s into the clip; at 25 fps that is frame 50.

Output files (written next to the originals):
  data/biovid/train_2class.txt   label ∈ {0, 1}
  data/biovid/test_2class.txt    label ∈ {0, 1}

Usage:
    python scripts/data/build_2class.py
    python scripts/data/build_2class.py --min_frame 50   # default
    python scripts/data/build_2class.py --min_frame 0    # no frame skip
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


# Original label → 2-class label  (None = discard)
LABEL_MAP: dict[int, int | None] = {
    0: 0,    # BLN  → class 0
    1: None, # PA1  → discard
    2: None, # PA2  → discard
    3: None, # PA3  → discard
    4: 1,    # PA4  → class 1
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

        # 1. class filter
        new_label = LABEL_MAP.get(orig_label)
        if new_label is None:
            continue

        # 2. frame index filter
        m = re.search(r"_(\d+)\.jpg", img_path)
        if m and int(m.group(1)) < min_frame:
            continue

        kept.append(f"{img_path} {new_label}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + "\n")
    return len(lines), len(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/biovid")
    parser.add_argument("--min_frame",  type=int, default=50,
                        help="Skip frames with index < min_frame (default 50 ≈ 2 s at 25 fps)")
    parser.add_argument("--splits",     nargs="+", default=["train", "test"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"Building 2-class lists  [BLN=0, PA4→1]  min_frame={args.min_frame}")
    for split in args.splits:
        src = data_dir / f"{split}.txt"
        dst = data_dir / f"{split}_2class.txt"
        if not src.exists():
            print(f"  [SKIP] {src} not found")
            continue
        total, kept = build(src, dst, args.min_frame)
        blns  = sum(1 for l in dst.read_text().splitlines() if l.endswith(" 0"))
        pa4s  = sum(1 for l in dst.read_text().splitlines() if l.endswith(" 1"))
        print(f"  [{split:5s}] {total:6d} → {kept:5d} lines"
              f"  (BLN={blns}, PA4={pa4s})  → {dst}")

    print("\nNext steps:")
    print("  1. Add config: num_ranks=2, classnames=[no pain, intense pain]")
    print("  2. Run: bash scripts/experiments/run_biovid_2class.sh --backbone vitb16 --max_epochs 50")


if __name__ == "__main__":
    main()
