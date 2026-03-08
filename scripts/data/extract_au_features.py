#!/usr/bin/env python3
"""Extract AU (Action Unit) features from OpenFace CSV outputs.

Reads per-video CSVs produced by OpenFace, extracts AU intensity columns,
and saves a single .npz file mapping frame_path → AU feature vector.

Usage:
    # All 17 AUs
    python scripts/data/extract_au_features.py \
        --openface_dir data/biovid/openface_csv \
        --data_file data/biovid/train.txt \
        --output data/biovid/au_features_all17.npz \
        --au_subset all

    # 8 pain-relevant AUs
    python scripts/data/extract_au_features.py \
        --openface_dir data/biovid/openface_csv \
        --data_file data/biovid/train.txt \
        --output data/biovid/au_features_pain8.npz \
        --au_subset pain

OpenFace CSV column convention:
    AU{nn}_r  — intensity (regression), range ~[0, 5]
    AU{nn}_c  — presence (classification), binary 0/1
    We only use the _r (intensity) columns.
"""
from __future__ import annotations

import argparse
import csv as csv_module
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# All 17 AU intensity columns available from OpenFace
ALL_AU_COLUMNS: List[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

# 8 pain-relevant AUs (Prkachin & Solomon, 2008; Lucey et al., 2011)
PAIN_AU_COLUMNS: List[str] = [
    "AU04_r",  # brow lowerer
    "AU06_r",  # cheek raiser
    "AU07_r",  # lid tightener
    "AU09_r",  # nose wrinkler
    "AU10_r",  # upper lip raiser
    "AU12_r",  # lip corner puller
    "AU25_r",  # lips part
    "AU26_r",  # jaw drop
]


def load_openface_csv(csv_path: str, au_columns: List[str]) -> Dict[int, np.ndarray]:
    """Load an OpenFace CSV and return {frame_number: au_vector}.

    Args:
        csv_path: Path to OpenFace output CSV.
        au_columns: List of AU column names to extract.

    Returns:
        Dict mapping frame number → numpy array of AU intensities.
    """
    result: Dict[int, np.ndarray] = {}
    with open(csv_path, "r") as f:
        reader = csv_module.DictReader(f)
        # OpenFace headers sometimes have leading spaces — validated here
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")

        for row in reader:
            # Clean row keys
            cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
            frame_num = int(cleaned_row.get("frame", "0"))
            au_values = []
            for col in au_columns:
                val_str = cleaned_row.get(col, "0.0")
                try:
                    au_values.append(float(val_str))
                except ValueError:
                    au_values.append(0.0)
            result[frame_num] = np.array(au_values, dtype=np.float32)

    return result


def build_frame_mapping(
    data_file: str,
    openface_dir: str,
    au_columns: List[str],
) -> Dict[str, np.ndarray]:
    """Build mapping from image path (as in data list) to AU feature vector.

    Strategy:
    1. Parse data_file to get all frame paths.
    2. For each frame, determine the video_id and frame_index from filename.
    3. Load the corresponding OpenFace CSV and extract AU for that frame.

    BioVid filename: images/071313_m_41-BL1-081_50.jpg
       video_id = "071313_m_41-BL1-081"
       frame_index = 50
    """
    # Parse all frame paths from data file
    frame_paths: List[str] = []
    with open(data_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                frame_paths.append(parts[0])

    # Group frames by video_id
    video_frames: Dict[str, List[tuple]] = defaultdict(list)
    for fp in frame_paths:
        stem = Path(fp).stem  # e.g., "071313_m_41-BL1-081_50"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            video_id = parts[0]
            frame_idx = int(parts[1])
        else:
            video_id = stem
            frame_idx = 0
        video_frames[video_id].append((fp, frame_idx))

    # Load OpenFace CSVs and build mapping
    result: Dict[str, np.ndarray] = {}
    au_dim = len(au_columns)
    missing_csv = 0
    missing_frame = 0

    for video_id, frames in video_frames.items():
        csv_path = os.path.join(openface_dir, f"{video_id}.csv")
        if not os.path.exists(csv_path):
            missing_csv += 1
            for fp, _ in frames:
                result[fp] = np.zeros(au_dim, dtype=np.float32)
            continue

        au_data = load_openface_csv(csv_path, au_columns)

        for fp, frame_idx in frames:
            if frame_idx in au_data:
                result[fp] = au_data[frame_idx]
            else:
                # Try 1-indexed (OpenFace default is 1-indexed)
                if (frame_idx + 1) in au_data:
                    result[fp] = au_data[frame_idx + 1]
                else:
                    missing_frame += 1
                    result[fp] = np.zeros(au_dim, dtype=np.float32)

    print(f"Built AU mapping: {len(result)} frames, "
          f"au_dim={au_dim}, "
          f"missing_csv={missing_csv} videos, "
          f"missing_frame={missing_frame} frames")

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract AU features from OpenFace CSVs")
    parser.add_argument(
        "--openface_dir", type=str, required=True,
        help="Directory containing OpenFace CSV files (one per video)",
    )
    parser.add_argument(
        "--data_file", type=str, required=True,
        help="Data list file (train.txt or test.txt) with frame paths",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output .npz file path",
    )
    parser.add_argument(
        "--au_subset", type=str, default="all", choices=["all", "pain"],
        help="AU subset: 'all' (17 AUs) or 'pain' (8 pain-relevant AUs)",
    )
    args = parser.parse_args()

    au_columns = ALL_AU_COLUMNS if args.au_subset == "all" else PAIN_AU_COLUMNS
    print(f"Extracting {len(au_columns)} AUs: {au_columns}")

    mapping = build_frame_mapping(args.data_file, args.openface_dir, au_columns)

    # Save as npz: keys are frame paths, values are AU vectors
    np.savez_compressed(args.output, **mapping)
    print(f"Saved to {args.output} ({len(mapping)} entries)")

    # Also generate for test file if the data_file is train
    if "train" in args.data_file:
        test_file = args.data_file.replace("train", "test")
        if os.path.exists(test_file):
            print(f"\nAlso processing test file: {test_file}")
            test_mapping = build_frame_mapping(test_file, args.openface_dir, au_columns)
            # Merge into single file
            mapping.update(test_mapping)
            np.savez_compressed(args.output, **mapping)
            print(f"Updated {args.output} with train+test ({len(mapping)} entries)")


if __name__ == "__main__":
    main()
