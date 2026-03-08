#!/usr/bin/env python3
"""Extract AU (Action Unit) features from face images using py-feat.

py-feat is a pure-Python toolbox that detects faces and extracts AU
intensities directly from images — no OpenFace compilation needed.

Install:
    pip install py-feat

Usage:
    # All AUs (17 common intensity AUs)
    python scripts/data/extract_au_pyfeat.py \
        --images_root data/biovid \
        --data_file data/biovid/train_skip2.txt \
        --output data/biovid/au_features_all17.npz \
        --au_subset all

    # 8 pain-relevant AUs only
    python scripts/data/extract_au_pyfeat.py \
        --images_root data/biovid \
        --data_file data/biovid/train_skip2.txt \
        --output data/biovid/au_features_pain8.npz \
        --au_subset pain

    # Also process test set (auto-detected):
    #   If data_file contains "train", the script automatically looks for
    #   the corresponding "test" file and merges both into one .npz.

AU columns:
    py-feat outputs ~20 AU intensities.  We select 17 that match OpenFace
    conventions (AU01..AU26 + AU45) or the 8 pain-relevant subset.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# ─── AU column definitions ───────────────────────────────────────────
# Map: our canonical name → py-feat column name
# py-feat names AUs as "AU01", "AU02", ..., "AU43"
# OpenFace uses "AU01_r", ..., "AU45_r"
# We store as float32 vectors; the naming only matters for column selection.

ALL_AU_NAMES: List[str] = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
    "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
    "AU20", "AU23", "AU25", "AU26", "AU45",
]

PAIN_AU_NAMES: List[str] = [
    "AU04",  # brow lowerer
    "AU06",  # cheek raiser
    "AU07",  # lid tightener
    "AU09",  # nose wrinkler
    "AU10",  # upper lip raiser
    "AU12",  # lip corner puller
    "AU25",  # lips part
    "AU26",  # jaw drop
]


def extract_au_for_images(
    image_paths: List[str],
    au_names: List[str],
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Extract AU features for a list of images using py-feat.

    Args:
        image_paths: Absolute paths to images.
        au_names:    AU columns to extract (e.g. ["AU04", "AU06", ...]).
        batch_size:  Batch size for py-feat detection.
        device:      "cuda" or "cpu".

    Returns:
        Dict mapping image path → numpy array of AU intensities [au_dim].
    """
    try:
        from feat import Detector
    except ImportError:
        print("ERROR: py-feat not installed.  Run:  pip install py-feat")
        sys.exit(1)

    print(f"Initializing py-feat Detector on {device}...")
    detector = Detector(device=device)

    au_dim = len(au_names)
    result: Dict[str, np.ndarray] = {}
    n_total = len(image_paths)
    n_failed = 0

    # Process in batches
    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_paths = image_paths[batch_start:batch_end]

        t0 = time.time()
        try:
            detections = detector.detect_image(batch_paths)
        except Exception as e:
            print(f"  WARNING: batch {batch_start}-{batch_end} failed: {e}")
            for p in batch_paths:
                result[p] = np.zeros(au_dim, dtype=np.float32)
            n_failed += len(batch_paths)
            continue
        dt = time.time() - t0

        # detections is a DataFrame with columns like AU01, AU02, ...
        # One row per detected face (may have 0 or >1 per image).
        # We group by input image and take the first face (highest confidence).
        available_cols = [c for c in au_names if c in detections.columns]
        missing_cols = [c for c in au_names if c not in detections.columns]
        if missing_cols and batch_start == 0:
            print(f"  WARNING: py-feat missing columns: {missing_cols}")
            print(f"  Available AU columns: {[c for c in detections.columns if c.startswith('AU')]}")

        # py-feat >= 0.6 uses 'input' column for image path
        if "input" in detections.columns:
            group_col = "input"
        elif "FaceRectX" in detections.columns:
            # Older versions — rows are ordered by input
            group_col = None
        else:
            group_col = None

        if group_col is not None:
            for img_path in batch_paths:
                # Match by filename in the 'input' column
                fname = os.path.basename(img_path)
                mask = detections[group_col].astype(str).str.contains(fname, regex=False)
                rows = detections.loc[mask]
                if len(rows) == 0:
                    result[img_path] = np.zeros(au_dim, dtype=np.float32)
                    n_failed += 1
                else:
                    # Take first detected face
                    au_vec = np.zeros(au_dim, dtype=np.float32)
                    for i, col in enumerate(au_names):
                        if col in rows.columns:
                            val = rows.iloc[0][col]
                            au_vec[i] = float(val) if not np.isnan(float(val)) else 0.0
                    result[img_path] = au_vec
        else:
            # Fallback: assume one row per image, in order
            for idx, img_path in enumerate(batch_paths):
                if idx < len(detections):
                    au_vec = np.zeros(au_dim, dtype=np.float32)
                    for i, col in enumerate(au_names):
                        if col in detections.columns:
                            val = detections.iloc[idx][col]
                            au_vec[i] = float(val) if not np.isnan(float(val)) else 0.0
                    result[img_path] = au_vec
                else:
                    result[img_path] = np.zeros(au_dim, dtype=np.float32)
                    n_failed += 1

        if (batch_start // batch_size) % 10 == 0 or batch_end == n_total:
            print(
                f"  [{batch_end:>6d}/{n_total}] "
                f"{dt:.1f}s/batch, failed={n_failed}"
            )

    print(f"Extracted AU features: {len(result)} images, "
          f"au_dim={au_dim}, failed={n_failed}")
    return result


def build_mapping_from_data_file(
    data_file: str,
    images_root: str,
    au_names: List[str],
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Read data list, extract AU features, return {relative_path → au_vec}.

    Args:
        data_file:   Path to data list (train_skip2.txt etc.)
        images_root: Root directory for images (e.g. data/biovid)
        au_names:    AU columns to extract
        batch_size:  py-feat batch size
        device:      "cuda" or "cpu"

    Returns:
        Dict mapping relative image path (as in data list) → AU vector.
    """
    # Parse data file → relative image paths
    rel_paths: List[str] = []
    with open(data_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_paths.append(parts[0])

    print(f"Data file: {data_file} → {len(rel_paths)} images")

    # Build absolute paths for py-feat
    abs_paths = [os.path.join(images_root, rp) for rp in rel_paths]

    # Verify a few images exist
    for p in abs_paths[:3]:
        if not os.path.isfile(p):
            print(f"ERROR: Image not found: {p}")
            print(f"Check --images_root (got: {images_root})")
            sys.exit(1)

    # Extract
    abs_result = extract_au_for_images(abs_paths, au_names, batch_size, device)

    # Map back to relative paths (keys must match data list for AUFeatureStore)
    result: Dict[str, np.ndarray] = {}
    for rel, absp in zip(rel_paths, abs_paths):
        result[rel] = abs_result.get(absp, np.zeros(len(au_names), dtype=np.float32))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract AU features from face images using py-feat"
    )
    parser.add_argument(
        "--images_root", type=str, required=True,
        help="Root directory for images (e.g. data/biovid)",
    )
    parser.add_argument(
        "--data_file", type=str, required=True,
        help="Data list file (train_skip2.txt etc.)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output .npz file path",
    )
    parser.add_argument(
        "--au_subset", type=str, default="all", choices=["all", "pain"],
        help="AU subset: 'all' (17 AUs) or 'pain' (8 pain-relevant AUs)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for py-feat detection (default: 32)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu' (default: cuda)",
    )
    args = parser.parse_args()

    au_names = ALL_AU_NAMES if args.au_subset == "all" else PAIN_AU_NAMES
    print(f"Extracting {len(au_names)} AUs: {au_names}")
    print(f"Device: {args.device}, Batch size: {args.batch_size}")

    # Extract from main data file
    mapping = build_mapping_from_data_file(
        args.data_file, args.images_root, au_names,
        args.batch_size, args.device,
    )

    # Auto-detect and process test file if data_file is train
    if "train" in args.data_file:
        test_file = args.data_file.replace("train", "test")
        if os.path.exists(test_file):
            print(f"\nAlso processing test file: {test_file}")
            test_mapping = build_mapping_from_data_file(
                test_file, args.images_root, au_names,
                args.batch_size, args.device,
            )
            mapping.update(test_mapping)

    # Save
    np.savez_compressed(args.output, **mapping)
    print(f"\nSaved to {args.output} ({len(mapping)} entries, "
          f"au_dim={len(au_names)})")


if __name__ == "__main__":
    main()
