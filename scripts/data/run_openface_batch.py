#!/usr/bin/env python3
"""Run OpenFace on BioVid image frames and produce per-video CSVs.

This script bridges the gap between BioVid's per-frame images and
extract_au_features.py's expectation of per-video OpenFace CSVs.

OpenFace's FaceLandmarkImg processes individual images and outputs one CSV
per image. This script:
  1. Groups frames by video_id
  2. Runs OpenFace on each video's frames
  3. Merges per-image CSVs into one per-video CSV with correct frame numbers
  4. Optionally generates .npz via extract_au_features.py

Prerequisites:
  - OpenFace installed (see run_openface_extract_au.sh install)
  - Set OPENFACE_DIR env var or pass --openface_dir

Usage:
    # Process all frames in data list
    python scripts/data/run_openface_batch.py \
        --images_root data/biovid \
        --data_file data/biovid/train_skip2.txt \
        --output_dir data/biovid/openface_csv \
        --openface_dir ~/OpenFace

    # Then generate NPZ (using existing script)
    python scripts/data/extract_au_features.py \
        --openface_dir data/biovid/openface_csv \
        --data_file data/biovid/train_skip2.txt \
        --output data/biovid/au_features_openface_all17.npz \
        --au_subset all
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_data_file(data_file: str) -> Dict[str, List[Tuple[str, int]]]:
    """Parse data list and group frames by video_id.

    Returns:
        {video_id: [(relative_path, frame_index), ...]}
    """
    video_frames: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    with open(data_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rel_path = parts[0]
            stem = Path(rel_path).stem  # e.g. "071313_m_41-BL1-081_50"
            name_parts = stem.rsplit("_", 1)
            if len(name_parts) == 2:
                video_id = name_parts[0]
                frame_idx = int(name_parts[1])
            else:
                video_id = stem
                frame_idx = 0
            video_frames[video_id].append((rel_path, frame_idx))

    # Sort frames within each video
    for vid in video_frames:
        video_frames[vid].sort(key=lambda x: x[1])

    return dict(video_frames)


def run_openface_on_frames(
    openface_bin: str,
    frame_paths: List[str],
    frame_indices: List[int],
    output_csv: str,
    tmp_base: str,
) -> Tuple[int, int]:
    """Run OpenFace FaceLandmarkImg on a list of frames, merge into one CSV.

    Args:
        openface_bin: Path to FaceLandmarkImg binary
        frame_paths: Absolute paths to image files
        frame_indices: Corresponding frame numbers (for the output CSV)
        output_csv: Path for the merged output CSV
        tmp_base: Base directory for temporary files

    Returns:
        (n_processed, n_failed)
    """
    tmp_in = tempfile.mkdtemp(dir=tmp_base)
    tmp_out = tempfile.mkdtemp(dir=tmp_base)

    try:
        # Create symlinks with sequential names for OpenFace
        # We track the mapping: sequential_name → (original_path, frame_idx)
        frame_map: Dict[str, int] = {}  # symlink_stem → frame_idx
        for i, (fpath, fidx) in enumerate(zip(frame_paths, frame_indices)):
            if not os.path.isfile(fpath):
                continue
            # Use zero-padded sequential name to preserve order
            ext = Path(fpath).suffix
            link_name = f"frame_{i:06d}{ext}"
            os.symlink(os.path.realpath(fpath), os.path.join(tmp_in, link_name))
            frame_map[f"frame_{i:06d}"] = fidx

        if not frame_map:
            return 0, len(frame_paths)

        # Run OpenFace
        cmd = [
            openface_bin,
            "-fdir", tmp_in,
            "-out_dir", tmp_out,
            "-aus",       # output AU intensities
            "-2Dfp",      # output 2D landmarks (minimal overhead)
            "-wild",      # better for unconstrained faces
            "-nomask",    # don't mask out face region
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            # OpenFace may fail on some frames but still produce partial output
            pass

        # Merge per-image CSVs into one per-video CSV
        header = None
        rows: List[Dict[str, str]] = []
        n_processed = 0
        n_failed = 0

        for stem, fidx in sorted(frame_map.items(), key=lambda x: x[1]):
            of_csv = os.path.join(tmp_out, f"{stem}.csv")
            if not os.path.isfile(of_csv):
                n_failed += 1
                continue

            with open(of_csv) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    n_failed += 1
                    continue

                if header is None:
                    # Clean header names (OpenFace sometimes has leading spaces)
                    header = [h.strip() for h in reader.fieldnames]

                for row in reader:
                    cleaned = {k.strip(): v.strip() for k, v in row.items()}
                    # Override frame number with our actual frame index
                    cleaned["frame"] = str(fidx)
                    rows.append(cleaned)
                    n_processed += 1
                    break  # Only take first row per image (first face)

        # Write merged CSV
        if header and rows:
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for row in rows:
                    # Ensure all header keys exist in row
                    out_row = {k: row.get(k, "") for k in header}
                    writer.writerow(out_row)

        return n_processed, n_failed

    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenFace on BioVid frames, produce per-video CSVs"
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
        "--output_dir", type=str, required=True,
        help="Output directory for per-video CSVs",
    )
    parser.add_argument(
        "--openface_dir", type=str,
        default=os.environ.get("OPENFACE_DIR", os.path.expanduser("~/OpenFace")),
        help="OpenFace installation directory (default: ~/OpenFace or $OPENFACE_DIR)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip videos that already have CSVs (default: True)",
    )
    parser.add_argument(
        "--no_skip_existing", action="store_false", dest="skip_existing",
        help="Re-process all videos even if CSVs exist",
    )
    args = parser.parse_args()

    # Locate OpenFace binary
    openface_bin = os.path.join(args.openface_dir, "build", "bin", "FaceLandmarkImg")
    if not os.path.isfile(openface_bin):
        print(f"ERROR: OpenFace binary not found at {openface_bin}")
        print(f"Install OpenFace first:")
        print(f"  bash scripts/data/run_openface_extract_au.sh install")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse data file
    print(f"Parsing {args.data_file}...")
    video_frames = parse_data_file(args.data_file)
    n_videos = len(video_frames)
    print(f"Found {n_videos} videos")

    # Also process test file if train
    all_data_files = [args.data_file]
    if "train" in args.data_file:
        test_file = args.data_file.replace("train", "test")
        if os.path.exists(test_file):
            print(f"Also including test file: {test_file}")
            test_frames = parse_data_file(test_file)
            for vid, frames in test_frames.items():
                if vid in video_frames:
                    # Merge, avoiding duplicates
                    existing_indices = {f[1] for f in video_frames[vid]}
                    for f in frames:
                        if f[1] not in existing_indices:
                            video_frames[vid].append(f)
                else:
                    video_frames[vid] = frames
            all_data_files.append(test_file)
            n_videos = len(video_frames)
            print(f"Total: {n_videos} videos (train + test)")

    # Create temp directory for OpenFace intermediate files
    tmp_base = tempfile.mkdtemp(prefix="openface_")

    try:
        n_done = 0
        n_skip = 0
        n_fail = 0
        total_frames = 0
        total_failed_frames = 0
        t_start = time.time()

        for video_id, frames in sorted(video_frames.items()):
            csv_out = os.path.join(args.output_dir, f"{video_id}.csv")

            # Skip if already exists
            if args.skip_existing and os.path.isfile(csv_out) and os.path.getsize(csv_out) > 0:
                n_skip += 1
                continue

            # Build absolute paths
            abs_paths = [
                os.path.join(args.images_root, rel_path)
                for rel_path, _ in frames
            ]
            indices = [idx for _, idx in frames]

            n_proc, n_f = run_openface_on_frames(
                openface_bin, abs_paths, indices, csv_out, tmp_base
            )
            total_frames += n_proc
            total_failed_frames += n_f

            n_done += 1
            if n_done % 20 == 0 or n_done <= 3:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_videos - n_done - n_skip) / rate if rate > 0 else 0
                print(
                    f"  [{n_done + n_skip:>5d}/{n_videos}] "
                    f"{video_id}: {n_proc} frames OK, {n_f} failed | "
                    f"speed={rate:.1f} vid/s, ETA={eta/60:.0f}min"
                )

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

    elapsed = time.time() - t_start
    print(f"\nOpenFace batch processing complete!")
    print(f"  Videos processed: {n_done}")
    print(f"  Videos skipped:   {n_skip}")
    print(f"  Total frames OK:  {total_frames}")
    print(f"  Total frames failed: {total_failed_frames}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {args.output_dir}/")

    print(f"\nNext: generate NPZ files:")
    for subset, dim in [("all", 17), ("pain", 8)]:
        npz_name = f"au_features_openface_{subset}{dim}.npz"
        npz_path = os.path.join(os.path.dirname(args.data_file), npz_name)
        print(f"  python scripts/data/extract_au_features.py \\")
        print(f"      --openface_dir {args.output_dir} \\")
        print(f"      --data_file {args.data_file} \\")
        print(f"      --output {npz_path} \\")
        print(f"      --au_subset {subset}")


if __name__ == "__main__":
    main()
