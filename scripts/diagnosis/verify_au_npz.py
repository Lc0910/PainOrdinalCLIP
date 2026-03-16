#!/usr/bin/env python3
"""Verify AU feature NPZ quality — run after extract_au_pyfeat.py or OpenFace pipeline.

Checks:
  1. NPZ loads without error
  2. Entry count matches data list (train + test)
  3. Feature dimension matches expected au_dim
  4. No constant-zero dimensions (the AU45 bug symptom)
  5. No NaN / Inf values
  6. Per-AU statistics (mean, std, min, max) — low std flags dead AUs
  7. Key format match: sample keys from NPZ vs data list
  8. Coverage: how many data list entries are missing from NPZ

Usage:
    python scripts/diagnosis/verify_au_npz.py \
        --npz data/biovid/au_features_all17.npz \
        --train_list data/biovid/train_skip2.txt \
        --test_list data/biovid/test_skip2.txt \
        --au_dim 17

    python scripts/diagnosis/verify_au_npz.py \
        --npz data/biovid/au_features_pain8.npz \
        --train_list data/biovid/train_skip2.txt \
        --test_list data/biovid/test_skip2.txt \
        --au_dim 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

# AU name lists — must match extract_au_pyfeat.py / extract_au_features.py
PYFEAT_ALL17 = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
    "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
    "AU20", "AU23", "AU25", "AU26", "AU43",
]
PYFEAT_PAIN8 = ["AU04", "AU06", "AU07", "AU09", "AU10", "AU12", "AU25", "AU26"]

OPENFACE_ALL17 = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
OPENFACE_PAIN8 = [
    "AU04_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU25_r", "AU26_r",
]


def load_data_list_keys(*list_files: str) -> List[str]:
    """Extract image paths (first column) from data list files."""
    keys: List[str] = []
    for fpath in list_files:
        if not fpath:
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    keys.append(parts[0])
    return keys


def guess_au_names(au_dim: int, npz_path: str) -> List[str]:
    """Guess AU column names based on dim and filename."""
    is_openface = "openface" in npz_path.lower()
    if au_dim == 17:
        return OPENFACE_ALL17 if is_openface else PYFEAT_ALL17
    elif au_dim == 8:
        return OPENFACE_PAIN8 if is_openface else PYFEAT_PAIN8
    else:
        return [f"AU_{i}" for i in range(au_dim)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify AU feature NPZ quality")
    parser.add_argument("--npz", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--train_list", type=str, default="", help="Train data list")
    parser.add_argument("--test_list", type=str, default="", help="Test data list")
    parser.add_argument("--au_dim", type=int, default=0, help="Expected AU dim (0=auto)")
    args = parser.parse_args()

    errors: List[str] = []
    warnings: List[str] = []

    # ── 1. Load NPZ ──────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  AU NPZ Verification: {args.npz}")
    print(f"{'='*60}\n")

    try:
        raw = np.load(args.npz, allow_pickle=False)
        data = {k: raw[k] for k in raw.files}
    except Exception as e:
        print(f"FATAL: Cannot load NPZ: {e}")
        sys.exit(1)

    n_entries = len(data)
    print(f"[1] NPZ loaded: {n_entries} entries")

    if n_entries == 0:
        print("FATAL: NPZ is empty")
        sys.exit(1)

    # ── 2. Check dimensions ──────────────────────────────────────────
    sample_key = list(data.keys())[0]
    sample_val = data[sample_key]
    actual_dim = sample_val.shape[0] if sample_val.ndim == 1 else sample_val.shape[-1]

    print(f"[2] Feature dim: {actual_dim}, dtype: {sample_val.dtype}")

    if args.au_dim > 0 and actual_dim != args.au_dim:
        errors.append(
            f"Dimension mismatch: expected {args.au_dim}, got {actual_dim}"
        )

    au_names = guess_au_names(actual_dim, args.npz)

    # ── 3. Stack all values for statistics ───────────────────────────
    all_vals = np.stack(list(data.values()))  # [N, au_dim]
    print(f"[3] Value matrix: {all_vals.shape}")

    # ── 4. NaN / Inf check ───────────────────────────────────────────
    n_nan = np.isnan(all_vals).sum()
    n_inf = np.isinf(all_vals).sum()
    if n_nan > 0:
        errors.append(f"Found {n_nan} NaN values")
    if n_inf > 0:
        errors.append(f"Found {n_inf} Inf values")
    print(f"[4] NaN={n_nan}, Inf={n_inf}")

    # ── 5. Per-AU statistics ─────────────────────────────────────────
    means = all_vals.mean(axis=0)
    stds = all_vals.std(axis=0)
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    zero_ratio = (all_vals == 0).mean(axis=0)  # fraction of zeros per AU

    print(f"\n[5] Per-AU statistics:")
    print(f"    {'AU':<10s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'%zero':>8s} {'status'}")
    print(f"    {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    dead_dims = []
    for i in range(actual_dim):
        name = au_names[i] if i < len(au_names) else f"dim_{i}"
        status = "OK"
        if stds[i] < 1e-6:
            status = "DEAD"
            dead_dims.append(name)
        elif stds[i] < 0.01:
            status = "LOW"
            warnings.append(f"{name} has very low std={stds[i]:.6f}")
        print(
            f"    {name:<10s} {means[i]:8.4f} {stds[i]:8.4f} "
            f"{mins[i]:8.4f} {maxs[i]:8.4f} {zero_ratio[i]*100:7.1f}% {status}"
        )

    if dead_dims:
        errors.append(
            f"Constant-zero dimensions: {dead_dims}. "
            f"If AU45/AU43, this is the known py-feat AU45 bug — re-extract with fixed script."
        )

    # ── 6. All-zero entries ──────────────────────────────────────────
    row_sums = np.abs(all_vals).sum(axis=1)
    n_all_zero = (row_sums == 0).sum()
    pct_zero = n_all_zero / n_entries * 100
    print(f"\n[6] All-zero entries: {n_all_zero}/{n_entries} ({pct_zero:.1f}%)")
    if pct_zero > 20:
        warnings.append(
            f"{pct_zero:.1f}% entries are all-zero — face detection likely failed on these frames"
        )

    # ── 7. Key format + coverage check ───────────────────────────────
    list_files = [f for f in [args.train_list, args.test_list] if f]
    if list_files:
        list_keys = load_data_list_keys(*list_files)
        npz_keys = set(data.keys())

        # Show sample keys for comparison
        print(f"\n[7] Key format check:")
        print(f"    NPZ sample keys (first 3):")
        for k in list(data.keys())[:3]:
            print(f"      '{k}'")
        print(f"    Data list sample keys (first 3):")
        for k in list_keys[:3]:
            print(f"      '{k}'")

        # Coverage
        missing = [k for k in list_keys if k not in npz_keys]
        extra = npz_keys - set(list_keys)
        coverage = (len(list_keys) - len(missing)) / len(list_keys) * 100 if list_keys else 0

        print(f"\n    Data list entries: {len(list_keys)}")
        print(f"    NPZ entries:      {n_entries}")
        print(f"    Missing from NPZ: {len(missing)}")
        print(f"    Extra in NPZ:     {len(extra)}")
        print(f"    Coverage:         {coverage:.1f}%")

        if len(missing) > 0:
            print(f"    First 5 missing keys:")
            for k in missing[:5]:
                print(f"      '{k}'")

        if coverage < 95:
            errors.append(
                f"NPZ coverage only {coverage:.1f}% — {len(missing)} entries missing. "
                f"Check key format (relative path) matches between NPZ and data list."
            )
        elif coverage < 100:
            warnings.append(f"NPZ coverage {coverage:.1f}% — {len(missing)} entries missing")
    else:
        print(f"\n[7] Skipped key check (no data list provided)")

    # ── 8. Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    ✗ {e}")
    if warnings:
        print(f"  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    ! {w}")
    if not errors and not warnings:
        print("  ALL CHECKS PASSED")
    elif not errors:
        print(f"  PASSED with {len(warnings)} warning(s)")
    else:
        print(f"  FAILED — {len(errors)} error(s)")
    print(f"{'='*60}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
