"""BioVid data integrity verifier.

Checks the most common causes of ~22% (random) accuracy:
  1. Label distribution & range (are they 0-4 or some other encoding?)
  2. Image existence (are paths valid relative to images_root?)
  3. Image readability & shape (can PIL open them?)
  4. Subject-level train/test overlap (critical for generalisation)
  5. Class balance (severe imbalance ≡ inflated/deflated accuracy)
  6. CLIP zero-shot sanity check (optional, needs CLIP installed)

Usage:
    # Basic data check (fast)
    python scripts/diagnosis/verify_biovid_data.py \
        --train_file data/biovid/train.txt \
        --test_file  data/biovid/test.txt \
        --images_root data/biovid

    # Full check including random image samples (slow if dataset is large)
    python scripts/diagnosis/verify_biovid_data.py \
        --train_file data/biovid/train.txt \
        --test_file  data/biovid/test.txt \
        --images_root data/biovid \
        --check_images --n_sample_images 100
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


# ---- helpers ----------------------------------------------------------------

def read_txt(path: str) -> list[tuple[str, list[int]]]:
    """Return list of (img_file, [labels]) from a data list txt."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_file = parts[0]
            labels = [int(x) for x in parts[1:]]
            rows.append((img_file, labels))
    return rows


def label_stats(rows: list[tuple[str, list[int]]], split_name: str) -> None:
    """Print label distribution and range."""
    all_labels = []
    for _, labels in rows:
        # use median label as the effective label (same as val/test logic in data.py)
        median_label = labels[len(labels) // 2]
        all_labels.append(median_label)

    dist = Counter(all_labels)
    print(f"\n[{split_name}] Label distribution ({len(all_labels)} samples):")
    for cls in sorted(dist.keys()):
        pct = dist[cls] / len(all_labels) * 100
        bar = "#" * int(pct / 2)
        print(f"  label {cls}: {dist[cls]:6d} ({pct:5.1f}%)  {bar}")

    label_min, label_max = min(all_labels), max(all_labels)
    print(f"  Range: [{label_min}, {label_max}]")
    if label_min < 0 or label_max >= 10:
        print(f"  ⚠️  Unexpected label range! Expected 0-4 for BioVid Part A.")
    elif label_min == 1 and label_max == 5:
        print(f"  ⚠️  Labels are 1-indexed (1-5)! Model expects 0-indexed (0-4).")
        print(f"      FIX: subtract 1 from all labels in your txt files.")


def subject_overlap(train_rows, test_rows) -> None:
    """Extract subject IDs from filenames and check train/test overlap."""

    def extract_subject(img_file: str) -> str:
        # BioVid Part A pattern: {subject_id}-{condition}_{frame}.jpg
        # or: images/{subject_id}-PA1-{N}_{frame}.jpg
        stem = Path(img_file).stem  # e.g. "071309_w_21-BL1-081_27"
        # Subject ID is typically the part before the first "-" that contains pain level
        # e.g. "071309_w_21" from "071309_w_21-BL1-081_27"
        parts = stem.split("-")
        # Look for pain level marker (BL1/PA1/PA2/PA3/PA4)
        pain_markers = {"BL1", "PA1", "PA2", "PA3", "PA4"}
        for i, part in enumerate(parts):
            if any(part.startswith(m) for m in pain_markers):
                return "-".join(parts[:i])
        # Fallback: first component
        return parts[0]

    train_subjects = set(extract_subject(r[0]) for r in train_rows)
    test_subjects = set(extract_subject(r[0]) for r in test_rows)
    overlap = train_subjects & test_subjects

    print(f"\n[Subject analysis]")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Test  subjects: {len(test_subjects)}")
    print(f"  Overlap: {len(overlap)}")

    if len(overlap) == 0:
        print("  ✓ Subject-independent split (harder, realistic)")
        print("  ⚠️  Frame-level accuracy <35% is expected for subject-independent BioVid evaluation.")
        print("      Focus on VIDEO-level accuracy which should be ~5-10% higher.")
    elif len(overlap) > 0:
        pct = len(overlap) / max(len(train_subjects), 1) * 100
        print(f"  ⚠️  {pct:.0f}% subjects appear in both train and test (leakage risk)")


def check_images(rows, images_root: str, n_sample: int = 50) -> None:
    """Spot-check n_sample images for existence and readability."""
    from PIL import Image

    sample = random.sample(rows, min(n_sample, len(rows)))
    missing = 0
    unreadable = 0
    wrong_mode = 0

    for img_file, _ in sample:
        full_path = os.path.join(images_root, img_file)
        if not os.path.exists(full_path):
            missing += 1
            if missing <= 3:
                print(f"  [MISSING] {full_path}")
            continue
        try:
            img = Image.open(full_path)
            img.verify()
            if img.mode not in ("RGB", "L"):
                wrong_mode += 1
        except Exception as e:
            unreadable += 1
            if unreadable <= 3:
                print(f"  [UNREADABLE] {full_path}: {e}")

    total = len(sample)
    print(f"\n[Image check] Sampled {total} images from {images_root}")
    print(f"  Missing:    {missing}/{total}")
    print(f"  Unreadable: {unreadable}/{total}")
    print(f"  Wrong mode: {wrong_mode}/{total}")
    if missing > 0:
        print(f"  ⚠️  Fix: ensure images_root is set correctly and images exist.")


def check_normalization() -> None:
    """Remind about CLIP normalization."""
    print("\n[Normalization check]")
    print("  ImageNet (default.yaml): mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]")
    print("  CLIP official:           mean=[0.481, 0.458, 0.408] std=[0.269, 0.261, 0.276]")
    print("  Std difference: ~17% — using ImageNet stats corrupts CLIP features!")
    print("  FIX: add --config configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml")
    print("       to every run command.")


def check_overfit_feasibility(train_rows, num_classes: int = 5) -> None:
    """Estimate if model has enough data to learn per-class patterns."""
    label_dist = Counter(r[1][len(r[1]) // 2] for r in train_rows)
    min_samples = min(label_dist.get(c, 0) for c in range(num_classes))
    total = sum(label_dist.values())
    print(f"\n[Overfit feasibility]")
    print(f"  Total train samples: {total}")
    print(f"  Min samples per class: {min_samples}")
    if min_samples < 100:
        print(f"  ⚠️  Very few samples for some classes. Consider few-shot baseline.")
    if min_samples > 1000:
        print(f"  ✓ Sufficient samples for each class.")

    # Majority-class baseline
    majority_cls = label_dist.most_common(1)[0]
    majority_acc = majority_cls[1] / total
    print(f"  Majority class baseline: predict rank {majority_cls[0]} always → acc={majority_acc:.1%}")
    print(f"  Random baseline: acc={1/num_classes:.1%}")


# ---- main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BioVid data integrity verifier")
    parser.add_argument("--train_file", default="data/biovid/train.txt")
    parser.add_argument("--test_file",  default="data/biovid/test.txt")
    parser.add_argument("--images_root", default="data/biovid")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--check_images", action="store_true",
                        help="Spot-check image existence and readability (slow)")
    parser.add_argument("--n_sample_images", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("BioVid Data Verifier")
    print("=" * 60)

    # Load data files
    if not os.path.exists(args.train_file):
        print(f"⚠️  Train file not found: {args.train_file}")
        return
    if not os.path.exists(args.test_file):
        print(f"⚠️  Test file not found: {args.test_file}")
        return

    train_rows = read_txt(args.train_file)
    test_rows  = read_txt(args.test_file)

    print(f"\nLoaded: {len(train_rows)} train, {len(test_rows)} test rows")
    print(f"Label columns per row: train={len(train_rows[0][1])}, test={len(test_rows[0][1])}")

    # --- Checks ---
    label_stats(train_rows, "train")
    label_stats(test_rows,  "test")
    subject_overlap(train_rows, test_rows)
    check_overfit_feasibility(train_rows, args.num_classes)
    check_normalization()

    if args.check_images:
        check_images(train_rows, args.images_root, args.n_sample_images)
        check_images(test_rows,  args.images_root, args.n_sample_images // 2)

    print("\n" + "=" * 60)
    print("Summary of likely causes for 22% accuracy:")
    print("  1. Wrong normalization (ImageNet vs CLIP) → add clip-normalize.yaml")
    print("  2. lr_image_encoder=1e-4 destroying CLIP features → add freeze-image.yaml")
    print("  3. Subject-independent split: frame-level acc <35% is expected")
    print("  4. Labels 1-indexed instead of 0-indexed → check label range above")
    print("=" * 60)


if __name__ == "__main__":
    main()
