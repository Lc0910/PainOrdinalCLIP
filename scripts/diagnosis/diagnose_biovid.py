"""BioVid prediction diagnosis tool.

Reads test_video_predictions.csv (or test_stats.json) from a results directory
and prints:
  1. Class distribution of ground-truth labels
  2. Class distribution of predicted labels (after rounding pred_max/pred_exp)
  3. Confusion matrix (5x5 for BioVid)
  4. Per-class accuracy and MAE
  5. Whether the model collapsed to a single class

Usage:
    python scripts/diagnosis/diagnose_biovid.py -d results/biovid-coop-frozen-rn50
    python scripts/diagnosis/diagnose_biovid.py -d results/  # all sub-dirs
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


# ---- helpers ----------------------------------------------------------------

def load_video_predictions(csv_path: Path) -> list[dict]:
    """Parse video-level CSV with columns: epoch, ckpt_path, video_id, gt, pred_exp, pred_max, n_frames."""
    import csv
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "video_id": row["video_id"],
                "gt": int(row["gt"]),
                "pred_exp": float(row["pred_exp"]),
                "pred_max": float(row["pred_max"]),
                "n_frames": int(row["n_frames"]),
            })
    return rows


def last_epoch_rows(rows: list[dict]) -> list[dict]:
    """Keep only rows from the last epoch (assumes CSV appended per epoch)."""
    if not rows:
        return []
    max_epoch = max(r["epoch"] for r in rows)
    return [r for r in rows if r["epoch"] == max_epoch]


def confusion_matrix(gts: list[int], preds: list[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(gts, preds):
        g_clip = max(0, min(num_classes - 1, g))
        p_clip = max(0, min(num_classes - 1, p))
        cm[g_clip][p_clip] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray, num_classes: int = 5) -> None:
    print("\n  Confusion matrix (rows=GT, cols=Pred):")
    header = "GT\\Pred | " + "  ".join(f"{i:3d}" for i in range(num_classes))
    print("  " + header)
    print("  " + "-" * len(header))
    for i in range(num_classes):
        row_str = "  ".join(f"{cm[i][j]:3d}" for j in range(num_classes))
        print(f"  rank {i}  | {row_str}")


def diagnose_directory(results_dir: Path, run_type: str = "test", num_classes: int = 5) -> None:
    csv_path = results_dir / f"{run_type}_video_predictions.csv"
    stats_path = results_dir / f"{run_type}_video_stats.json"
    frame_stats_path = results_dir / f"{run_type}_stats.json"

    print(f"\n{'=' * 60}")
    print(f"Directory: {results_dir}")

    # --- Load video predictions ---
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found")
        return

    rows = load_video_predictions(csv_path)
    if not rows:
        print("  [SKIP] CSV is empty")
        return

    rows = last_epoch_rows(rows)
    print(f"  Epoch: {rows[0]['epoch']}  |  Videos: {len(rows)}")

    gts = [r["gt"] for r in rows]
    preds_max = [int(round(r["pred_max"])) for r in rows]
    preds_exp = [int(round(r["pred_exp"])) for r in rows]

    # --- GT distribution ---
    gt_dist = Counter(gts)
    print("\n  GT label distribution:")
    for cls in range(num_classes):
        bar = "#" * gt_dist.get(cls, 0)
        print(f"    rank {cls}: {gt_dist.get(cls, 0):4d}  {bar}")

    # --- Pred distribution ---
    pred_dist_max = Counter(preds_max)
    pred_dist_exp = Counter(preds_exp)
    print("\n  Predicted label distribution (pred_max):")
    for cls in range(num_classes):
        bar = "#" * pred_dist_max.get(cls, 0)
        print(f"    rank {cls}: {pred_dist_max.get(cls, 0):4d}  {bar}")

    # --- Collapse detection ---
    top1_count = max(pred_dist_max.values()) if pred_dist_max else 0
    total = len(rows)
    collapse_ratio = top1_count / total if total > 0 else 0
    if collapse_ratio > 0.7:
        dominant_cls = pred_dist_max.most_common(1)[0][0]
        print(f"\n  ⚠️  MODEL COLLAPSE DETECTED: {top1_count}/{total} ({collapse_ratio:.0%}) "
              f"predictions are rank {dominant_cls}")

    # --- Accuracy & MAE ---
    acc_max = sum(1 for g, p in zip(gts, preds_max) if g == p) / total
    acc_exp = sum(1 for g, p in zip(gts, preds_exp) if g == p) / total
    mae_max = sum(abs(g - p) for g, p in zip(gts, preds_max)) / total
    mae_exp = sum(abs(g - p) for g, p in zip(gts, preds_exp)) / total

    print(f"\n  Video-level metrics (pred_max): acc={acc_max:.4f}  mae={mae_max:.4f}")
    print(f"  Video-level metrics (pred_exp): acc={acc_exp:.4f}  mae={mae_exp:.4f}")

    # --- Per-class accuracy ---
    print("\n  Per-class accuracy (pred_max):")
    for cls in range(num_classes):
        cls_gts = [p for g, p in zip(gts, preds_max) if g == cls]
        if not cls_gts:
            print(f"    rank {cls}: no samples")
            continue
        cls_acc = sum(1 for p in cls_gts if p == cls) / len(cls_gts)
        print(f"    rank {cls}: {cls_acc:.2%}  (n={len(cls_gts)})")

    # --- Confusion matrix ---
    cm = confusion_matrix(gts, preds_max, num_classes)
    print_confusion_matrix(cm, num_classes)

    # --- Frame-level stats (last entry in JSONL) ---
    if frame_stats_path.exists():
        lines = frame_stats_path.read_text().strip().splitlines()
        if lines:
            try:
                frame_stats = json.loads(lines[-1])
                print(f"\n  Frame-level (last epoch): "
                      f"acc_max={frame_stats.get('acc_max_metric', '?'):.4f}  "
                      f"mae_max={frame_stats.get('mae_max_metric', '?'):.4f}")
            except json.JSONDecodeError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose BioVid model predictions")
    parser.add_argument("-d", "--dir", required=True, help="Results directory (or parent of multiple)")
    parser.add_argument("-r", "--run_type", default="test", choices=["test", "val"])
    parser.add_argument("-n", "--num_classes", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"Path not found: {root}")
        return

    # Find all versioned experiment dirs (have test_stats.json)
    candidate_dirs = sorted(set(
        p.parent for p in root.rglob(f"{args.run_type}_video_predictions.csv")
    ))

    if not candidate_dirs:
        # Try treating root itself as the experiment dir
        candidate_dirs = [root]

    for d in candidate_dirs:
        diagnose_directory(d, run_type=args.run_type, num_classes=args.num_classes)

    print(f"\n{'=' * 60}")
    print("Tip: If model collapsed, check lr_image_encoder in config.")
    print("     Add --config configs/base_cfgs/runner_cfg/optim_sched/image_encoder/freeze-image.yaml")


if __name__ == "__main__":
    main()
