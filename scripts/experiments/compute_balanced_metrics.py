"""Compute balanced accuracy, macro F1, and per-class metrics from saved
test predictions without re-running test.

Reads `test_video_predictions.csv` (written by Runner._video_level_aggregation)
which contains one row per video with columns:
    epoch, ckpt_path, video_id, gt, pred_exp, pred_max, n_frames

For each experiment directory, computes:
    - overall accuracy
    - balanced accuracy (unweighted mean of per-class recall)
    - macro F1 (unweighted mean of per-class F1)
    - per-class accuracy / precision / recall / F1
    - confusion matrix

Usage:
    python scripts/experiments/compute_balanced_metrics.py
    python scripts/experiments/compute_balanced_metrics.py --results-dir results
    python scripts/experiments/compute_balanced_metrics.py \
        --pattern "biovid-3cls-*"
    python scripts/experiments/compute_balanced_metrics.py \
        --experiments biovid-3cls-ordinalclip-ft1e5 biovid-3cls-baseline-rn50-ft1e5

The script is read-only: it does not load any model checkpoints and requires
no GPU. It just parses the predictions CSV that was saved during test.

Requires: pandas, scikit-learn (both standard dependencies, no new installs).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
except ImportError:
    print(
        "ERROR: scikit-learn is required. Install with: pip install scikit-learn",
        file=sys.stderr,
    )
    sys.exit(1)


@dataclass(frozen=True)
class ExperimentMetrics:
    """Metrics for a single experiment."""

    name: str
    num_classes: int
    num_videos: int
    last_epoch: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    per_class_support: List[int]
    confusion: List[List[int]]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "num_videos": self.num_videos,
            "last_epoch": self.last_epoch,
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class_precision": [round(v, 4) for v in self.per_class_precision],
            "per_class_recall": [round(v, 4) for v in self.per_class_recall],
            "per_class_f1": [round(v, 4) for v in self.per_class_f1],
            "per_class_support": self.per_class_support,
            "confusion": self.confusion,
        }


def find_latest_version(experiment_dir: Path) -> Optional[Path]:
    """Return the newest version_N directory, or None if none exist."""
    versions = sorted(
        (p for p in experiment_dir.glob("version_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_", 1)[1].isdigit() else -1,
    )
    return versions[-1] if versions else None


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load the latest-epoch predictions from a test_video_predictions.csv.

    CSV schema: epoch, ckpt_path, video_id, gt, pred_exp, pred_max, n_frames

    Returns:
        y_true: [N] ground-truth labels (int)
        y_pred: [N] predicted labels (rounded pred_max)
        last_epoch: The epoch number of the predictions used.
    """
    rows: List[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    # Keep only rows from the latest epoch (Runner appends all test runs).
    last_epoch = max(int(r["epoch"]) for r in rows)
    latest = [r for r in rows if int(r["epoch"]) == last_epoch]

    y_true = np.array([int(r["gt"]) for r in latest], dtype=np.int64)
    y_pred = np.array(
        [int(round(float(r["pred_max"]))) for r in latest],
        dtype=np.int64,
    )
    return y_true, y_pred, last_epoch


def compute_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    last_epoch: int,
) -> ExperimentMetrics:
    num_classes = int(max(y_true.max(), y_pred.max())) + 1

    # Clip predictions into the valid label range (rounded pred_max may be
    # out of range, especially when the model has not converged).
    y_pred_clipped = np.clip(y_pred, 0, num_classes - 1)

    overall_acc = accuracy_score(y_true, y_pred_clipped)
    balanced_acc = balanced_accuracy_score(y_true, y_pred_clipped)
    macro_f1 = f1_score(y_true, y_pred_clipped, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred_clipped, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_clipped, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred_clipped, labels=list(range(num_classes)))

    return ExperimentMetrics(
        name=name,
        num_classes=num_classes,
        num_videos=int(len(y_true)),
        last_epoch=last_epoch,
        accuracy=float(overall_acc),
        balanced_accuracy=float(balanced_acc),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class_precision=[float(v) for v in precision],
        per_class_recall=[float(v) for v in recall],
        per_class_f1=[float(v) for v in f1],
        per_class_support=[int(v) for v in support],
        confusion=cm.tolist(),
    )


def scan_experiments(
    results_dir: Path,
    names: Optional[List[str]],
    pattern: Optional[str],
) -> List[Path]:
    """Locate experiment directories to evaluate."""
    if names:
        dirs = [results_dir / n for n in names]
        missing = [d for d in dirs if not d.is_dir()]
        if missing:
            print(
                "WARNING: missing experiment directories: "
                + ", ".join(str(d) for d in missing),
                file=sys.stderr,
            )
        return [d for d in dirs if d.is_dir()]

    if pattern:
        return sorted(p for p in results_dir.glob(pattern) if p.is_dir())

    return sorted(p for p in results_dir.iterdir() if p.is_dir())


def format_table(metrics_list: List[ExperimentMetrics]) -> str:
    """Format a summary table for terminal output."""
    header_fmt = (
        "{name:<48s} {acc:>10s} {bal:>12s} {mf1:>10s} {wf1:>10s} {ep:>6s}"
    )
    row_fmt = (
        "{name:<48s} {acc:>10.4f} {bal:>12.4f} {mf1:>10.4f} {wf1:>10.4f} {ep:>6d}"
    )

    lines = []
    lines.append(header_fmt.format(
        name="Experiment",
        acc="Acc",
        bal="BalancedAcc",
        mf1="MacroF1",
        wf1="WeightedF1",
        ep="Epoch",
    ))
    lines.append("-" * 100)
    for m in metrics_list:
        lines.append(row_fmt.format(
            name=m.name[-48:],
            acc=m.accuracy,
            bal=m.balanced_accuracy,
            mf1=m.macro_f1,
            wf1=m.weighted_f1,
            ep=m.last_epoch,
        ))
    return "\n".join(lines)


def format_per_class(metrics_list: List[ExperimentMetrics]) -> str:
    """Format per-class recall/F1 details."""
    lines = []
    for m in metrics_list:
        lines.append("")
        lines.append(f"== {m.name} ==")
        lines.append(f"   videos={m.num_videos}  classes={m.num_classes}  epoch={m.last_epoch}")
        lines.append(f"   {'class':<8s} {'prec':>8s} {'recall':>8s} {'f1':>8s} {'support':>10s}")
        for c in range(m.num_classes):
            lines.append(
                f"   {c:<8d} "
                f"{m.per_class_precision[c]:>8.4f} "
                f"{m.per_class_recall[c]:>8.4f} "
                f"{m.per_class_f1[c]:>8.4f} "
                f"{m.per_class_support[c]:>10d}"
            )
        lines.append("   confusion matrix (rows=gt, cols=pred):")
        for row in m.confusion:
            lines.append("     " + " ".join(f"{v:>6d}" for v in row))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute balanced accuracy / macro F1 from test_video_predictions.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory (default: results)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Explicit experiment names (under results/). Overrides --pattern.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Glob pattern under results/, e.g. 'biovid-3cls-*'",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to dump all metrics as JSON",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Silently skip experiments without predictions CSV",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"ERROR: results dir not found: {args.results_dir}", file=sys.stderr)
        return 2

    candidates = scan_experiments(
        args.results_dir, args.experiments, args.pattern
    )
    if not candidates:
        print("No experiment directories matched.", file=sys.stderr)
        return 1

    metrics_list: List[ExperimentMetrics] = []
    for exp_dir in candidates:
        latest = find_latest_version(exp_dir)
        if latest is None:
            if not args.skip_empty:
                print(f"[SKIP] {exp_dir.name}: no version_N directory")
            continue

        csv_path = latest / "test_video_predictions.csv"
        if not csv_path.is_file():
            if not args.skip_empty:
                print(f"[SKIP] {exp_dir.name}: missing test_video_predictions.csv")
            continue

        try:
            y_true, y_pred, last_epoch = load_predictions(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {exp_dir.name}: {exc}", file=sys.stderr)
            continue

        metrics_list.append(
            compute_metrics(exp_dir.name, y_true, y_pred, last_epoch)
        )

    if not metrics_list:
        print("No experiments with valid predictions found.", file=sys.stderr)
        return 1

    print(format_table(metrics_list))
    print(format_per_class(metrics_list))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([m.to_dict() for m in metrics_list], indent=2)
        )
        print(f"\nMetrics JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
