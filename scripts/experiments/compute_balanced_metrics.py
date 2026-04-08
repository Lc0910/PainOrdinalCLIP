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
    """Metrics for a single experiment.

    Contains both raw (test-set-distribution-weighted) and balanced
    (per-class-equal-weighted) metrics so the reader can see how much
    the imbalance is hiding.
    """

    name: str
    num_classes: int
    num_videos: int
    last_epoch: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    mae: float
    mae_continuous: float
    macro_mae: float
    macro_mae_continuous: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    per_class_support: List[int]
    per_class_mae: List[float]
    per_class_mae_continuous: List[float]
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
            "mae": round(self.mae, 4),
            "mae_continuous": round(self.mae_continuous, 4),
            "macro_mae": round(self.macro_mae, 4),
            "macro_mae_continuous": round(self.macro_mae_continuous, 4),
            "per_class_precision": [round(v, 4) for v in self.per_class_precision],
            "per_class_recall": [round(v, 4) for v in self.per_class_recall],
            "per_class_f1": [round(v, 4) for v in self.per_class_f1],
            "per_class_support": self.per_class_support,
            "per_class_mae": [round(v, 4) for v in self.per_class_mae],
            "per_class_mae_continuous": [round(v, 4) for v in self.per_class_mae_continuous],
            "confusion": self.confusion,
        }


def find_latest_version(experiment_dir: Path) -> Optional[Path]:
    """Return the newest version_N directory, or None if none exist."""
    versions = sorted(
        (p for p in experiment_dir.glob("version_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_", 1)[1].isdigit() else -1,
    )
    return versions[-1] if versions else None


def _iter_append_blocks(rows: List[dict]) -> Iterable[List[dict]]:
    """Split CSV rows into append blocks.

    Runner appends one full test block (one row per video) per test run.
    Blocks are delimited by the (epoch, ckpt_path) pair changing between
    consecutive rows, OR by a duplicate-header row reappearing in the
    stream (which happens if the CSV was manually concatenated).

    Each yielded block is a contiguous slice preserving file order.
    """
    if not rows:
        return
    block: List[dict] = []
    last_key: Optional[tuple] = None
    for row in rows:
        # Handle re-inserted header rows: `epoch` column literally equal to
        # the string "epoch". These break int parsing and always indicate a
        # new block boundary.
        if row.get("epoch") == "epoch":
            if block:
                yield block
                block = []
            last_key = None
            continue

        try:
            key = (int(row["epoch"]), str(row.get("ckpt_path", "")))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid epoch value in CSV row: {row!r}"
            ) from exc

        if last_key is not None and key != last_key:
            yield block
            block = []
        block.append(row)
        last_key = key

    if block:
        yield block


def load_predictions(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load the most recent test block from test_video_predictions.csv.

    Runner writes the CSV in append mode (runner.py lines 205-216), so each
    test run appends one full block of rows (one per video) with a fixed
    (epoch, ckpt_path) key. This function identifies the final block by
    scanning sequentially and slicing on (epoch, ckpt_path) changes.

    CSV schema: epoch, ckpt_path, video_id, gt, pred_exp, pred_max, n_frames

    Returns:
        y_true: [N] ground-truth labels (int).
        y_pred_exp: [N] continuous expectation predictions (pred_exp, float).
            This is Runner's "mae_exp" / "acc_exp" input.
        y_pred_max: [N] continuous argmax-averaged predictions (pred_max,
            float). This is Runner's "mae_max" / "acc_max" input. It is
            NOT the rounded class label — rounding happens only for the
            discrete classification metrics (acc / F1 / confusion).
        last_epoch: The epoch of the final test block.
    """
    rows: List[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    blocks = list(_iter_append_blocks(rows))
    if not blocks:
        raise ValueError(f"No valid blocks in {csv_path}")

    latest = blocks[-1]

    # Sanity check: video_id should be unique within a single test block.
    video_ids = [r["video_id"] for r in latest]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError(
            f"Duplicate video_id in latest block of {csv_path}: "
            f"{len(video_ids)} rows but only {len(set(video_ids))} unique IDs"
        )

    last_epoch = int(latest[0]["epoch"])

    y_true = np.array([int(r["gt"]) for r in latest], dtype=np.int64)
    y_pred_exp = np.array(
        [float(r["pred_exp"]) for r in latest], dtype=np.float64
    )
    y_pred_max = np.array(
        [float(r["pred_max"]) for r in latest], dtype=np.float64
    )

    # Fail loudly on NaN / inf instead of silently producing a NaN MAE.
    if not np.all(np.isfinite(y_pred_exp)):
        raise ValueError(
            f"Non-finite values in pred_exp of {csv_path} "
            f"(found {int((~np.isfinite(y_pred_exp)).sum())} bad values)"
        )
    if not np.all(np.isfinite(y_pred_max)):
        raise ValueError(
            f"Non-finite values in pred_max of {csv_path} "
            f"(found {int((~np.isfinite(y_pred_max)).sum())} bad values)"
        )

    return y_true, y_pred_exp, y_pred_max, last_epoch


def _per_class_float_mae(
    y_true: np.ndarray,
    y_pred_float: np.ndarray,
    labels: List[int],
) -> List[float]:
    """MAE computed separately per ground-truth class using float predictions.

    Matches Runner's video-level `mae_*_metric` definition: MAE is computed
    directly on continuous pred_exp / pred_max floats, without rounding.

    Returns one MAE value per class in `labels` order. If a class has zero
    ground-truth samples its MAE is NaN.
    """
    per_class: List[float] = []
    for c in labels:
        mask = y_true == c
        if mask.sum() == 0:
            per_class.append(float("nan"))
        else:
            per_class.append(
                float(np.mean(np.abs(y_pred_float[mask] - float(c))))
            )
    return per_class


def compute_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred_exp: np.ndarray,
    y_pred_max: np.ndarray,
    last_epoch: int,
    num_classes: Optional[int] = None,
) -> ExperimentMetrics:
    """Compute all metrics for one experiment.

    Args:
        name: Experiment identifier (directory name).
        y_true: [N] ground-truth labels (int).
        y_pred_exp: [N] continuous expectation predictions (float, Runner's
            pred_exp after video-level averaging). Used for mae_continuous.
        y_pred_max: [N] continuous argmax-averaged predictions (float,
            Runner's pred_max). Used for the raw MAE that matches Runner.
        last_epoch: Epoch of the predictions being evaluated.
        num_classes: Optional explicit number of classes. If None, inferred
            from `max(y_true, ceil(y_pred_max)) + 1` but ONLY for label
            enumeration purposes — predictions are never clipped.

    Metric definitions:
        - `mae` / `mae_continuous`: match Runner's `mae_max_metric` /
          `mae_exp_metric` exactly — float-based, no rounding, no clipping.
        - `macro_mae` / `macro_mae_continuous`: the balanced-subsample
          equivalent. Average of per-class float MAE.
        - `accuracy` / `balanced_accuracy` / `macro_f1`: use the discrete
          prediction obtained by rounding `y_pred_max`. This matches
          Runner's `acc_max_metric`.
        - Per-class metrics and the confusion matrix use the SAME label
          enumeration as the summary metrics to keep row/col semantics
          aligned.
    """
    # Fail loudly on empty input.
    if len(y_true) == 0:
        raise ValueError(f"{name}: empty predictions")

    # Discrete prediction for accuracy / F1 / confusion.
    # Use rint (round-half-to-even) on the float pred_max to reproduce
    # Runner's `round(pred_max) == gt` accuracy definition.
    y_pred_discrete = np.rint(y_pred_max).astype(np.int64)

    # Determine the authoritative label set:
    #   - Use caller's num_classes when provided.
    #   - Otherwise fall back to gt label space plus any out-of-range
    #     predictions, but surface a warning rather than silently masking.
    max_gt = int(y_true.max())
    min_gt = int(y_true.min())
    max_pred = int(np.ceil(y_pred_max.max()))
    min_pred = int(np.floor(y_pred_max.min()))

    if num_classes is None:
        inferred_k = max(max_gt, max_pred) + 1
        if min_pred < 0 or max_pred > max_gt:
            print(
                f"[WARN] {name}: predictions span [{min_pred}, {max_pred}] "
                f"while ground-truth spans [{min_gt}, {max_gt}]. "
                f"Extending label set to {inferred_k} classes. "
                f"MAE is still computed from float predictions (no clipping).",
                file=sys.stderr,
            )
        num_classes = inferred_k

    labels = list(range(num_classes))

    # Discrete classification metrics — pass explicit labels everywhere so
    # summary and per-class views agree.
    overall_acc = accuracy_score(y_true, y_pred_discrete)
    balanced_acc = balanced_accuracy_score(y_true, y_pred_discrete)
    macro_f1 = f1_score(
        y_true, y_pred_discrete, labels=labels, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred_discrete, labels=labels, average="weighted", zero_division=0
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_discrete, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred_discrete, labels=labels)

    # MAE metrics — use raw float predictions, no rounding, no clipping.
    # These match Runner's video-level mae_max_metric / mae_exp_metric.
    mae = float(np.mean(np.abs(y_pred_max - y_true)))
    mae_cont = float(np.mean(np.abs(y_pred_exp - y_true)))

    per_class_mae = _per_class_float_mae(y_true, y_pred_max, labels)
    per_class_mae_cont = _per_class_float_mae(y_true, y_pred_exp, labels)

    # Macro MAE: mean over classes that have at least one ground-truth
    # sample. Empty classes are skipped (documented below) rather than
    # silently zeroed.
    valid_mae = [v for v in per_class_mae if not np.isnan(v)]
    valid_mae_cont = [v for v in per_class_mae_cont if not np.isnan(v)]
    if not valid_mae or not valid_mae_cont:
        raise ValueError(f"{name}: no ground-truth samples in any class")
    macro_mae = float(np.mean(valid_mae))
    macro_mae_cont = float(np.mean(valid_mae_cont))

    return ExperimentMetrics(
        name=name,
        num_classes=num_classes,
        num_videos=int(len(y_true)),
        last_epoch=last_epoch,
        accuracy=float(overall_acc),
        balanced_accuracy=float(balanced_acc),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        mae=mae,
        mae_continuous=mae_cont,
        macro_mae=macro_mae,
        macro_mae_continuous=macro_mae_cont,
        per_class_precision=[float(v) for v in precision],
        per_class_recall=[float(v) for v in recall],
        per_class_f1=[float(v) for v in f1],
        per_class_support=[int(v) for v in support],
        per_class_mae=per_class_mae,
        per_class_mae_continuous=per_class_mae_cont,
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
    """Format a summary table for terminal output.

    Columns:
        Acc          - overall accuracy (imbalance-dominated)
        BalAcc       - balanced accuracy (per-class recall mean)
        MacroF1      - macro-averaged F1
        MAE          - raw MAE (imbalance-dominated)
        MacroMAE     - MAE averaged across classes (balanced-test equivalent)
        Epoch        - checkpoint epoch of the evaluated predictions
    """
    header_fmt = (
        "{name:<46s} {acc:>8s} {bal:>8s} {mf1:>8s} "
        "{mae:>8s} {mmae:>10s} {ep:>6s}"
    )
    row_fmt = (
        "{name:<46s} {acc:>8.4f} {bal:>8.4f} {mf1:>8.4f} "
        "{mae:>8.4f} {mmae:>10.4f} {ep:>6d}"
    )

    lines = []
    lines.append(header_fmt.format(
        name="Experiment",
        acc="Acc",
        bal="BalAcc",
        mf1="MacroF1",
        mae="MAE",
        mmae="MacroMAE",
        ep="Epoch",
    ))
    lines.append("-" * 100)
    for m in metrics_list:
        # macro_mae can be NaN if all classes are empty, but compute_metrics
        # already raises in that case, so this is defensive only.
        mae_val = m.mae if np.isfinite(m.mae) else float("nan")
        mmae_val = m.macro_mae if np.isfinite(m.macro_mae) else float("nan")
        lines.append(row_fmt.format(
            name=m.name[-46:],
            acc=m.accuracy,
            bal=m.balanced_accuracy,
            mf1=m.macro_f1,
            mae=mae_val,
            mmae=mmae_val,
            ep=m.last_epoch,
        ))
    return "\n".join(lines)


def format_per_class(metrics_list: List[ExperimentMetrics]) -> str:
    """Format per-class recall/F1/MAE details."""
    lines = []
    for m in metrics_list:
        lines.append("")
        lines.append(f"== {m.name} ==")
        lines.append(
            f"   videos={m.num_videos}  classes={m.num_classes}  epoch={m.last_epoch}"
        )
        lines.append(
            f"   raw MAE={m.mae:.4f}  macro MAE={m.macro_mae:.4f}  "
            f"(continuous: raw={m.mae_continuous:.4f}, macro={m.macro_mae_continuous:.4f})"
        )
        lines.append(
            f"   {'class':<8s} {'prec':>8s} {'recall':>8s} {'f1':>8s} "
            f"{'mae':>8s} {'support':>10s}"
        )
        for c in range(m.num_classes):
            mae_c = m.per_class_mae[c]
            mae_str = f"{mae_c:>8.4f}" if np.isfinite(mae_c) else f"{'n/a':>8s}"
            lines.append(
                f"   {c:<8d} "
                f"{m.per_class_precision[c]:>8.4f} "
                f"{m.per_class_recall[c]:>8.4f} "
                f"{m.per_class_f1[c]:>8.4f} "
                f"{mae_str} "
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
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=(
            "Explicit number of classes for the label space. If omitted, "
            "the script infers it from ground-truth and prediction bounds "
            "and prints a warning when predictions fall outside the gt "
            "range. Predictions are never clipped."
        ),
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
            y_true, y_pred_exp, y_pred_max, last_epoch = load_predictions(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {exp_dir.name}: {exc}", file=sys.stderr)
            continue

        try:
            metrics_list.append(
                compute_metrics(
                    exp_dir.name,
                    y_true,
                    y_pred_exp,
                    y_pred_max,
                    last_epoch,
                    num_classes=args.num_classes,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {exp_dir.name}: {exc}", file=sys.stderr)
            continue

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
