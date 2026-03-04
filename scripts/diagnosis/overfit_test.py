"""Overfit sanity test for BioVid.

Trains on a tiny subset (N=50 per class) for 200 epochs.
If training accuracy reaches >70%, the model CAN learn and
the issue is generalisation (subject-independent evaluation).
If training accuracy stays ~20%, there is a fundamental problem
(wrong labels, wrong normalization, data loading bug).

Usage:
    python scripts/diagnosis/overfit_test.py \
        --config configs/default.yaml \
        --config configs/base_cfgs/data_cfg/datasets/biovid/biovid.yaml \
        --config configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml \
        --config configs/base_cfgs/runner_cfg/model/image_encoder/clip-rn50.yaml \
        --config configs/base_cfgs/runner_cfg/model/text_encoder/clip-rn50-cntprt.yaml \
        --config configs/base_cfgs/runner_cfg/model/baseline.yaml \
        --config configs/base_cfgs/runner_cfg/optim_sched/image_encoder/tune-image.yaml \
        --n_per_class 50 \
        --max_epochs 200

Interpretation:
    train_acc > 70% after 200 epochs → model CAN learn; issue is generalisation
    train_acc < 30% after  50 epochs → fundamental problem (labels/norm/data)
"""

from __future__ import annotations  # Python 3.8 compatible PEP585 generics

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def build_overfit_subset(train_file: str, n_per_class: int, seed: int = 42) -> str:
    """Sample n_per_class samples per class from train_file, write to a tmp file."""
    random.seed(seed)
    by_class: dict[int, list[str]] = defaultdict(list)
    with open(train_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            labels = [int(x) for x in parts[1:]]
            label = labels[len(labels) // 2]  # median label
            by_class[label].append(line.strip())

    subset_lines = []
    for cls, lines in sorted(by_class.items()):
        sampled = random.sample(lines, min(n_per_class, len(lines)))
        subset_lines.extend(sampled)
        print(f"  Class {cls}: {len(sampled)} samples sampled (available: {len(lines)})")

    tmp_path = "/tmp/biovid_overfit_train.txt"
    with open(tmp_path, "w") as f:
        f.write("\n".join(subset_lines) + "\n")

    print(f"  Overfit subset: {len(subset_lines)} samples → {tmp_path}")
    return tmp_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", dest="configs", default=[])
    parser.add_argument("--n_per_class", type=int, default=50,
                        help="Number of samples per class for overfit test")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--output_dir", default="results/biovid-overfit-test")
    parser.add_argument("--train_file", default="data/biovid/train.txt",
                        help="Source train.txt to subsample from")
    args = parser.parse_args()

    print("=" * 60)
    print("BioVid Overfit Sanity Test")
    print(f"  n_per_class = {args.n_per_class}")
    print(f"  max_epochs  = {args.max_epochs}")
    print("=" * 60)

    # Build subset
    print("\n[1] Building overfit subset...")
    subset_path = build_overfit_subset(args.train_file, args.n_per_class)

    # Build run.py command
    configs_str = " ".join(f"--config {c}" for c in args.configs)
    cmd = (
        f"python scripts/run.py "
        f"{configs_str} "
        f"--output_dir {args.output_dir} "
        f"--cfg_options "
        f"trainer_cfg.max_epochs={args.max_epochs} "
        f"runner_cfg.optimizer_and_scheduler_cfg.lr_scheduler_cfg.max_epochs={args.max_epochs} "
        f"data_cfg.train_data_file={subset_path} "
        f"data_cfg.val_data_file={subset_path} "  # overfit: val=train
        f"data_cfg.test_data_file={subset_path} "
        f"data_cfg.train_dataloder_cfg.batch_size=16 "
        f"data_cfg.eval_dataloder_cfg.batch_size=16"
    )

    print("\n[2] Running overfit experiment:")
    print(f"  {cmd}")
    print()

    ret = os.system(cmd)
    if ret != 0:
        print(f"\n⚠️  Experiment failed (exit code {ret})")
        return

    # Read results
    result_dir = Path(args.output_dir)
    stats_files = sorted(result_dir.rglob("val_stats.json"))
    if not stats_files:
        print("No val_stats.json found.")
        return

    print("\n[3] Results (should overfit to training data):")
    for sf in stats_files[-1:]:
        lines = sf.read_text().strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
            acc_max = last.get("acc_max_metric", "?")
            acc_exp = last.get("acc_exp_metric", "?")
            mae_max = last.get("mae_max_metric", "?")
            # guard against missing fields returning the '?' sentinel (a str, not float)
            fmt_acc_max = f"{acc_max:.4f}" if isinstance(acc_max, float) else str(acc_max)
            fmt_acc_exp = f"{acc_exp:.4f}" if isinstance(acc_exp, float) else str(acc_exp)
            fmt_mae_max = f"{mae_max:.4f}" if isinstance(mae_max, float) else str(mae_max)
            print(f"  val(=train) acc_max={fmt_acc_max}  acc_exp={fmt_acc_exp}  mae_max={fmt_mae_max}")

            if isinstance(acc_max, float):
                if acc_max > 0.70:
                    print("\n  ✅ Model CAN overfit. Issue is generalisation across subjects.")
                    print("     Focus on: CLIP normalization, frozen image encoder, video-level eval.")
                elif acc_max > 0.40:
                    print("\n  ⚠️  Partial learning. Check normalization & learning rate.")
                else:
                    print("\n  ❌ Model CANNOT overfit. Fundamental problem.")
                    print("     Check: label format (0-indexed?), image paths, CLIP normalization.")


if __name__ == "__main__":
    main()
