#!/usr/bin/env bash
# ============================================================
# BioVid Baseline-only validation run
# ============================================================
# Purpose: verify that the three key fixes work correctly
#   Fix 1: last_project now in optimizer (was never trained)
#   Fix 2: CLIP normalization (was using ImageNet stats)
#   Fix 3: image encoder frozen (CoOp-style)
#
# Expected results after fixes:
#   frame-level acc_max:  35-50%   (was 22% ≈ random)
#   video-level acc_max:  40-55%   (aggregated over frames)
#   MAE:                  ~0.8-1.2  (was ~2.0 ≈ random)
#
# If accuracy is STILL ~22%:
#   → Run data verifier: python scripts/diagnosis/verify_biovid_data.py
#   → Run overfit test:  python scripts/diagnosis/overfit_test.py
#   → Check training acc in CSV logs (should rise above 50% if model learns)
#
# Usage:
#   bash scripts/experiments/run_biovid_baseline_only.sh
#   bash scripts/experiments/run_biovid_baseline_only.sh --max_epochs 30  # quick check
# ============================================================

set -euo pipefail

MAX_EPOCHS=100
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_epochs)
            MAX_EPOCHS="${2:?--max_epochs requires a value}"
            shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_baseline() {
    local backbone="$1"
    local img_cfg="$2"
    local txt_cfg="$3"
    local out_dir="results/biovid-baseline-${backbone}-v2"

    echo ""
    echo "=== Baseline | ${backbone} | output: ${out_dir} ==="
    python scripts/run.py \
        --config configs/default.yaml \
        --config configs/base_cfgs/data_cfg/datasets/biovid/biovid.yaml \
        --config configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml \
        --config "${img_cfg}" \
        --config "${txt_cfg}" \
        --config configs/base_cfgs/runner_cfg/model/baseline.yaml \
        --config configs/base_cfgs/runner_cfg/optim_sched/image_encoder/freeze-image.yaml \
        --config configs/base_cfgs/runner_cfg/optim_sched/prompt_learner/tune-ctx-rank-2e-3.yaml \
        --config configs/base_cfgs/runner_cfg/optim_sched/lr_decay/lr_decay_cosine_max_epochs_100.yaml \
        --output_dir "${out_dir}" \
        --cfg_options \
        trainer_cfg.max_epochs=${MAX_EPOCHS} \
        runner_cfg.optimizer_and_scheduler_cfg.lr_scheduler_cfg.max_epochs=${MAX_EPOCHS}
}

# RN50
run_baseline "rn50" \
    "configs/base_cfgs/runner_cfg/model/image_encoder/clip-rn50.yaml" \
    "configs/base_cfgs/runner_cfg/model/text_encoder/clip-rn50-cntprt.yaml"

# ViT-B/16
run_baseline "vitb16" \
    "configs/base_cfgs/runner_cfg/model/image_encoder/clip-vitb16.yaml" \
    "configs/base_cfgs/runner_cfg/model/text_encoder/clip-vitb16-cntprt.yaml"

echo ""
echo "=== Parsing results ==="
python scripts/experiments/parse_video_results.py -d results/ -r test
python scripts/experiments/parse_video_results.py -d results/ -r val

echo ""
echo "=== Diagnosis ==="
python scripts/diagnosis/diagnose_biovid.py -d results/biovid-baseline-rn50-v2
python scripts/diagnosis/diagnose_biovid.py -d results/biovid-baseline-vitb16-v2

echo ""
echo "=== Done ==="
