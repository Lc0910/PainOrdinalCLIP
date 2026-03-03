#!/usr/bin/env bash
# ============================================================
# BioVid Part A - Minimal Ablation (OrdinalCLIP only)
# ============================================================
# 1) Original config
# 2) num_base_ranks=5 + init_rank
# 3) (2) + lr_image=1e-5 + weak augment
#
# Usage:
#   bash scripts/experiments/run_biovid_min_ablation.sh
#   bash scripts/experiments/run_biovid_min_ablation.sh --backbone vitb16 --max_epochs 50
#   bash scripts/experiments/run_biovid_min_ablation.sh --test_only --max_epochs 50
# ============================================================

set -euo pipefail

DEFAULT_CFG="configs/default.yaml"
BIOVID_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid.yaml"

BACKBONE="rn50"
MAX_EPOCHS="50"
TEST_ONLY_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbone)
            BACKBONE="${2:-}"
            if [[ -z "${BACKBONE}" ]]; then
                echo "Error: --backbone requires rn50 or vitb16"
                exit 1
            fi
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="${2:-}"
            if [[ -z "${MAX_EPOCHS}" ]]; then
                echo "Error: --max_epochs requires a value"
                exit 1
            fi
            shift 2
            ;;
        --test_only)
            TEST_ONLY_FLAG="--test_only"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/experiments/run_biovid_min_ablation.sh [--backbone rn50|vitb16] [--max_epochs 50] [--test_only]"
            exit 1
            ;;
    esac
done

if [[ "${BACKBONE}" == "rn50" ]]; then
    IMAGE_CFG="configs/base_cfgs/runner_cfg/model/image_encoder/clip-rn50.yaml"
    TEXT_CFG="configs/base_cfgs/runner_cfg/model/text_encoder/clip-rn50-cntprt.yaml"
    BACKBONE_TAG="rn50"
elif [[ "${BACKBONE}" == "vitb16" ]]; then
    IMAGE_CFG="configs/base_cfgs/runner_cfg/model/image_encoder/clip-vitb16.yaml"
    TEXT_CFG="configs/base_cfgs/runner_cfg/model/text_encoder/clip-vitb16-cntprt.yaml"
    BACKBONE_TAG="vitb16"
else
    echo "Unsupported backbone: ${BACKBONE}. Choose rn50 or vitb16."
    exit 1
fi

run_ablation() {
    local exp_name="$1"
    local output_dir="$2"
    shift 2

    echo "=== ${exp_name} | ${BACKBONE_TAG} | epochs=${MAX_EPOCHS} ==="
    python scripts/run.py \
        --config "${DEFAULT_CFG}" \
        --config "${BIOVID_CFG}" \
        --config "${IMAGE_CFG}" \
        --config "${TEXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config configs/base_cfgs/runner_cfg/model/prompt_learner/rank-prompt-learner.yaml \
        --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-biovid.yaml \
        "$@" \
        --output_dir "${output_dir}" \
        ${TEST_ONLY_FLAG} \
        --cfg_options \
        trainer_cfg.max_epochs=${MAX_EPOCHS} \
        runner_cfg.optimizer_and_scheduler_cfg.lr_scheduler_cfg.max_epochs=${MAX_EPOCHS}
}

# 1) Original config (num_base_ranks=3)
run_ablation \
    "Ablation-1 Original" \
    "results/biovid-ablation1-original-${BACKBONE_TAG}" \
    --config configs/base_cfgs/runner_cfg/model/prompt_learner/rank_prompt_learner/num_bsae_rank/num-base-rank-3.yaml

# 2) num_base_ranks=5 + init_rank
run_ablation \
    "Ablation-2 base5+initrank" \
    "results/biovid-ablation2-base5-initrank-${BACKBONE_TAG}" \
    --config configs/base_cfgs/runner_cfg/model/prompt_learner/rank_prompt_learner/num_bsae_rank/num-base-rank-5.yaml \
    --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-biovid.yaml

# 3) (2) + lr_image=1e-5 + weak augment
run_ablation \
    "Ablation-3 +lr1e-5+weakaug" \
    "results/biovid-ablation3-base5-initrank-lr1e5-weakaug-${BACKBONE_TAG}" \
    --config configs/base_cfgs/runner_cfg/model/prompt_learner/rank_prompt_learner/num_bsae_rank/num-base-rank-5.yaml \
    --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-biovid.yaml \
    --config configs/base_cfgs/runner_cfg/optim_sched/image_encoder/tune-image-1e-5.yaml \
    --config configs/base_cfgs/data_cfg/transforms/weak-augment.yaml

echo "=== Parse ablation results ==="
python scripts/experiments/parse_video_results.py -d results/ -r test
python scripts/experiments/parse_video_results.py -d results/ -r val

echo "=== Done ==="
