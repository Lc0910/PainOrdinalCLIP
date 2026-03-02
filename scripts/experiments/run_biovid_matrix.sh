#!/usr/bin/env bash
# ============================================================
# BioVid Part A - Baseline / CoOp / OrdinalCLIP experiments
# ============================================================
# Pain levels: BL1/PA1/PA2/PA3/PA4 -> 0~4 (num_ranks=5)
# val = test (no separate validation split available)
#
# Usage:
#   bash scripts/experiments/run_biovid_matrix.sh
#   bash scripts/experiments/run_biovid_matrix.sh --test_only
#
# Output per experiment:
#   results/biovid-{model}-{backbone}/version_N/
#     val_stats.json / test_stats.json                (frame-level)
#     val_video_stats.json / test_video_stats.json    (video-level)
#     val_video_predictions.csv / test_video_predictions.csv
# ============================================================

set -euo pipefail

DEFAULT_CFG="configs/default.yaml"
BIOVID_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid.yaml"

TEST_ONLY_FLAG=""
if [[ "${1:-}" == "--test_only" ]]; then
    TEST_ONLY_FLAG="--test_only"
fi

run_one() {
    local model_name="$1"
    local backbone_tag="$2"
    local image_cfg="$3"
    local text_cfg="$4"
    local output_dir="$5"
    shift 5

    echo "=== ${model_name} | ${backbone_tag} ==="
    python scripts/run.py \
        --config "${DEFAULT_CFG}" \
        --config "${BIOVID_CFG}" \
        --config "${image_cfg}" \
        --config "${text_cfg}" \
        "$@" \
        --output_dir "${output_dir}" \
        ${TEST_ONLY_FLAG}
}

for backbone in rn50 vitb16; do
    if [[ "${backbone}" == "rn50" ]]; then
        IMAGE_CFG="configs/base_cfgs/runner_cfg/model/image_encoder/clip-rn50.yaml"
        TEXT_CFG="configs/base_cfgs/runner_cfg/model/text_encoder/clip-rn50-cntprt.yaml"
        BACKBONE_TAG="rn50"
    else
        IMAGE_CFG="configs/base_cfgs/runner_cfg/model/image_encoder/clip-vitb16.yaml"
        TEXT_CFG="configs/base_cfgs/runner_cfg/model/text_encoder/clip-vitb16-cntprt.yaml"
        BACKBONE_TAG="vitb16"
    fi

    run_one \
        "Baseline" "${BACKBONE_TAG}" "${IMAGE_CFG}" "${TEXT_CFG}" "results/biovid-baseline-${BACKBONE_TAG}" \
        --config configs/base_cfgs/runner_cfg/model/baseline.yaml

    run_one \
        "CoOp" "${BACKBONE_TAG}" "${IMAGE_CFG}" "${TEXT_CFG}" "results/biovid-coop-${BACKBONE_TAG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config configs/base_cfgs/runner_cfg/model/prompt_learner/plain-prompt-learner.yaml \
        --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-biovid.yaml \
        --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-biovid.yaml

    run_one \
        "OrdinalCLIP" "${BACKBONE_TAG}" "${IMAGE_CFG}" "${TEXT_CFG}" "results/biovid-ordinalclip-${BACKBONE_TAG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config configs/base_cfgs/runner_cfg/model/prompt_learner/rank-prompt-learner.yaml \
        --config configs/base_cfgs/runner_cfg/model/prompt_learner/rank_prompt_learner/num_bsae_rank/num-base-rank-3.yaml \
        --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-biovid.yaml

done

echo "=== Parsing results (video-aware) ==="
python scripts/experiments/parse_video_results.py -d results/ -r test
python scripts/experiments/parse_video_results.py -d results/ -r val

echo "=== Done ==="
