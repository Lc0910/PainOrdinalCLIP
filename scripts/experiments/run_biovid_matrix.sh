#!/usr/bin/env bash
# ============================================================
# BioVid Part A - Comprehensive Experiment Matrix
# ============================================================
# Pain levels: BL1/PA1/PA2/PA3/PA4 -> 0~4 (num_ranks=5)
# val = test (no separate validation split available)
#
# Experiment groups:
#   A. Baseline       (frozen CLIP + linear probe)
#   B. CoOp-frozen    (freeze image encoder, learn context+rank, correct CoOp)
#   B2-B4. CoOp ablations: prompt variants, ordinal soft label
#   C. CoOp-finetune  (fine-tune image encoder with lr=1e-5)
#   D. OrdinalCLIP-frozen   (rank interpolation, frozen image encoder)
#   E. OrdinalCLIP-finetune (rank interpolation, lr_img=1e-5)
#
# Key fixes vs previous run:
#   1. freeze-image.yaml added to CoOp/OrdinalCLIP (lr_image_encoder was 1e-4)
#   2. clip-normalize.yaml added to ALL experiments
#      default.yaml uses ImageNet mean/std; CLIP requires its own normalization
#      (std differs by 17% → corrupts CLIP attention weights → ~random accuracy)
#
# Usage:
#   bash scripts/experiments/run_biovid_matrix.sh
#   bash scripts/experiments/run_biovid_matrix.sh --backbone rn50 --max_epochs 50
#   bash scripts/experiments/run_biovid_matrix.sh --test_only
#
# Output per experiment:
#   results/biovid-{group}-{backbone}/version_N/
#     val_stats.json / test_stats.json               (frame-level)
#     val_video_stats.json / test_video_stats.json   (video-level)
#     val_video_predictions.csv
# ============================================================

set -euo pipefail

# ---- Config shortcuts -------------------------------------
DEFAULT_CFG="configs/default.yaml"
BIOVID_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid.yaml"
# FIX: Use CLIP's own normalization instead of ImageNet default
CLIP_NORM="configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml"
FREEZE_IMG="configs/base_cfgs/runner_cfg/optim_sched/image_encoder/freeze-image.yaml"
FINETUNE_IMG_1E5="configs/base_cfgs/runner_cfg/optim_sched/image_encoder/tune-image-1e-5.yaml"
COSINE_LR="configs/base_cfgs/runner_cfg/optim_sched/lr_decay/lr_decay_cosine_max_epochs_100.yaml"
CTX_RANK_2E3="configs/base_cfgs/runner_cfg/optim_sched/prompt_learner/tune-ctx-rank-2e-3.yaml"
CTX_V1="configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-biovid.yaml"
CTX_V2="configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-biovid-v2.yaml"
CTX_V3="configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-biovid-v3.yaml"
RANK_V1="configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-biovid.yaml"
RANK_V2="configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-biovid-v2.yaml"
RANK_V3="configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-biovid-v3.yaml"
PLAIN_PL="configs/base_cfgs/runner_cfg/model/prompt_learner/plain-prompt-learner.yaml"
RANK_PL="configs/base_cfgs/runner_cfg/model/prompt_learner/rank-prompt-learner.yaml"
NUM_BASE3="configs/base_cfgs/runner_cfg/model/prompt_learner/rank_prompt_learner/num_bsae_rank/num-base-rank-3.yaml"
KL_05="configs/base_cfgs/runner_cfg/loss/kl-weight-0.5.yaml"
KL_0="configs/base_cfgs/runner_cfg/loss/kl-weight-0.yaml"
ORDINAL_SOFT="configs/base_cfgs/runner_cfg/loss/ordinal-soft-label.yaml"

TEST_ONLY_FLAG=""
MAX_EPOCHS="100"
BACKBONE_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test_only)
            TEST_ONLY_FLAG="--test_only"
            shift ;;
        --max_epochs)
            MAX_EPOCHS="${2:?--max_epochs requires a value}"
            shift 2 ;;
        --backbone)
            BACKBONE_FILTER="${2:?--backbone requires rn50|vitb16|all}"
            shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash $0 [--test_only] [--max_epochs N] [--backbone rn50|vitb16|all]"
            exit 1 ;;
    esac
done

# ---- Helper -----------------------------------------------
run_one() {
    local label="$1"
    local output_dir="$2"
    shift 2
    echo ""
    echo "============================================================"
    echo " ${label}"
    echo " output: ${output_dir}"
    echo "============================================================"
    python scripts/run.py \
        --config "${DEFAULT_CFG}" \
        --config "${BIOVID_CFG}" \
        --config "${CLIP_NORM}" \
        "$@" \
        --output_dir "${output_dir}" \
        ${TEST_ONLY_FLAG} \
        --cfg_options \
        trainer_cfg.max_epochs=${MAX_EPOCHS} \
        runner_cfg.optimizer_and_scheduler_cfg.lr_scheduler_cfg.max_epochs=${MAX_EPOCHS}
}

# ---- Experiment loop --------------------------------------
for backbone in rn50 vitb16; do
    if [[ "${BACKBONE_FILTER}" != "all" && "${BACKBONE_FILTER}" != "${backbone}" ]]; then
        continue
    fi

    if [[ "${backbone}" == "rn50" ]]; then
        IMG_CFG="configs/base_cfgs/runner_cfg/model/image_encoder/clip-rn50.yaml"
        TXT_CFG="configs/base_cfgs/runner_cfg/model/text_encoder/clip-rn50-cntprt.yaml"
    else
        IMG_CFG="configs/base_cfgs/runner_cfg/model/image_encoder/clip-vitb16.yaml"
        TXT_CFG="configs/base_cfgs/runner_cfg/model/text_encoder/clip-vitb16-cntprt.yaml"
    fi

    # ---- A. Baseline (frozen CLIP + linear probe) --------
    run_one "A | Baseline | ${backbone}" \
        "results/biovid-baseline-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/baseline.yaml \
        --config "${FREEZE_IMG}" \
        --config "${COSINE_LR}"

    # ---- B. CoOp-frozen (correct standard CoOp) ----------
    # FIX: freeze image encoder so prompt tokens learn properly
    run_one "B | CoOp-frozen [v1 ctx+rank] | ${backbone}" \
        "results/biovid-coop-frozen-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${PLAIN_PL}" \
        --config "${CTX_V1}" \
        --config "${RANK_V1}" \
        --config "${FREEZE_IMG}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_05}"

    # ---- B2. CoOp-frozen, CLIP-aligned prompt "a photo of a face showing" + v2 classnames
    run_one "B2 | CoOp-frozen [v2 photo-face ctx | v2 rank] | ${backbone}" \
        "results/biovid-coop-frozen-v2-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${PLAIN_PL}" \
        --config "${CTX_V2}" \
        --config "${RANK_V2}" \
        --config "${FREEZE_IMG}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_05}"

    # ---- B3. CoOp-frozen, expression-focused prompt + v3 classnames
    run_one "B3 | CoOp-frozen [v3 facial-expr ctx | v3 rank] | ${backbone}" \
        "results/biovid-coop-frozen-v3-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${PLAIN_PL}" \
        --config "${CTX_V3}" \
        --config "${RANK_V3}" \
        --config "${FREEZE_IMG}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_05}"

    # ---- B4. CoOp-frozen + ordinal Gaussian soft label CE (no KL)
    run_one "B4 | CoOp-frozen [ordinal-soft CE] | ${backbone}" \
        "results/biovid-coop-frozen-softlabel-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${PLAIN_PL}" \
        --config "${CTX_V1}" \
        --config "${RANK_V1}" \
        --config "${FREEZE_IMG}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_0}" \
        --config "${ORDINAL_SOFT}"

    # ---- C. CoOp with small image encoder LR (1e-5) ------
    run_one "C | CoOp-finetune [lr_img=1e-5] | ${backbone}" \
        "results/biovid-coop-finetune-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${PLAIN_PL}" \
        --config "${CTX_V1}" \
        --config "${RANK_V1}" \
        --config "${FINETUNE_IMG_1E5}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_05}"

    # ---- D. OrdinalCLIP-frozen (rank interpolation) ------
    run_one "D | OrdinalCLIP-frozen [num_base=3] | ${backbone}" \
        "results/biovid-ordinalclip-frozen-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${RANK_PL}" \
        --config "${NUM_BASE3}" \
        --config "${CTX_V1}" \
        --config "${FREEZE_IMG}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_05}"

    # ---- E. OrdinalCLIP with small image encoder LR ------
    run_one "E | OrdinalCLIP-finetune [lr_img=1e-5] | ${backbone}" \
        "results/biovid-ordinalclip-finetune-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${RANK_PL}" \
        --config "${NUM_BASE3}" \
        --config "${CTX_V1}" \
        --config "${FINETUNE_IMG_1E5}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_05}"

done

echo ""
echo "=== Parsing results (video-aware) ==="
python scripts/experiments/parse_video_results.py -d results/ -r test
python scripts/experiments/parse_video_results.py -d results/ -r val

echo ""
echo "=== Done ==="
