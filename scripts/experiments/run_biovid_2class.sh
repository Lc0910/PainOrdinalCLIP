#!/usr/bin/env bash
# ============================================================
# BioVid 2-class: BLN (0) vs PA4 (1)
# Frames with index < 50 already filtered out in data list.
# Chance level = 50%, literature (optical flow) = 71.8%
#
# PURPOSE
# -------
# 5-class experiments (BLN/PA1-PA4) all clustered at 20-23%
# (random = 20%), indicating CLIP image features cannot
# linearly separate intermediate pain levels.
#
# This experiment isolates the EASIEST binary contrast
# (no pain vs strongest pain) to answer:
#   "Can CLIP features separate pain AT ALL?"
#
# EXPERIMENTS (per backbone: rn50, vitb16)
# -----------------------------------------
# A. Baseline    — frozen CLIP + 2-class linear head
#                  Pure linear probe; diagnostic floor.
# B. CoOp-frozen — learn context prompts, freeze image encoder
#                  Tests whether prompt engineering helps.
# C. CoOp-ft 1e-4— learn prompts + aggressive image fine-tune
#                  Tests domain adaptation (may overfit).
# D. CoOp-ft 1e-5— learn prompts + conservative image fine-tune
#                  Safer adaptation, less risk of destroying
#                  pretrained features.
#
# EXPECTED RESULTS & DECISION CRITERIA
# -------------------------------------
#  > 65%  CLIP features are discriminative for pain.
#         → proceed with OrdinalCLIP / Siamese on 5-class.
#  55-65% Marginal signal exists but is weak.
#         → try Siamese ranking head or AU feature fusion.
#  < 55%  CLIP cannot separate even the extremes.
#         → pivot to AU-based features or temporal models.
#
# Usage:
#   bash scripts/experiments/run_biovid_2class.sh
#   bash scripts/experiments/run_biovid_2class.sh --backbone vitb16 --max_epochs 50
# ============================================================
set -euo pipefail

DEFAULT_CFG="configs/default.yaml"
BIOVID2_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_2class.yaml"
CLIP_NORM="configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml"
FREEZE_IMG="configs/base_cfgs/runner_cfg/optim_sched/image_encoder/freeze-image.yaml"
FINETUNE_1E5="configs/base_cfgs/runner_cfg/optim_sched/image_encoder/tune-image-1e-5.yaml"
FINETUNE_1E4="configs/base_cfgs/runner_cfg/optim_sched/image_encoder/tune-image-1e-4.yaml"
CTX_RANK_2E3="configs/base_cfgs/runner_cfg/optim_sched/prompt_learner/tune-ctx-rank-2e-3.yaml"
COSINE_LR="configs/base_cfgs/runner_cfg/optim_sched/lr_decay/lr_decay_cosine_max_epochs_100.yaml"
PLAIN_PL="configs/base_cfgs/runner_cfg/model/prompt_learner/plain-prompt-learner.yaml"
CTX_V1="configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-biovid.yaml"
RANK_V1="configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-biovid-2class.yaml"
KL_05="configs/base_cfgs/runner_cfg/loss/kl-weight-0.5.yaml"
KL_0="configs/base_cfgs/runner_cfg/loss/kl-weight-0.yaml"

MAX_EPOCHS="100"
BACKBONE_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_epochs) MAX_EPOCHS="${2:?}"; shift 2 ;;
        --backbone)   BACKBONE_FILTER="${2:?}"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

run_one() {
    local label="$1"; local output_dir="$2"; shift 2
    echo ""
    echo "============================================================"
    echo " ${label}"
    echo " output: ${output_dir}"
    echo "============================================================"
    python scripts/run.py \
        --config "${DEFAULT_CFG}" \
        --config "${BIOVID2_CFG}" \
        --config "${CLIP_NORM}" \
        "$@" \
        --output_dir "${output_dir}" \
        --cfg_options \
        trainer_cfg.max_epochs=${MAX_EPOCHS} \
        runner_cfg.optimizer_and_scheduler_cfg.lr_scheduler_cfg.max_epochs=${MAX_EPOCHS}
}

for backbone in rn50 vitb16; do
    [[ "${BACKBONE_FILTER}" != "all" && "${BACKBONE_FILTER}" != "${backbone}" ]] && continue

    if [[ "${backbone}" == "rn50" ]]; then
        IMG_CFG="configs/base_cfgs/runner_cfg/model/image_encoder/clip-rn50.yaml"
        TXT_CFG="configs/base_cfgs/runner_cfg/model/text_encoder/clip-rn50-cntprt.yaml"
    else
        IMG_CFG="configs/base_cfgs/runner_cfg/model/image_encoder/clip-vitb16.yaml"
        TXT_CFG="configs/base_cfgs/runner_cfg/model/text_encoder/clip-vitb16-cntprt.yaml"
    fi

    # A. Baseline — frozen CLIP + linear head (2 classes)
    run_one "2cls-A | Baseline | ${backbone}" \
        "results/biovid-2cls-baseline-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/baseline.yaml \
        --config "${FREEZE_IMG}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_0}"

    # B. CoOp-frozen
    run_one "2cls-B | CoOp-frozen | ${backbone}" \
        "results/biovid-2cls-coop-frozen-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${PLAIN_PL}" \
        --config "${CTX_V1}" \
        --config "${RANK_V1}" \
        --config "${FREEZE_IMG}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_0}"

    # C. CoOp + finetune image encoder (lr=1e-4, more aggressive)
    run_one "2cls-C | CoOp-finetune lr=1e-4 | ${backbone}" \
        "results/biovid-2cls-coop-finetune1e4-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${PLAIN_PL}" \
        --config "${CTX_V1}" \
        --config "${RANK_V1}" \
        --config "${FINETUNE_1E4}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_0}"

    # D. CoOp + finetune image encoder (lr=1e-5, conservative)
    run_one "2cls-D | CoOp-finetune lr=1e-5 | ${backbone}" \
        "results/biovid-2cls-coop-finetune1e5-${backbone}" \
        --config "${IMG_CFG}" \
        --config "${TXT_CFG}" \
        --config configs/base_cfgs/runner_cfg/model/ordinalclip.yaml \
        --config "${PLAIN_PL}" \
        --config "${CTX_V1}" \
        --config "${RANK_V1}" \
        --config "${FINETUNE_1E5}" \
        --config "${CTX_RANK_2E3}" \
        --config "${COSINE_LR}" \
        --config "${KL_0}"
done

echo ""
echo "=== 2-class results ==="
python scripts/experiments/parse_video_results.py -d results/ -r test
echo "=== Done ==="
