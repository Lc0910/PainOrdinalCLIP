#!/usr/bin/env bash
# ============================================================
# BioVid 2-class Siamese Stage 2: Pairwise Ranking Head
# ============================================================
#
# PREREQUISITE
# ------------
# Stage 1 best checkpoint must exist. Default:
#   results/biovid-2cls-coop-finetune1e5-vitb16/version_0/ckpts/
# Override with --backbone_ckpt <path>.
#
# PURPOSE
# -------
# Stage 1 achieved 65.8% video accuracy (CoOp-ft 1e-5 ViT-B/16).
# This falls in the 55-65% marginal zone, indicating CLIP features
# have pain signal but are not strongly discriminative.
#
# Stage 2 adds a Siamese pairwise ranking head (Fabio dissertation)
# to learn relative pain ordering, which may capture ordinal
# structure that softmax CE misses.
#
# EXPERIMENTS
# -----------
# E. Ranking-only   — BCE_rank only (no CE), MLP head, frozen backbone
#                     Tests pure pairwise signal in isolation.
# F. Joint-default  — BCE_rank + CE_a + CE_b, MLP head, frozen backbone
#                     Main experiment (P0-1 joint loss).
# G. Joint-unfreeze — Joint loss, MLP head, unfreeze backbone (lr=1e-6)
#                     Full fine-tune with Siamese objective.
# H. Linear-ablation— Joint loss, LINEAR head, frozen backbone
#                     Capacity lower-bound (P1-2).
#
# DECISION CRITERIA
# -----------------
#  Stage 2 video acc >= 68%  →  Siamese helps, proceed to 5-class.
#  Stage 2 video acc 63-68%  →  Marginal gain, investigate AU fusion.
#  Stage 2 video acc < 63%   →  Siamese not helpful, pivot approach.
#
# METRICS TRACKED (P1-3)
# ----------------------
# Training: pairwise accuracy, pairwise AUC (epoch-level)
# Val/Test: frame-level MAE/acc, video-level MAE/acc
#
# Usage:
#   bash scripts/experiments/run_biovid_2class_siamese.sh
#   bash scripts/experiments/run_biovid_2class_siamese.sh --backbone_ckpt path/to/best.ckpt
#   bash scripts/experiments/run_biovid_2class_siamese.sh --exp F  # run single experiment
# ============================================================
set -euo pipefail

SIAMESE_CFG="configs/siamese_default.yaml"
BIOVID2_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_2class.yaml"
CLIP_NORM="configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml"

# Stage 1 best checkpoint (CoOp-ft 1e-5 ViT-B/16)
BACKBONE_CKPT=""
MAX_EPOCHS="50"
PAIRS_PER_EPOCH="10000"
EXP_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbone_ckpt) BACKBONE_CKPT="${2:?}"; shift 2 ;;
        --max_epochs)    MAX_EPOCHS="${2:?}"; shift 2 ;;
        --pairs)         PAIRS_PER_EPOCH="${2:?}"; shift 2 ;;
        --exp)           EXP_FILTER="${2:?}"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Auto-detect backbone checkpoint if not specified
if [[ -z "${BACKBONE_CKPT}" ]]; then
    CKPT_DIR="results/biovid-2cls-coop-finetune1e5-vitb16"
    # Find the latest version with a best checkpoint
    BACKBONE_CKPT=$(find "${CKPT_DIR}" -name "epoch=*.ckpt" -not -name "last.ckpt" 2>/dev/null | head -1 || true)
    if [[ -z "${BACKBONE_CKPT}" ]]; then
        echo "ERROR: No Stage 1 checkpoint found in ${CKPT_DIR}/"
        echo "Run Stage 1 first or specify --backbone_ckpt <path>"
        exit 1
    fi
    echo "Auto-detected backbone checkpoint: ${BACKBONE_CKPT}"
fi

run_siamese() {
    local label="$1"; local output_dir="$2"; shift 2
    # Remaining "$@" are bare key=value pairs (NO --cfg_options prefix).
    # This function wraps them together with base options under a single
    # --cfg_options (argparse.REMAINDER captures everything after it).
    echo ""
    echo "============================================================"
    echo " ${label}"
    echo " output: ${output_dir}"
    echo " backbone: ${BACKBONE_CKPT}"
    echo "============================================================"
    python scripts/run_siamese.py \
        --config "${SIAMESE_CFG}" \
        --config "${BIOVID2_CFG}" \
        --config "${CLIP_NORM}" \
        --output_dir "${output_dir}" \
        --cfg_options \
        trainer_cfg.max_epochs=${MAX_EPOCHS} \
        runner_cfg.optimizer_and_scheduler_cfg.lr_scheduler_cfg.max_epochs=${MAX_EPOCHS} \
        runner_cfg.load_weights_cfg.backbone_ckpt_path="${BACKBONE_CKPT}" \
        data_cfg.pairs_per_epoch=${PAIRS_PER_EPOCH} \
        "$@"
}

should_run() {
    [[ "${EXP_FILTER}" == "all" || "${EXP_FILTER}" == "$1" ]]
}

# ============================================================
# E. Ranking-only (no CE) — isolate pairwise signal
# ============================================================
if should_run "E"; then
    run_siamese "2cls-E | Ranking-only MLP | frozen" \
        "results/biovid-2cls-siamese-E-ranking-only" \
        runner_cfg.loss_weights.ranking_loss=1.0 \
        runner_cfg.loss_weights.ce_loss_a=0.0 \
        runner_cfg.loss_weights.ce_loss_b=0.0 \
        runner_cfg.ranking_head_cfg.head_type=mlp
fi

# ============================================================
# F. Joint loss (default λ) — main experiment
# ============================================================
if should_run "F"; then
    run_siamese "2cls-F | Joint MLP | frozen" \
        "results/biovid-2cls-siamese-F-joint-mlp" \
        runner_cfg.loss_weights.ranking_loss=1.0 \
        runner_cfg.loss_weights.ce_loss_a=0.5 \
        runner_cfg.loss_weights.ce_loss_b=0.5 \
        runner_cfg.ranking_head_cfg.head_type=mlp
fi

# ============================================================
# G. Joint loss + unfreeze backbone (lr=1e-6)
# ============================================================
if should_run "G"; then
    run_siamese "2cls-G | Joint MLP | unfreeze backbone lr=1e-6" \
        "results/biovid-2cls-siamese-G-joint-unfreeze" \
        runner_cfg.loss_weights.ranking_loss=1.0 \
        runner_cfg.loss_weights.ce_loss_a=0.5 \
        runner_cfg.loss_weights.ce_loss_b=0.5 \
        runner_cfg.ranking_head_cfg.head_type=mlp \
        runner_cfg.freeze_backbone=false \
        runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_backbone=1.0e-06
fi

# ============================================================
# H. Linear head ablation — capacity lower-bound (P1-2)
# ============================================================
if should_run "H"; then
    run_siamese "2cls-H | Joint Linear | frozen" \
        "results/biovid-2cls-siamese-H-joint-linear" \
        runner_cfg.loss_weights.ranking_loss=1.0 \
        runner_cfg.loss_weights.ce_loss_a=0.5 \
        runner_cfg.loss_weights.ce_loss_b=0.5 \
        runner_cfg.ranking_head_cfg.head_type=linear
fi

echo ""
echo "=== Siamese Stage 2 Results ==="
python scripts/experiments/parse_video_results.py -d results/ -r test -f "siamese"
echo "=== Done ==="
