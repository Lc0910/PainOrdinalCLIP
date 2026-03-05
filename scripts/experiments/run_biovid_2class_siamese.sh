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
BIOVID2_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_2class_siamese.yaml"
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

# Helper: find best checkpoint by lowest metric value in filename.
# Filenames: epoch=XX-val_mae_max_metric=Y.YYYY.ckpt  (lower metric = better)
find_best_ckpt() {
    local search_dir="$1"
    find "${search_dir}" -name "epoch=*.ckpt" -not -name "last.ckpt" 2>/dev/null \
        | sort -t'=' -k3 -n \
        | head -1
}

# Auto-detect backbone checkpoint if not specified
if [[ -z "${BACKBONE_CKPT}" ]]; then
    CKPT_DIR="results/biovid-2cls-coop-finetune1e5-vitb16"
    BACKBONE_CKPT=$(find_best_ckpt "${CKPT_DIR}")
    if [[ -z "${BACKBONE_CKPT}" ]]; then
        echo "ERROR: No Stage 1 checkpoint found in ${CKPT_DIR}/"
        echo "Run Stage 1 first or specify --backbone_ckpt <path>"
        exit 1
    fi
    echo "Auto-detected backbone checkpoint: ${BACKBONE_CKPT}"
fi

run_siamese() {
    local label="$1"; local output_dir="$2"; shift 2
    # Remaining "$@" are a mix of:
    #   --flags   (e.g. --test_only)  → placed BEFORE --cfg_options
    #   key=value (e.g. runner_cfg.x=y) → placed AFTER --cfg_options
    # We separate them so argparse.REMAINDER doesn't swallow the flags.
    # NOTE: Only valueless flags (--test_only, --debug) are supported.
    # Flags with values (--config path) must be added to the python call directly.
    local extra_flags=()
    local cfg_opts=()
    for arg in "$@"; do
        if [[ "$arg" == --* ]]; then
            extra_flags+=("$arg")
        else
            cfg_opts+=("$arg")
        fi
    done

    echo ""
    echo "============================================================"
    echo " ${label}"
    echo " output: ${output_dir}"
    echo " backbone: ${BACKBONE_CKPT}"
    if [[ ${#extra_flags[@]} -gt 0 ]]; then
        echo " flags: ${extra_flags[*]}"
    fi
    echo "============================================================"
    python scripts/run_siamese.py \
        --config "${SIAMESE_CFG}" \
        --config "${BIOVID2_CFG}" \
        --config "${CLIP_NORM}" \
        --output_dir "${output_dir}" \
        "${extra_flags[@]}" \
        --cfg_options \
        trainer_cfg.max_epochs=${MAX_EPOCHS} \
        runner_cfg.optimizer_and_scheduler_cfg.lr_scheduler_cfg.max_epochs=${MAX_EPOCHS} \
        runner_cfg.load_weights_cfg.backbone_ckpt_path="${BACKBONE_CKPT}" \
        data_cfg.pairs_per_epoch=${PAIRS_PER_EPOCH} \
        "${cfg_opts[@]}"
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

# ============================================================
# I. Anchor inference on F checkpoint — ensemble α=0.5
#    TEST-ONLY: loads F's trained ranking head, computes anchors,
#    evaluates with ensemble α=0.5 (cls weight).
# ============================================================
if should_run "I"; then
    # Find F's best checkpoint (contains trained ranking head weights)
    F_CKPT=$(find_best_ckpt "results/biovid-2cls-siamese-F-joint-mlp")
    if [[ -z "${F_CKPT}" ]]; then
        echo "SKIP I: No F checkpoint found. Run experiment F first."
    else
        echo "Using F checkpoint: ${F_CKPT}"
        run_siamese "2cls-I | Anchor ensemble α=0.5 | F ckpt (test-only)" \
            "results/biovid-2cls-siamese-I-anchor-ens05" \
            --test_only \
            runner_cfg.ckpt_path="${F_CKPT}" \
            runner_cfg.ranking_head_cfg.head_type=mlp \
            runner_cfg.anchor_inference_cfg.enabled=true \
            runner_cfg.anchor_inference_cfg.ensemble_alpha=0.5 \
            runner_cfg.anchor_inference_cfg.anchor_mode=single
    fi
fi

# ============================================================
# J. Anchor inference on F checkpoint — rank-only (α=0.0)
#    TEST-ONLY: pure ranking head prediction without cls fusion.
# ============================================================
if should_run "J"; then
    F_CKPT=$(find_best_ckpt "results/biovid-2cls-siamese-F-joint-mlp")
    if [[ -z "${F_CKPT}" ]]; then
        echo "SKIP J: No F checkpoint found. Run experiment F first."
    else
        echo "Using F checkpoint: ${F_CKPT}"
        run_siamese "2cls-J | Anchor rank-only α=0.0 | F ckpt (test-only)" \
            "results/biovid-2cls-siamese-J-anchor-rank-only" \
            --test_only \
            runner_cfg.ckpt_path="${F_CKPT}" \
            runner_cfg.ranking_head_cfg.head_type=mlp \
            runner_cfg.anchor_inference_cfg.enabled=true \
            runner_cfg.anchor_inference_cfg.ensemble_alpha=0.0 \
            runner_cfg.anchor_inference_cfg.anchor_mode=single
    fi
fi

# ============================================================
# K. Alpha sweep on F checkpoint — α ∈ {0.0, 0.1, ..., 1.0}
#    TEST-ONLY: 11 runs with different ensemble weights.
#    α=1.0 → pure cls (baseline), α=0.0 → pure rank.
# ============================================================
if should_run "K"; then
    F_CKPT=$(find_best_ckpt "results/biovid-2cls-siamese-F-joint-mlp")
    if [[ -z "${F_CKPT}" ]]; then
        echo "SKIP K: No F checkpoint found. Run experiment F first."
    else
        echo "Using F checkpoint: ${F_CKPT}"
        for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
            alpha_tag=$(echo "${alpha}" | tr '.' '')
            run_siamese "2cls-K | Alpha sweep α=${alpha} | F ckpt (test-only)" \
                "results/biovid-2cls-siamese-K-alpha-${alpha_tag}" \
                --test_only \
                runner_cfg.ckpt_path="${F_CKPT}" \
                runner_cfg.ranking_head_cfg.head_type=mlp \
                runner_cfg.anchor_inference_cfg.enabled=true \
                runner_cfg.anchor_inference_cfg.ensemble_alpha=${alpha} \
                runner_cfg.anchor_inference_cfg.anchor_mode=single
        done
    fi
fi

echo ""
echo "=== Siamese Stage 2 Results ==="
python scripts/experiments/parse_video_results.py -d results/ -r test -f "siamese"
echo "=== Done ==="
