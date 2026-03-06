#!/usr/bin/env bash
# ============================================================
# BioVid 2-class Siamese Stage 2 — Fabio (2025) dissertation §5.3
# ============================================================
#
# PREREQUISITE
# ------------
# Stage 1 best checkpoint must exist. Default:
#   results/biovid-2cls-coop-finetune1e5-vitb16/version_0/ckpts/
# Override with --backbone_ckpt <path>.
#
# ARCHITECTURE (aligned to Fabio §5.3)
# --------------------------------------
# OrdinalCLIP backbone (frozen) → SharedMLP (D→256→128)
#   → RegressionHead (128→1→sigmoid, ŷ∈(0,1))
#   → ConcatRankingHead ([ei∥ej]→256→1)
#
# TRAINING LOSS
# -------------
#   L = mse_loss * L_mse + ranking_loss * L_hinge
#   L_mse   = MSE(ŷ_a, y_a) + MSE(ŷ_b, y_b)
#   L_hinge = mean max(0, |y_a-y_b|·margin_scale − η·s_ab)
#
# EXPERIMENTS
# -----------
# E. MSE-only       — mse_loss=1.0, ranking_loss=0.0 (regression only, no ranking)
#                     Tests pure regression signal in isolation.
# F. Joint-default  — mse_loss=1.0, ranking_loss=0.5, LINEAR head, frozen backbone
#                     Main experiment (Fabio paper defaults).
# G. Joint-unfreeze — Same as F, but unfreeze backbone (lr=1e-6)
#                     Full fine-tune with Siamese objective.
# H. MLP-ablation   — Joint loss, MLP head, frozen backbone
#                     Higher-capacity head variant.
#
# ANCHOR INFERENCE (post-training)
# ---------------------------------
# I. Anchor ensemble α=0.5 — F checkpoint + anchor-based ranking + cls fusion
# J. Anchor rank-only α=0.0 — F checkpoint + pure ranking head prediction
# K. Alpha sweep α∈{0.0..1.0} — Find optimal ensemble weight
#
# DECISION CRITERIA
# -----------------
#  Stage 2 video acc >= 68%  →  Siamese helps, proceed to 5-class.
#  Stage 2 video acc 63-68%  →  Marginal gain, investigate AU fusion.
#  Stage 2 video acc < 63%   →  Siamese not helpful, pivot approach.
#
# Usage:
#   bash scripts/experiments/run_biovid_2class_siamese.sh
#   bash scripts/experiments/run_biovid_2class_siamese.sh --backbone_ckpt path/to/best.ckpt
#   bash scripts/experiments/run_biovid_2class_siamese.sh --exp F  # run single experiment
# ============================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"
SIAMESE_CFG="configs/siamese_default.yaml"
BIOVID2_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_2class_siamese.yaml"
CLIP_NORM="configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml"

# Stage 1 best checkpoint (CoOp-ft 1e-5 ViT-B/16)
BACKBONE_CKPT=""
MAX_EPOCHS="100"       # CosineAnnealing T_max; EarlyStopping (patience=15) prevents overfitting
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
    # Separate --flags from key=value cfg_options
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
    ${PYTHON} scripts/run_siamese.py \
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
# E. MSE-only — pure regression, no ranking loss
#    Baseline: tests whether regression alone beats Stage 1 cls.
# ============================================================
if should_run "E"; then
    run_siamese "2cls-E | MSE-only | frozen" \
        "results/biovid-2cls-siamese-E-mse-only" \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.0 \
        runner_cfg.loss_weights.margin_scale=1.0 \
        runner_cfg.ranking_head_cfg.head_type=linear
fi

# ============================================================
# F. Joint loss (paper defaults) — main experiment
#    mse=1.0, hinge λ=0.5, linear head, frozen backbone
# ============================================================
if should_run "F"; then
    run_siamese "2cls-F | Joint linear | frozen (paper defaults)" \
        "results/biovid-2cls-siamese-F-joint-linear" \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.loss_weights.margin_scale=1.0 \
        runner_cfg.ranking_head_cfg.head_type=linear
fi

# ============================================================
# G. Joint loss + unfreeze backbone (lr=1e-6)
#    Full fine-tune: SharedMLP + heads + backbone.
# ============================================================
if should_run "G"; then
    run_siamese "2cls-G | Joint linear | unfreeze backbone lr=1e-6" \
        "results/biovid-2cls-siamese-G-joint-unfreeze" \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.loss_weights.margin_scale=1.0 \
        runner_cfg.ranking_head_cfg.head_type=linear \
        runner_cfg.freeze_backbone=false \
        runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_backbone=1.0e-06 \
        runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_siamese_heads=1.0e-04
fi

# ============================================================
# H. MLP head ablation — higher-capacity ConcatRankingHead
#    Compares linear (paper default) vs MLP for ranking head.
# ============================================================
if should_run "H"; then
    run_siamese "2cls-H | Joint MLP head | frozen" \
        "results/biovid-2cls-siamese-H-joint-mlp" \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.loss_weights.margin_scale=1.0 \
        runner_cfg.ranking_head_cfg.head_type=mlp \
        runner_cfg.ranking_head_cfg.hidden_dims=256
fi

# ============================================================
# I. Anchor inference on F checkpoint — ensemble α=0.5
#    TEST-ONLY: loads F's trained heads, computes anchors,
#    evaluates with ensemble α=0.5 (cls+rank fusion).
# ============================================================
if should_run "I"; then
    F_CKPT=$(find_best_ckpt "results/biovid-2cls-siamese-F-joint-linear")
    if [[ -z "${F_CKPT}" ]]; then
        echo "SKIP I: No F checkpoint found. Run experiment F first."
    else
        echo "Using F checkpoint: ${F_CKPT}"
        run_siamese "2cls-I | Anchor ensemble α=0.5 | F ckpt (test-only)" \
            "results/biovid-2cls-siamese-I-anchor-ens05" \
            --test_only \
            runner_cfg.ckpt_path="${F_CKPT}" \
            runner_cfg.ranking_head_cfg.head_type=linear \
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
    F_CKPT=$(find_best_ckpt "results/biovid-2cls-siamese-F-joint-linear")
    if [[ -z "${F_CKPT}" ]]; then
        echo "SKIP J: No F checkpoint found. Run experiment F first."
    else
        echo "Using F checkpoint: ${F_CKPT}"
        run_siamese "2cls-J | Anchor rank-only α=0.0 | F ckpt (test-only)" \
            "results/biovid-2cls-siamese-J-anchor-rank-only" \
            --test_only \
            runner_cfg.ckpt_path="${F_CKPT}" \
            runner_cfg.ranking_head_cfg.head_type=linear \
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
    F_CKPT=$(find_best_ckpt "results/biovid-2cls-siamese-F-joint-linear")
    if [[ -z "${F_CKPT}" ]]; then
        echo "SKIP K: No F checkpoint found. Run experiment F first."
    else
        echo "Using F checkpoint: ${F_CKPT}"
        for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
            alpha_tag=$(echo "${alpha}" | tr -d '.')
            run_siamese "2cls-K | Alpha sweep α=${alpha} | F ckpt (test-only)" \
                "results/biovid-2cls-siamese-K-alpha-${alpha_tag}" \
                --test_only \
                runner_cfg.ckpt_path="${F_CKPT}" \
                runner_cfg.ranking_head_cfg.head_type=linear \
                runner_cfg.anchor_inference_cfg.enabled=true \
                runner_cfg.anchor_inference_cfg.ensemble_alpha=${alpha} \
                runner_cfg.anchor_inference_cfg.anchor_mode=single
        done
    fi
fi

echo ""
echo "=== Siamese Stage 2 Results ==="
${PYTHON} scripts/experiments/parse_video_results.py -d results/ -r test -f "siamese"
echo "=== Done ==="
