#!/usr/bin/env bash
# ============================================================
# BioVid 5-class Siamese Stage 2 — Fabio (2025) dissertation §5.3
# ============================================================
#
# PREREQUISITE
# ------------
# Stage 1 best checkpoint (5-class OrdinalCLIP) must exist.
# Override with --backbone_ckpt <path>.
#
# ARCHITECTURE (aligned to Fabio §5.3)
# --------------------------------------
# OrdinalCLIP backbone (frozen) → SharedMLP (D→256→128)
#   → RegressionHead (128→1→sigmoid, ŷ∈(0,4))
#   → ConcatRankingHead ([ei∥ej]→256→1)
#
# TRAINING LOSS
# -------------
#   L = mse_loss * L_mse + ranking_loss * L_hinge
#   L_mse   = MSE(ŷ_a, y_a) + MSE(ŷ_b, y_b)
#   L_hinge = mean max(0, |y_a-y_b|·margin_scale − η·s_ab)
#
# Multi-class anchor inference uses cumulative link approach:
#   s_k = concat_ranking_head([e_x ∥ e_anchor_k]) for k=0..3
#   P(class=0) = 1 - σ(s_0)
#   P(class=k) = σ(s_{k-1}) - σ(s_k)
#   P(class=4) = σ(s_3)
#
# EXPERIMENTS
# -----------
# L. Joint-default  — mse+hinge, linear head, frozen backbone (paper defaults)
# M. Joint-unfreeze — Joint loss, linear head, unfreeze backbone (lr=1e-6)
# N. MLP-ablation   — Joint loss, MLP head, frozen backbone
# O. Anchor ensemble α=0.5 (test-only on L checkpoint)
# P. Alpha sweep α∈{0.0, 0.1, ..., 1.0} (test-only on L checkpoint)
#
# Usage:
#   bash scripts/experiments/run_biovid_5class_siamese.sh
#   bash scripts/experiments/run_biovid_5class_siamese.sh --backbone_ckpt path/to/best.ckpt
#   bash scripts/experiments/run_biovid_5class_siamese.sh --exp L  # run single experiment
# ============================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"
SIAMESE_CFG="configs/siamese_default.yaml"
BIOVID5_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_siamese.yaml"
CLIP_NORM="configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml"

# Stage 1 best checkpoint (5-class)
BACKBONE_CKPT=""
MAX_EPOCHS="100"       # CosineAnnealing T_max; EarlyStopping (patience=15) prevents overfitting
PAIRS_PER_EPOCH="20000"
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
    for CKPT_DIR in \
        "results/biovid-5cls-coop-finetune1e5-vitb16" \
        "results/biovid-ordinalclip-vitb16" \
        "results/biovid-5cls-ordinalclip"; do
        BACKBONE_CKPT=$(find_best_ckpt "${CKPT_DIR}" 2>/dev/null || true)
        if [[ -n "${BACKBONE_CKPT}" ]]; then
            break
        fi
    done
    if [[ -z "${BACKBONE_CKPT}" ]]; then
        echo "ERROR: No Stage 1 (5-class) checkpoint found."
        echo "Run Stage 1 first or specify --backbone_ckpt <path>"
        exit 1
    fi
    echo "Auto-detected backbone checkpoint: ${BACKBONE_CKPT}"
fi

run_siamese() {
    local label="$1"; local output_dir="$2"; shift 2
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
        --config "${BIOVID5_CFG}" \
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
# L. Joint loss (paper defaults) — main experiment
#    mse=1.0, hinge λ=0.5, linear head, frozen backbone
# ============================================================
if should_run "L"; then
    run_siamese "5cls-L | Joint linear | frozen (paper defaults)" \
        "results/biovid-5cls-siamese-L-joint-linear" \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.loss_weights.margin_scale=1.0 \
        runner_cfg.ranking_head_cfg.head_type=linear
fi

# ============================================================
# M. Joint loss + unfreeze backbone (lr=1e-6)
#    Full fine-tune: SharedMLP + heads + backbone.
# ============================================================
if should_run "M"; then
    run_siamese "5cls-M | Joint linear | unfreeze backbone lr=1e-6" \
        "results/biovid-5cls-siamese-M-joint-unfreeze" \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.loss_weights.margin_scale=1.0 \
        runner_cfg.ranking_head_cfg.head_type=linear \
        runner_cfg.freeze_backbone=false \
        runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_backbone=1.0e-06 \
        runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_siamese_heads=1.0e-04
fi

# ============================================================
# N. MLP head ablation — higher-capacity ConcatRankingHead
#    Compares linear (paper default) vs MLP for ranking head.
# ============================================================
if should_run "N"; then
    run_siamese "5cls-N | Joint MLP head | frozen" \
        "results/biovid-5cls-siamese-N-joint-mlp" \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.loss_weights.margin_scale=1.0 \
        runner_cfg.ranking_head_cfg.head_type=mlp \
        runner_cfg.ranking_head_cfg.hidden_dims=256
fi

# ============================================================
# O. Anchor inference on L checkpoint — ensemble α=0.5
#    TEST-ONLY: cumulative link multi-class anchor inference
#    Computes 5 class anchors in SharedMLP embedding space.
# ============================================================
if should_run "O"; then
    L_CKPT=$(find_best_ckpt "results/biovid-5cls-siamese-L-joint-linear")
    if [[ -z "${L_CKPT}" ]]; then
        echo "SKIP O: No L checkpoint found. Run experiment L first."
    else
        echo "Using L checkpoint: ${L_CKPT}"
        run_siamese "5cls-O | Anchor ensemble α=0.5 | L ckpt (test-only)" \
            "results/biovid-5cls-siamese-O-anchor-ens05" \
            --test_only \
            runner_cfg.ckpt_path="${L_CKPT}" \
            runner_cfg.ranking_head_cfg.head_type=linear \
            runner_cfg.anchor_inference_cfg.enabled=true \
            runner_cfg.anchor_inference_cfg.ensemble_alpha=0.5
    fi
fi

# ============================================================
# P. Alpha sweep on L checkpoint — α ∈ {0.0, 0.1, ..., 1.0}
#    TEST-ONLY: 11 runs with different ensemble weights.
#    α=1.0 → pure cls (baseline), α=0.0 → pure rank.
# ============================================================
if should_run "P"; then
    L_CKPT=$(find_best_ckpt "results/biovid-5cls-siamese-L-joint-linear")
    if [[ -z "${L_CKPT}" ]]; then
        echo "SKIP P: No L checkpoint found. Run experiment L first."
    else
        echo "Using L checkpoint: ${L_CKPT}"
        for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
            alpha_tag=$(echo "${alpha}" | tr -d '.')
            run_siamese "5cls-P | Alpha sweep α=${alpha} | L ckpt (test-only)" \
                "results/biovid-5cls-siamese-P-alpha-${alpha_tag}" \
                --test_only \
                runner_cfg.ckpt_path="${L_CKPT}" \
                runner_cfg.ranking_head_cfg.head_type=linear \
                runner_cfg.anchor_inference_cfg.enabled=true \
                runner_cfg.anchor_inference_cfg.ensemble_alpha=${alpha}
        done
    fi
fi

echo ""
echo "=== 5-class Siamese Stage 2 Results ==="
${PYTHON} scripts/experiments/parse_video_results.py -d results/ -r test -f "5cls-siamese"
echo "=== Done ==="
