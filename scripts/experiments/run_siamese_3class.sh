#!/bin/bash
# ==========================================================================
# Siamese Stage 2 Ranking — BioVid 3-class
# ==========================================================================
#
# Trains Siamese ranking heads on top of a frozen Stage 1 OrdinalCLIP
# backbone. Uses the best 3-class OrdinalCLIP checkpoint (ft1e5) as the
# backbone.
#
# Three experimental variants (Fabio 2025 §5):
#   E-mse-only    : regression head only, no ranking loss
#   E-rank-only   : ranking head only, no MSE loss
#   F-joint       : joint MSE + Hinge ranking (paper default)
#
# Usage:
#   bash scripts/experiments/run_siamese_3class.sh              # run all
#   bash scripts/experiments/run_siamese_3class.sh --dry-run    # preview
#   bash scripts/experiments/run_siamese_3class.sh --skip-done  # resume
#   bash scripts/experiments/run_siamese_3class.sh --only joint # one variant
#
# Prerequisites:
#   1. 3-class Stage 1 experiments completed:
#        results/biovid-3cls-ordinalclip-ft1e5/version_N/ckpts/*.ckpt
#   2. python scripts/data/build_3class.py    # if not already done
# ==========================================================================

set -euo pipefail

# --- Config ---
SIAMESE_DEFAULT_CFG="configs/siamese_default.yaml"
DATA_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_3class_siamese.yaml"
TRANSFORMS_CFG="configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml"

# Stage 1 backbone source experiment
BACKBONE_SOURCE="results/biovid-3cls-ordinalclip-ft1e5"

RESULT_BASE="results"
DRY_RUN=false
SKIP_DONE=false
ONLY_VARIANT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true;    shift ;;
        --skip-done) SKIP_DONE=true;  shift ;;
        --only)      ONLY_VARIANT="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Helpers ---
find_latest_version() {
    local base_dir="$1"
    local latest
    latest=$(ls -d "${base_dir}"/version_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$latest" ]; then
        echo "$latest"
    else
        echo "${base_dir}/version_0"
    fi
}

find_best_ckpt() {
    # Find the best checkpoint (smallest val_mae_max_metric) in a version dir.
    # Returns empty string if no ckpt found.
    local version_dir="$1"
    local ckpt_dir="${version_dir}/ckpts"
    [ -d "$ckpt_dir" ] || return 0
    # ModelCheckpoint saves files like "epoch=17-val_mae_max_metric=0.3796.ckpt"
    ls "$ckpt_dir"/epoch=*.ckpt 2>/dev/null | \
        grep -v "last\.ckpt" | \
        sort -t= -k3 -n | \
        head -1
}

run_experiment() {
    local name="$1"
    shift
    local output_dir="${RESULT_BASE}/${name}"

    if [ "$SKIP_DONE" = true ]; then
        local latest
        latest=$(find_latest_version "$output_dir")
        if [ -f "${latest}/test_stats.json" ] || [ -f "${latest}/test_video_predictions.csv" ]; then
            echo "[SKIP] ${name} — test output exists in ${latest}"
            return
        fi
    fi

    echo ""
    echo "================================================================"
    echo "  EXPERIMENT: ${name}"
    echo "  Output:     ${output_dir}"
    echo "  Backbone:   ${BACKBONE_CKPT}"
    echo "  Time:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    local cmd=(python scripts/run_siamese.py "$@")

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] ${cmd[*]}"
        return
    fi

    mkdir -p "$output_dir"
    "${cmd[@]}" 2>&1 | tee "${output_dir}_train.log" || {
        echo "[FAIL] ${name} — see ${output_dir}_train.log"
        return 1
    }

    echo "[DONE] ${name} — $(date '+%H:%M:%S')"
}

# --- Preflight ---
preflight() {
    echo "=========================================="
    echo "  Siamese 3-class Suite — Preflight"
    echo "=========================================="

    local ok=true

    if [ ! -f "data/biovid/train_3class.txt" ]; then
        echo "  [FAIL] train_3class.txt missing"
        echo "         Run: python scripts/data/build_3class.py"
        exit 1
    fi
    echo "  [OK] BioVid 3class data"

    # Find Stage 1 backbone checkpoint
    local stage1_version
    stage1_version=$(find_latest_version "$BACKBONE_SOURCE")
    if [ ! -d "$stage1_version" ]; then
        echo "  [FAIL] Stage 1 source not found: $BACKBONE_SOURCE"
        echo "         Run the 3-class OrdinalCLIP experiments first:"
        echo "         bash scripts/experiments/run_3class_ordinalclip.sh"
        exit 1
    fi

    BACKBONE_CKPT=$(find_best_ckpt "$stage1_version")
    if [ -z "$BACKBONE_CKPT" ]; then
        echo "  [FAIL] No checkpoint found in ${stage1_version}/ckpts/"
        exit 1
    fi
    echo "  [OK] Stage 1 backbone: ${BACKBONE_CKPT}"

    python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
        && echo "  [OK] GPU" \
        || { echo "  [FAIL] No GPU"; exit 1; }

    echo "=========================================="
}

# --- Main ---

preflight

echo ""
echo "Starting Siamese 3-class suite at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# =============================
# Variant 1: MSE only (regression head only, no ranking loss)
# =============================
should_run_variant() {
    [ -z "$ONLY_VARIANT" ] || [ "$ONLY_VARIANT" = "$1" ]
}

if should_run_variant "mse"; then
    run_experiment "biovid-3cls-siamese-mse-only" \
        --config "$SIAMESE_DEFAULT_CFG" \
        --config "$DATA_CFG" \
        --config "$TRANSFORMS_CFG" \
        --cfg_options \
            "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-siamese-mse-only" \
            "runner_cfg.load_weights_cfg.backbone_ckpt_path=${BACKBONE_CKPT}" \
            "runner_cfg.loss_weights.mse_loss=1.0" \
            "runner_cfg.loss_weights.ranking_loss=0.0"
fi

# =============================
# Variant 2: Ranking only (hinge ranking loss only, no MSE)
# =============================
if should_run_variant "ranking"; then
    run_experiment "biovid-3cls-siamese-ranking-only" \
        --config "$SIAMESE_DEFAULT_CFG" \
        --config "$DATA_CFG" \
        --config "$TRANSFORMS_CFG" \
        --cfg_options \
            "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-siamese-ranking-only" \
            "runner_cfg.load_weights_cfg.backbone_ckpt_path=${BACKBONE_CKPT}" \
            "runner_cfg.loss_weights.mse_loss=0.0" \
            "runner_cfg.loss_weights.ranking_loss=1.0"
fi

# =============================
# Variant 3: Joint linear head (MSE + Hinge, Fabio paper default)
# =============================
if should_run_variant "joint"; then
    run_experiment "biovid-3cls-siamese-joint-linear" \
        --config "$SIAMESE_DEFAULT_CFG" \
        --config "$DATA_CFG" \
        --config "$TRANSFORMS_CFG" \
        --cfg_options \
            "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-siamese-joint-linear" \
            "runner_cfg.load_weights_cfg.backbone_ckpt_path=${BACKBONE_CKPT}" \
            "runner_cfg.loss_weights.mse_loss=1.0" \
            "runner_cfg.loss_weights.ranking_loss=0.5" \
            "runner_cfg.ranking_head_cfg.head_type=linear"
fi

# =============================
# Summary
# =============================

echo ""
echo "================================================================"
echo "  SIAMESE 3-CLASS EXPERIMENTS COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""
echo "Run compute_balanced_metrics.py to get the full metric breakdown:"
echo ""
echo "  python scripts/experiments/compute_balanced_metrics.py \\"
echo "      --pattern 'biovid-3cls-siamese-*' \\"
echo "      --num-classes 3"
