#!/usr/bin/env bash
# ============================================================
# BioVid 5-class Siamese Stage 2 + AU Fusion
# ============================================================
#
# PREREQUISITE
# ------------
# 1. Stage 1 best checkpoint (5-class OrdinalCLIP) must exist.
# 2. AU features must be extracted via:
#    python scripts/data/extract_au_features.py \
#        --openface_dir data/biovid/openface_csv \
#        --data_file data/biovid/train.txt \
#        --output data/biovid/au_features_all17.npz --au_subset all
#    python scripts/data/extract_au_features.py \
#        --openface_dir data/biovid/openface_csv \
#        --data_file data/biovid/train.txt \
#        --output data/biovid/au_features_pain8.npz --au_subset pain
#
# ARCHITECTURE
# ------------
# OrdinalCLIP backbone (frozen) → [CLIP feat ∥ AU feat] → SharedMLP → Heads
#   AU fusion = early concatenation before SharedMLP
#   SharedMLP input: D + au_dim (512+17=529 or 512+8=520)
#
# EXPERIMENTS
# -----------
# Q. 17 AU + frozen backbone (paper defaults)
# R. 8 pain AU + frozen backbone
# S. 8 pain AU + MLP ranking head
# T. 8 pain AU + unfreeze backbone (lr=1e-6)
#
# Usage:
#   bash scripts/experiments/run_biovid_5class_siamese_au.sh
#   bash scripts/experiments/run_biovid_5class_siamese_au.sh --backbone_ckpt path/to/best.ckpt
#   bash scripts/experiments/run_biovid_5class_siamese_au.sh --exp Q
# ============================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"
SIAMESE_CFG="configs/siamese_default.yaml"
BIOVID5_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_siamese_skip2.yaml"
CLIP_NORM="configs/base_cfgs/data_cfg/transforms/clip-normalize.yaml"

# Paths
BACKBONE_CKPT=""
AU_NPZ_ALL17="data/biovid/au_features_all17.npz"
AU_NPZ_PAIN8="data/biovid/au_features_pain8.npz"
MAX_EPOCHS="100"
PAIRS_PER_EPOCH="20000"
EXP_FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbone_ckpt) BACKBONE_CKPT="${2:?}"; shift 2 ;;
        --max_epochs)    MAX_EPOCHS="${2:?}"; shift 2 ;;
        --pairs)         PAIRS_PER_EPOCH="${2:?}"; shift 2 ;;
        --au_all17)      AU_NPZ_ALL17="${2:?}"; shift 2 ;;
        --au_pain8)      AU_NPZ_PAIN8="${2:?}"; shift 2 ;;
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
        "results/biovid-5cls-ordinalclip-frozen-vitb16" \
        "results/biovid-5cls-coop-frozen-vitb16" \
        "results/biovid-5cls-coop-finetune1e5-vitb16" \
        "results/biovid-ordinalclip-vitb16"; do
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

# Verify AU files exist
verify_au() {
    local au_path="$1"
    if [[ ! -f "${au_path}" ]]; then
        echo "ERROR: AU feature file not found: ${au_path}"
        echo "Run extract_au_features.py first. See script header for commands."
        exit 1
    fi
}

run_siamese_au() {
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
# Q. 17 AU + frozen backbone (paper defaults)
#    Full AU set: all 17 OpenFace intensity AUs
#    SharedMLP input: 512 + 17 = 529
# ============================================================
if should_run "Q"; then
    verify_au "${AU_NPZ_ALL17}"
    run_siamese_au "5cls-Q | AU-17 + frozen backbone" \
        "results/biovid-5cls-siamese-Q-au17-frozen" \
        runner_cfg.au_cfg.enabled=true \
        runner_cfg.au_cfg.au_npz_path="${AU_NPZ_ALL17}" \
        runner_cfg.au_cfg.au_dim=17 \
        runner_cfg.au_cfg.au_dropout=0.1 \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.ranking_head_cfg.head_type=linear
fi

# ============================================================
# R. 8 pain AU + frozen backbone
#    Pain-relevant subset: AU04, AU06, AU07, AU09, AU10, AU12, AU25, AU26
#    SharedMLP input: 512 + 8 = 520
# ============================================================
if should_run "R"; then
    verify_au "${AU_NPZ_PAIN8}"
    run_siamese_au "5cls-R | AU-8pain + frozen backbone" \
        "results/biovid-5cls-siamese-R-au8-frozen" \
        runner_cfg.au_cfg.enabled=true \
        runner_cfg.au_cfg.au_npz_path="${AU_NPZ_PAIN8}" \
        runner_cfg.au_cfg.au_dim=8 \
        runner_cfg.au_cfg.au_dropout=0.1 \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.ranking_head_cfg.head_type=linear
fi

# ============================================================
# S. 8 pain AU + MLP ranking head
#    Higher-capacity ConcatRankingHead with AU features
# ============================================================
if should_run "S"; then
    verify_au "${AU_NPZ_PAIN8}"
    run_siamese_au "5cls-S | AU-8pain + MLP head" \
        "results/biovid-5cls-siamese-S-au8-mlp" \
        runner_cfg.au_cfg.enabled=true \
        runner_cfg.au_cfg.au_npz_path="${AU_NPZ_PAIN8}" \
        runner_cfg.au_cfg.au_dim=8 \
        runner_cfg.au_cfg.au_dropout=0.1 \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.ranking_head_cfg.head_type=mlp \
        runner_cfg.ranking_head_cfg.hidden_dims=256
fi

# ============================================================
# T. 8 pain AU + unfreeze backbone (lr=1e-6)
#    Full pipeline fine-tuning with AU features
# ============================================================
if should_run "T"; then
    verify_au "${AU_NPZ_PAIN8}"
    run_siamese_au "5cls-T | AU-8pain + unfreeze backbone lr=1e-6" \
        "results/biovid-5cls-siamese-T-au8-unfreeze" \
        runner_cfg.au_cfg.enabled=true \
        runner_cfg.au_cfg.au_npz_path="${AU_NPZ_PAIN8}" \
        runner_cfg.au_cfg.au_dim=8 \
        runner_cfg.au_cfg.au_dropout=0.1 \
        runner_cfg.loss_weights.mse_loss=1.0 \
        runner_cfg.loss_weights.ranking_loss=0.5 \
        runner_cfg.ranking_head_cfg.head_type=linear \
        runner_cfg.freeze_backbone=false \
        runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_backbone=1.0e-06 \
        runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_siamese_heads=1.0e-04
fi

echo ""
echo "=== 5-class Siamese + AU Results ==="
${PYTHON} scripts/experiments/parse_video_results.py -d results/ -r test -f "5cls-siamese" 2>/dev/null || true
echo "=== Done ==="
