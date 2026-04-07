#!/bin/bash
# ==========================================================================
# FER Backbone Experiments — BioVid 3-class Pain Intensity
# ==========================================================================
#
# Label scheme:
#   class 0 = {BLN, PA1, PA2}  (no pain + low pain)
#   class 1 = {PA3}            (moderate pain)
#   class 2 = {PA4}            (severe pain)
#
# Data is imbalanced; class weights are computed from training set.
#
# Usage:
#   bash scripts/experiments/run_fer_experiments_3class.sh              # run all
#   bash scripts/experiments/run_fer_experiments_3class.sh --dry-run    # preview
#   bash scripts/experiments/run_fer_experiments_3class.sh --skip-done  # resume
#
# Prerequisites:
#   1. conda activate ordinalclip
#   2. pip install timm hsemotion
#   3. bash scripts/download_fer_weights.sh all
#   4. python scripts/data/build_3class.py    # generate train_3class.txt
# ==========================================================================

set -euo pipefail

# --- Config ---
DEFAULT_CFG="configs/default.yaml"
DATA_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_3class.yaml"
FREEZE_CFG="configs/base_cfgs/runner_cfg/optim_sched/image_encoder/freeze-image.yaml"

DAN_CFG="configs/base_cfgs/runner_cfg/model/fer_baseline_dan.yaml"
HSE_B0_CFG="configs/base_cfgs/runner_cfg/model/fer_baseline_hsemotion.yaml"
BASELINE_CFG="configs/base_cfgs/runner_cfg/model/baseline.yaml"
ORDINALCLIP_CFG="configs/base_cfgs/runner_cfg/model/ordinalclip.yaml"

FT_LR="1e-5"
RESULT_BASE="results"

# Class weights: inverse-frequency normalized so sum == num_classes (3).
# Will be auto-computed at runtime from train_3class.txt if available;
# fallback to roughly 0.5/1.5/1.0 for {0,1,2} given typical imbalance.
COMPUTE_WEIGHTS_PY=$(cat <<'PYEOF'
import sys
from collections import Counter
counts = Counter()
with open("data/biovid/train_3class.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            counts[int(parts[1])] += 1
total = sum(counts.values())
nc = 3
weights = [total / (nc * counts.get(i, 1)) for i in range(nc)]
print(f"[{weights[0]:.4f},{weights[1]:.4f},{weights[2]:.4f}]")
PYEOF
)

if [ -f "data/biovid/train_3class.txt" ]; then
    CLASS_WEIGHTS=$(python -c "$COMPUTE_WEIGHTS_PY")
else
    CLASS_WEIGHTS="[0.5,1.5,1.0]"
fi

DRY_RUN=false
SKIP_DONE=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --skip-done) SKIP_DONE=true ;;
    esac
done

# --- Helper: find latest version_N directory ---
find_latest_version() {
    local base_dir="$1"
    local latest=$(ls -d "${base_dir}"/version_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$latest" ]; then
        echo "$latest"
    else
        echo "${base_dir}/version_0"
    fi
}

# --- Helper: run one experiment ---
run_experiment() {
    local name="$1"
    shift
    local output_dir="${RESULT_BASE}/${name}"

    if [ "$SKIP_DONE" = true ]; then
        local latest=$(find_latest_version "$output_dir")
        if [ -f "${latest}/test_stats.json" ]; then
            echo "[SKIP] ${name} — test_stats.json exists in ${latest}"
            return
        fi
    fi

    echo ""
    echo "================================================================"
    echo "  EXPERIMENT: ${name}"
    echo "  Output:     ${output_dir}"
    echo "  Time:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    local cmd=(python scripts/run.py "$@")

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
    echo "  FER 3-Class Experiment Suite — Preflight"
    echo "=========================================="

    local ok=true

    if [ ! -f "data/biovid/train_3class.txt" ]; then
        echo "  [WARN] train_3class.txt missing"
        echo "         Run: python scripts/data/build_3class.py"
        ok=false
    else
        local n_train=$(wc -l < "data/biovid/train_3class.txt")
        local n_test=$(wc -l < "data/biovid/test_3class.txt")
        echo "  [OK] BioVid 3class: train=${n_train} test=${n_test}"
        echo "  [OK] class_weights: ${CLASS_WEIGHTS}"
    fi

    if [ ! -f ".cache/fer/dan_affecnet7.pth" ]; then
        echo "  [WARN] DAN weights missing"
        ok=false
    else
        echo "  [OK] DAN weights"
    fi

    python -c "import hsemotion" 2>/dev/null && echo "  [OK] hsemotion package" || { echo "  [WARN] hsemotion missing"; ok=false; }
    python -c "import timm" 2>/dev/null && echo "  [OK] timm package" || { echo "  [WARN] timm missing"; ok=false; }
    python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && echo "  [OK] GPU" || { echo "  [WARN] No GPU"; ok=false; }

    echo "=========================================="

    if [ "$ok" = false ] && [ "$DRY_RUN" = false ]; then
        echo "  Some checks failed. Continue anyway? [y/N]"
        read -r answer
        if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
            exit 1
        fi
    fi
}

# --- Main ---

preflight

echo ""
echo "Starting 3-class experiment suite at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Class weights: ${CLASS_WEIGHTS}"
echo ""

# =============================
# DAN
# =============================

run_experiment "biovid-3cls-dan-frozen" \
    --config "$DEFAULT_CFG" \
    --config "$DAN_CFG" \
    --config "$DATA_CFG" \
    --config "$FREEZE_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-dan-frozen" \
        "runner_cfg.class_weights=${CLASS_WEIGHTS}" \
        "data_cfg.balanced_sampling=true"

run_experiment "biovid-3cls-dan-ft1e5" \
    --config "$DEFAULT_CFG" \
    --config "$DAN_CFG" \
    --config "$DATA_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-dan-ft1e5" \
        "runner_cfg.class_weights=${CLASS_WEIGHTS}" \
        "data_cfg.balanced_sampling=true" \
        "runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_image_encoder=${FT_LR}"

# =============================
# HSEmotion-B0
# =============================

run_experiment "biovid-3cls-hsemotion-b0-frozen" \
    --config "$DEFAULT_CFG" \
    --config "$HSE_B0_CFG" \
    --config "$DATA_CFG" \
    --config "$FREEZE_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-hsemotion-b0-frozen" \
        "runner_cfg.class_weights=${CLASS_WEIGHTS}" \
        "data_cfg.balanced_sampling=true"

run_experiment "biovid-3cls-hsemotion-b0-ft1e5" \
    --config "$DEFAULT_CFG" \
    --config "$HSE_B0_CFG" \
    --config "$DATA_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-hsemotion-b0-ft1e5" \
        "runner_cfg.class_weights=${CLASS_WEIGHTS}" \
        "data_cfg.balanced_sampling=true" \
        "runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_image_encoder=${FT_LR}"

# =============================
# CLIP Baseline (control)
# =============================

run_experiment "biovid-3cls-baseline-vitb16-frozen-ctrl" \
    --config "$DEFAULT_CFG" \
    --config "$BASELINE_CFG" \
    --config "$DATA_CFG" \
    --config "$FREEZE_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-baseline-vitb16-frozen-ctrl" \
        "runner_cfg.class_weights=${CLASS_WEIGHTS}" \
        "data_cfg.balanced_sampling=true"

run_experiment "biovid-3cls-baseline-vitb16-ft1e5-ctrl" \
    --config "$DEFAULT_CFG" \
    --config "$BASELINE_CFG" \
    --config "$DATA_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-baseline-vitb16-ft1e5-ctrl" \
        "runner_cfg.class_weights=${CLASS_WEIGHTS}" \
        "data_cfg.balanced_sampling=true" \
        "runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_image_encoder=${FT_LR}"

# =============================
# Original OrdinalCLIP (CLIP image+text encoder + ordinal prompts)
# =============================
# Uses default.yaml's RN50 backbone + PlainPromptLearner with num_ranks=3
# from biovid_3class.yaml (overrides default num_ranks=100).

run_experiment "biovid-3cls-ordinalclip-frozen" \
    --config "$DEFAULT_CFG" \
    --config "$ORDINALCLIP_CFG" \
    --config "$DATA_CFG" \
    --config "$FREEZE_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-ordinalclip-frozen" \
        "runner_cfg.class_weights=${CLASS_WEIGHTS}" \
        "data_cfg.balanced_sampling=true"

run_experiment "biovid-3cls-ordinalclip-ft1e5" \
    --config "$DEFAULT_CFG" \
    --config "$ORDINALCLIP_CFG" \
    --config "$DATA_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-ordinalclip-ft1e5" \
        "runner_cfg.class_weights=${CLASS_WEIGHTS}" \
        "data_cfg.balanced_sampling=true" \
        "runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_image_encoder=${FT_LR}"

# =============================
# Summary
# =============================

echo ""
echo "================================================================"
echo "  ALL 3-CLASS EXPERIMENTS COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

printf "%-50s %8s %8s\n" "Experiment" "Acc(max)" "MAE(max)"
printf "%-50s %8s %8s\n" "--------------------------------------------------" "--------" "--------"

for dir in \
    biovid-3cls-dan-frozen \
    biovid-3cls-dan-ft1e5 \
    biovid-3cls-hsemotion-b0-frozen \
    biovid-3cls-hsemotion-b0-ft1e5 \
    biovid-3cls-baseline-vitb16-frozen-ctrl \
    biovid-3cls-baseline-vitb16-ft1e5-ctrl \
    biovid-3cls-ordinalclip-frozen \
    biovid-3cls-ordinalclip-ft1e5; do

    latest=$(find_latest_version "${RESULT_BASE}/${dir}")
    stats_file="${latest}/test_stats.json"
    if [ -f "$stats_file" ]; then
        acc=$(python -c "import json; d=json.load(open('${stats_file}')); print(f\"{d.get('test_acc_max_metric', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
        mae=$(python -c "import json; d=json.load(open('${stats_file}')); print(f\"{d.get('test_mae_max_metric', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
        printf "%-50s %8s %8s\n" "$dir" "$acc" "$mae"
    else
        printf "%-50s %8s %8s\n" "$dir" "MISSING" "MISSING"
    fi
done
