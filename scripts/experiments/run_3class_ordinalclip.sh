#!/bin/bash
# ==========================================================================
# BioVid 3-class — Original OrdinalCLIP + CLIP Baseline (main branch)
# ==========================================================================
#
# This script runs ONLY the original OrdinalCLIP and CLIP Baseline on the
# 3-class BioVid task. For FER backbone variants, switch to feat/fer-backbone
# and use scripts/experiments/run_fer_experiments_3class.sh instead.
#
# Label scheme:
#   class 0 = {BLN, PA1, PA2}  (no pain + low pain)
#   class 1 = {PA3}            (moderate pain)
#   class 2 = {PA4}            (severe pain)
#
# Usage:
#   bash scripts/experiments/run_3class_ordinalclip.sh              # run all
#   bash scripts/experiments/run_3class_ordinalclip.sh --dry-run    # preview
#   bash scripts/experiments/run_3class_ordinalclip.sh --skip-done  # resume
#
# Prerequisites:
#   1. conda activate ordinalclip
#   2. python scripts/data/build_3class.py    # generate train_3class.txt
# ==========================================================================

set -euo pipefail

# --- Config ---
DEFAULT_CFG="configs/default.yaml"
DATA_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_3class.yaml"
FREEZE_CFG="configs/base_cfgs/runner_cfg/optim_sched/image_encoder/freeze-image.yaml"
BASELINE_CFG="configs/base_cfgs/runner_cfg/model/baseline.yaml"
ORDINALCLIP_CFG="configs/base_cfgs/runner_cfg/model/ordinalclip.yaml"

FT_LR="1e-5"
RESULT_BASE="results"

# Class weights: inverse-frequency, computed from train file at runtime
COMPUTE_WEIGHTS_PY=$(cat <<'PYEOF'
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
    CLASS_WEIGHTS="[0.5,1.5,1.5]"
fi

DRY_RUN=false
SKIP_DONE=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --skip-done) SKIP_DONE=true ;;
    esac
done

# --- Helpers ---
find_latest_version() {
    local base_dir="$1"
    local latest=$(ls -d "${base_dir}"/version_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$latest" ]; then
        echo "$latest"
    else
        echo "${base_dir}/version_0"
    fi
}

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
    echo "  3-Class OrdinalCLIP Suite — Preflight"
    echo "=========================================="

    if [ ! -f "data/biovid/train_3class.txt" ]; then
        echo "  [FAIL] train_3class.txt missing"
        echo "         Run: python scripts/data/build_3class.py"
        exit 1
    fi

    local n_train=$(wc -l < "data/biovid/train_3class.txt")
    local n_test=$(wc -l < "data/biovid/test_3class.txt")
    echo "  [OK] BioVid 3class: train=${n_train} test=${n_test}"
    echo "  [OK] class_weights: ${CLASS_WEIGHTS}"

    python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && echo "  [OK] GPU" || echo "  [WARN] No GPU"

    echo "=========================================="
}

# --- Main ---

preflight

echo ""
echo "Starting 3-class OrdinalCLIP suite at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# =============================
# Original OrdinalCLIP
# =============================

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
# CLIP Baseline (control)
# =============================

run_experiment "biovid-3cls-baseline-rn50-frozen" \
    --config "$DEFAULT_CFG" \
    --config "$BASELINE_CFG" \
    --config "$DATA_CFG" \
    --config "$FREEZE_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-baseline-rn50-frozen" \
        "runner_cfg.class_weights=${CLASS_WEIGHTS}" \
        "data_cfg.balanced_sampling=true"

run_experiment "biovid-3cls-baseline-rn50-ft1e5" \
    --config "$DEFAULT_CFG" \
    --config "$BASELINE_CFG" \
    --config "$DATA_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-3cls-baseline-rn50-ft1e5" \
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
    biovid-3cls-ordinalclip-frozen \
    biovid-3cls-ordinalclip-ft1e5 \
    biovid-3cls-baseline-rn50-frozen \
    biovid-3cls-baseline-rn50-ft1e5; do

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
