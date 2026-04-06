#!/bin/bash
# ==========================================================================
# FER Backbone Experiments — BioVid 5-class Pain Intensity
# ==========================================================================
#
# One-shot script to run all FER encoder experiments sequentially.
#
# Usage:
#   bash scripts/experiments/run_fer_experiments.sh              # run all
#   bash scripts/experiments/run_fer_experiments.sh --dry-run    # print commands only
#   bash scripts/experiments/run_fer_experiments.sh --skip-done  # skip existing results
#
# Prerequisites:
#   1. conda activate ordinalclip
#   2. pip install timm hsemotion
#   3. bash scripts/download_fer_weights.sh all
#   4. Verify: ls .cache/fer/dan_affecnet7.pth
#
# Experiment matrix (8 runs):
#   Encoder         | Frozen | Finetune (lr=1e-5)
#   ----------------+--------+-------------------
#   DAN (ResNet-18) |   ✓    |        ✓
#   HSEmotion-B0    |   ✓    |        ✓
#   HSEmotion-B2    |   ✓    |        ✓
#   Baseline (CLIP) |   ✓    |        ✓           ← control group
# ==========================================================================

set -euo pipefail

# --- Config ---
DEFAULT_CFG="configs/default.yaml"
DATA_CFG="configs/base_cfgs/data_cfg/datasets/biovid/biovid_skip2.yaml"
FREEZE_CFG="configs/base_cfgs/runner_cfg/optim_sched/image_encoder/freeze-image.yaml"

DAN_CFG="configs/base_cfgs/runner_cfg/model/fer_baseline_dan.yaml"
HSE_B0_CFG="configs/base_cfgs/runner_cfg/model/fer_baseline_hsemotion.yaml"
VIT_CFG="configs/base_cfgs/runner_cfg/model/fer_baseline_vit.yaml"

FT_LR="1e-5"
RESULT_BASE="results"

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
    # Find highest version_N, fall back to version_0
    local latest=$(ls -d "${base_dir}"/version_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$latest" ]; then
        echo "$latest"
    else
        echo "${base_dir}/version_0"
    fi
}

# --- Helper: run one experiment ---
# NOTE: scripts/run.py uses --cfg_options with nargs=REMAINDER, so only ONE
# --cfg_options is allowed per command. All overrides must be in a single block.
run_experiment() {
    local name="$1"
    shift
    local output_dir="${RESULT_BASE}/${name}"

    # Skip if results already exist (check latest version)
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

    # Build command — all args are passed through, output_dir is prepended
    # to --cfg_options so it appears in the single REMAINDER block.
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

# --- Preflight checks ---
preflight() {
    echo "=========================================="
    echo "  FER Experiment Suite — Preflight Check"
    echo "=========================================="

    local ok=true

    # Check DAN weights
    if [ ! -f ".cache/fer/dan_affecnet7.pth" ]; then
        echo "  [WARN] DAN weights missing: .cache/fer/dan_affecnet7.pth"
        echo "         Run: bash scripts/download_fer_weights.sh dan"
        ok=false
    else
        echo "  [OK] DAN weights"
    fi

    # Check HSEmotion
    if python -c "import hsemotion" 2>/dev/null; then
        echo "  [OK] hsemotion package"
    else
        echo "  [WARN] hsemotion not installed. Run: pip install hsemotion"
        ok=false
    fi

    # Check timm
    if python -c "import timm" 2>/dev/null; then
        echo "  [OK] timm package"
    else
        echo "  [WARN] timm not installed. Run: pip install timm"
        ok=false
    fi

    # Check data
    if [ ! -f "data/biovid/train_skip2.txt" ]; then
        echo "  [WARN] BioVid skip2 data missing"
        ok=false
    else
        local n_train=$(wc -l < "data/biovid/train_skip2.txt")
        local n_test=$(wc -l < "data/biovid/test_skip2.txt")
        echo "  [OK] BioVid skip2: train=${n_train} test=${n_test}"
    fi

    # Check GPU
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        echo "  [OK] GPU: ${gpu_name}"
    else
        echo "  [WARN] No GPU detected"
        ok=false
    fi

    echo "=========================================="

    if [ "$ok" = false ] && [ "$DRY_RUN" = false ]; then
        echo "  Some checks failed. Continue anyway? [y/N]"
        read -r answer
        if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
            echo "Aborted."
            exit 1
        fi
    fi
}

# --- Main ---

preflight

echo ""
echo "Starting experiment suite at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# =============================
# Group 1: DAN (ResNet-18 + FER)
# =============================

run_experiment "biovid-5cls-dan-frozen" \
    --config "$DEFAULT_CFG" \
    --config "$DAN_CFG" \
    --config "$DATA_CFG" \
    --config "$FREEZE_CFG" \
    --cfg_options "runner_cfg.output_dir=${RESULT_BASE}/biovid-5cls-dan-frozen"

run_experiment "biovid-5cls-dan-ft1e5" \
    --config "$DEFAULT_CFG" \
    --config "$DAN_CFG" \
    --config "$DATA_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-5cls-dan-ft1e5" \
        "runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_image_encoder=${FT_LR}"

# =============================
# Group 2: HSEmotion-B0 (EfficientNet-B0 + FER)
# =============================

run_experiment "biovid-5cls-hsemotion-b0-frozen" \
    --config "$DEFAULT_CFG" \
    --config "$HSE_B0_CFG" \
    --config "$DATA_CFG" \
    --config "$FREEZE_CFG" \
    --cfg_options "runner_cfg.output_dir=${RESULT_BASE}/biovid-5cls-hsemotion-b0-frozen"

run_experiment "biovid-5cls-hsemotion-b0-ft1e5" \
    --config "$DEFAULT_CFG" \
    --config "$HSE_B0_CFG" \
    --config "$DATA_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-5cls-hsemotion-b0-ft1e5" \
        "runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_image_encoder=${FT_LR}"

# =============================
# Group 3: HSEmotion-B2 (EfficientNet-B2 + FER)
# =============================
# B2 uses 260x260 input — override input_size and input_resize

run_experiment "biovid-5cls-hsemotion-b2-frozen" \
    --config "$DEFAULT_CFG" \
    --config "$HSE_B0_CFG" \
    --config "$DATA_CFG" \
    --config "$FREEZE_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-5cls-hsemotion-b2-frozen" \
        "runner_cfg.model_cfg.image_encoder_name=hsemotion_b2" \
        "runner_cfg.model_cfg.encoder_kwargs.hsemotion_model_name=enet_b2_8" \
        "data_cfg.transforms_cfg.input_resize=[292,292]" \
        "data_cfg.transforms_cfg.input_size=[260,260]"

run_experiment "biovid-5cls-hsemotion-b2-ft1e5" \
    --config "$DEFAULT_CFG" \
    --config "$HSE_B0_CFG" \
    --config "$DATA_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-5cls-hsemotion-b2-ft1e5" \
        "runner_cfg.model_cfg.image_encoder_name=hsemotion_b2" \
        "runner_cfg.model_cfg.encoder_kwargs.hsemotion_model_name=enet_b2_8" \
        "runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_image_encoder=${FT_LR}" \
        "data_cfg.transforms_cfg.input_resize=[292,292]" \
        "data_cfg.transforms_cfg.input_size=[260,260]"

# =============================
# Group 4: Control — CLIP Baseline (existing)
# =============================
# Re-run baseline with same data split for fair comparison

BASELINE_CFG="configs/base_cfgs/runner_cfg/model/baseline.yaml"

run_experiment "biovid-5cls-baseline-vitb16-frozen-ctrl" \
    --config "$DEFAULT_CFG" \
    --config "$BASELINE_CFG" \
    --config "$DATA_CFG" \
    --config "$FREEZE_CFG" \
    --cfg_options "runner_cfg.output_dir=${RESULT_BASE}/biovid-5cls-baseline-vitb16-frozen-ctrl"

run_experiment "biovid-5cls-baseline-vitb16-ft1e5-ctrl" \
    --config "$DEFAULT_CFG" \
    --config "$BASELINE_CFG" \
    --config "$DATA_CFG" \
    --cfg_options \
        "runner_cfg.output_dir=${RESULT_BASE}/biovid-5cls-baseline-vitb16-ft1e5-ctrl" \
        "runner_cfg.optimizer_and_scheduler_cfg.param_dict_cfg.lr_image_encoder=${FT_LR}"

# =============================
# Summary
# =============================

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""
echo "Results summary:"
echo ""

printf "%-45s %8s %8s\n" "Experiment" "Acc(max)" "MAE(max)"
printf "%-45s %8s %8s\n" "---------------------------------------------" "--------" "--------"

for dir in \
    biovid-5cls-dan-frozen \
    biovid-5cls-dan-ft1e5 \
    biovid-5cls-hsemotion-b0-frozen \
    biovid-5cls-hsemotion-b0-ft1e5 \
    biovid-5cls-hsemotion-b2-frozen \
    biovid-5cls-hsemotion-b2-ft1e5 \
    biovid-5cls-baseline-vitb16-frozen-ctrl \
    biovid-5cls-baseline-vitb16-ft1e5-ctrl; do

    latest=$(find_latest_version "${RESULT_BASE}/${dir}")
    stats_file="${latest}/test_stats.json"
    if [ -f "$stats_file" ]; then
        acc=$(python -c "import json; d=json.load(open('${stats_file}')); print(f\"{d.get('test_acc_max_metric', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
        mae=$(python -c "import json; d=json.load(open('${stats_file}')); print(f\"{d.get('test_mae_max_metric', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
        printf "%-45s %8s %8s\n" "$dir" "$acc" "$mae"
    else
        printf "%-45s %8s %8s\n" "$dir" "MISSING" "MISSING"
    fi
done

echo ""
echo "Parse full results with:"
echo "  python scripts/experiments/parse_results.py -d results/ -p test_stats.json"
