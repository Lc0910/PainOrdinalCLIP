#!/usr/bin/env bash
# ============================================================
# Video-level Aggregation Strategy Ablation
# ============================================================
# Test mean / max / topk_mean on ALL existing checkpoints.
# No training — test_only mode, reuses saved checkpoints.
#
# Usage:
#   bash scripts/experiments/run_video_agg_ablation.sh
#   bash scripts/experiments/run_video_agg_ablation.sh --strategies "max topk5"
#   bash scripts/experiments/run_video_agg_ablation.sh --experiments "biovid-5cls-ordinalclip-frozen-vitb16-skip2"
#   bash scripts/experiments/run_video_agg_ablation.sh --dry_run
# ============================================================

set -euo pipefail

# -----------------------------------------------------------
# Configurable parameters
# -----------------------------------------------------------
STRATEGIES="mean max topk5 topk10"   # which strategies to test
EXPERIMENTS=""                        # empty = auto-discover all
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategies)  STRATEGIES="${2:?--strategies requires values}"; shift 2 ;;
        --experiments) EXPERIMENTS="${2:?--experiments requires values}"; shift 2 ;;
        --dry_run)     DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# -----------------------------------------------------------
# Auto-discover experiments: find all version dirs with config.yaml + ckpts/
# -----------------------------------------------------------
if [[ -z "${EXPERIMENTS}" ]]; then
    echo "[INFO] Auto-discovering experiments in results/ ..."
    EXPR_DIRS=()
    while IFS= read -r cfg; do
        expr_dir="$(dirname "$cfg")"
        # Must have a ckpts/ directory with at least one .ckpt file
        if ls "${expr_dir}"/ckpts/*.ckpt 1>/dev/null 2>&1; then
            EXPR_DIRS+=("${expr_dir}")
        fi
    done < <(find results/ -name "config.yaml" -path "*/version_*" 2>/dev/null | sort)
else
    # User specified experiment names (without version), find the latest version
    EXPR_DIRS=()
    for name in ${EXPERIMENTS}; do
        latest=$(find "results/${name}" -name "config.yaml" -path "*/version_*" 2>/dev/null | sort | tail -1)
        if [[ -n "${latest}" ]]; then
            EXPR_DIRS+=("$(dirname "${latest}")")
        else
            echo "[WARN] No config.yaml found for: ${name}, skipping"
        fi
    done
fi

echo "[INFO] Found ${#EXPR_DIRS[@]} experiment(s) to test"
echo "[INFO] Strategies: ${STRATEGIES}"
echo ""

# -----------------------------------------------------------
# Map strategy name → cfg_options
# -----------------------------------------------------------
strategy_to_cfg() {
    local s="$1"
    case "$s" in
        mean)    echo "runner_cfg.video_agg_strategy=mean" ;;
        max)     echo "runner_cfg.video_agg_strategy=max" ;;
        topk5)   echo "runner_cfg.video_agg_strategy=topk_mean runner_cfg.video_agg_topk=5" ;;
        topk10)  echo "runner_cfg.video_agg_strategy=topk_mean runner_cfg.video_agg_topk=10" ;;
        topk3)   echo "runner_cfg.video_agg_strategy=topk_mean runner_cfg.video_agg_topk=3" ;;
        topk20)  echo "runner_cfg.video_agg_strategy=topk_mean runner_cfg.video_agg_topk=20" ;;
        *) echo "Unknown strategy: $s" >&2; exit 1 ;;
    esac
}

# -----------------------------------------------------------
# Run ablation
# -----------------------------------------------------------
TOTAL=0
SUCCESS=0
FAIL=0

for expr_dir in "${EXPR_DIRS[@]}"; do
    expr_name="$(basename "$(dirname "${expr_dir}")")/$(basename "${expr_dir}")"

    for strategy in ${STRATEGIES}; do
        out_dir="${expr_dir}/agg_${strategy}"

        # Skip if already done
        if [[ -f "${out_dir}/test_video_stats.json" ]]; then
            echo "[SKIP] ${expr_name} | ${strategy} (already exists)"
            continue
        fi

        cfg_opts=$(strategy_to_cfg "${strategy}")
        TOTAL=$((TOTAL + 1))

        # Auto-detect entry script: Siamese experiments use run_siamese.py
        if [[ "${expr_dir}" == *siamese* ]]; then
            RUN_SCRIPT="scripts/run_siamese.py"
        else
            RUN_SCRIPT="scripts/run.py"
        fi

        if ${DRY_RUN}; then
            echo "[DRY] python ${RUN_SCRIPT} --config ${expr_dir}/config.yaml --test_only --output_dir ${out_dir} --cfg_options ${cfg_opts}"
            continue
        fi

        echo ""
        echo "========================================================"
        echo "[${TOTAL}] ${expr_name} | strategy=${strategy} | ${RUN_SCRIPT}"
        echo "========================================================"

        if python "${RUN_SCRIPT}" \
            --config "${expr_dir}/config.yaml" \
            --test_only \
            --output_dir "${out_dir}" \
            --cfg_options ${cfg_opts}; then
            SUCCESS=$((SUCCESS + 1))
            echo "[OK] ${expr_name} | ${strategy}"
        else
            FAIL=$((FAIL + 1))
            echo "[FAIL] ${expr_name} | ${strategy}"
        fi
    done
done

# -----------------------------------------------------------
# Summary table
# -----------------------------------------------------------
echo ""
echo "========================================================"
echo "  AGGREGATION ABLATION RESULTS"
echo "========================================================"
echo ""

# Header
printf "%-55s" "Experiment"
for strategy in ${STRATEGIES}; do
    printf "| %-18s" "${strategy}"
done
echo ""
printf "%-55s" "$(printf '%0.s-' {1..55})"
for strategy in ${STRATEGIES}; do
    printf "| %-18s" "$(printf '%0.s-' {1..18})"
done
echo ""

# Rows
for expr_dir in "${EXPR_DIRS[@]}"; do
    expr_name="$(basename "$(dirname "${expr_dir}")")"
    printf "%-55s" "${expr_name}"

    for strategy in ${STRATEGIES}; do
        stats_file="${expr_dir}/agg_${strategy}/test_video_stats.json"
        if [[ -f "${stats_file}" ]]; then
            # Read the last line (latest test run)
            last_line=$(tail -1 "${stats_file}")
            acc=$(echo "${last_line}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['acc_max_metric']*100:.2f}%\")" 2>/dev/null || echo "err")
            mae=$(echo "${last_line}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['mae_max_metric']:.4f}\")" 2>/dev/null || echo "err")
            printf "| %6s / %7s  " "${acc}" "${mae}"
        else
            printf "| %-18s" "  --"
        fi
    done
    echo ""
done

echo ""
echo "Format: acc_max / mae_max"
echo ""

# Detailed JSON dump
echo "========================================================"
echo "  DETAILED JSON"
echo "========================================================"
for expr_dir in "${EXPR_DIRS[@]}"; do
    expr_name="$(basename "$(dirname "${expr_dir}")")"
    for strategy in ${STRATEGIES}; do
        stats_file="${expr_dir}/agg_${strategy}/test_video_stats.json"
        if [[ -f "${stats_file}" ]]; then
            echo ""
            echo "--- ${expr_name} | ${strategy} ---"
            tail -1 "${stats_file}" | python3 -m json.tool 2>/dev/null || tail -1 "${stats_file}"
        fi
    done
done

echo ""
echo "========================================================"
echo "  DONE  (total=${TOTAL}, success=${SUCCESS}, fail=${FAIL})"
echo "========================================================"
