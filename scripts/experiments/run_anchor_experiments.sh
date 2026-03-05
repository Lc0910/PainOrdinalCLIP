#!/bin/bash
# Run anchor-based inference experiments I/J/K
# Usage: bash scripts/experiments/run_anchor_experiments.sh

set -euo pipefail

BACKBONE_CKPT="results/biovid-2cls-coop-finetune1e5-vitb16/version_0/ckpts/epoch=00-val_mae_max_metric=0.3733.ckpt"

# Verify F checkpoint exists
F_CKPT=$(find results/biovid-2cls-siamese-F-joint-mlp -name "epoch=*.ckpt" -not -name "last.ckpt" 2>/dev/null | sort -t'=' -k3 -n | head -1 || true)
if [[ -z "${F_CKPT}" ]]; then
    echo "ERROR: No F checkpoint found. Run experiment F first."
    exit 1
fi
echo "Backbone: ${BACKBONE_CKPT}"
echo "F checkpoint: ${F_CKPT}"
echo ""

for exp in I J K; do
    echo "========== Running experiment ${exp} =========="
    bash scripts/experiments/run_biovid_2class_siamese.sh \
        --backbone_ckpt "${BACKBONE_CKPT}" \
        --exp ${exp}
    echo ""
done

echo "=== All done ==="
