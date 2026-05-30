#!/bin/bash
# Sweep EdLoss_pcgrad across alpha values in separate tmux windows.
# Stage-1 checkpoints must already exist in checkpoints/Ef/.
#
# Usage:
#   chmod +x sweep_EdLoss_pcgrad.sh
#   ./sweep_EdLoss_pcgrad.sh

ALPHAS=(0.2 0.3 0.5)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for alpha in "${ALPHAS[@]}"; do
    log="edloss_pcgrad_${alpha}.log"
    echo "Starting EdLoss_pcgrad_${alpha} → ${log}"
    python -u mlstabilitytest/train_models.py allMP_current Ef EdLoss_pcgrad_${alpha} \
        2>&1 | tee "${log}"
    echo "Done with EdLoss_pcgrad_${alpha}"
done

echo "All runs complete."
