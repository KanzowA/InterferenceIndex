#!/bin/bash
# sweep_EdLoss.sh
# Runs EdLoss α sweep from 0.0 to 1.0 in 0.1 steps (11 runs × 5 folds).
# Uses the normalised convex-combination formulation:
#   L = (1-α)·MSE/MSE_ref + α·Ed²/Ed²_ref
# Usage:  bash sweep_EdLoss.sh [Ef|Ed]   (default: Ef)
# Monitor: tail -f logs/sweep_EdLoss.log

TARGET=${1:-Ef}
ALPHAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Force CPU to avoid NVML driver mismatch crash
# Remove this line once admin fixes the nvidia kernel module
export CUDA_VISIBLE_DEVICES=""

# Find python — works on cluster (conda) and local (python3)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate mlstab 2>/dev/null
PYTHON=$(which python 2>/dev/null || which python3)
echo "Using python: $PYTHON ($($PYTHON --version))"
TOTAL=${#ALPHAS[@]}
LOGDIR="logs"
LOGFILE="$LOGDIR/sweep_EdLoss_${TARGET}.log"

mkdir -p "$LOGDIR"

echo "=====================================================" | tee -a "$LOGFILE"
echo "  EdLoss sweep — target: $TARGET" | tee -a "$LOGFILE"
echo "  α values: ${ALPHAS[*]}" | tee -a "$LOGFILE"
echo "  Started: $(date)" | tee -a "$LOGFILE"
echo "=====================================================" | tee -a "$LOGFILE"

for i in "${!ALPHAS[@]}"; do
    ALP=${ALPHAS[$i]}
    RUN=$((i + 1))
    MODEL="EdLoss_${ALP}"

    echo "" | tee -a "$LOGFILE"
    echo "-----------------------------------------------------" | tee -a "$LOGFILE"
    echo "  [$RUN/$TOTAL]  α = $ALP  →  $MODEL" | tee -a "$LOGFILE"
    echo "  $(date)" | tee -a "$LOGFILE"
    echo "-----------------------------------------------------" | tee -a "$LOGFILE"

    $PYTHON -u mlstabilitytest/train_models.py allMP_current_single "$TARGET" "$MODEL" \
        2>&1 | tee -a "$LOGFILE"

    STATUS=${PIPESTATUS[0]}
    if [ $STATUS -ne 0 ]; then
        echo "  [ERROR] Run $MODEL exited with status $STATUS — stopping." | tee -a "$LOGFILE"
        exit $STATUS
    fi

    echo "  [DONE] α = $ALP finished at $(date)" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "=====================================================" | tee -a "$LOGFILE"
echo "  EdLoss sweep complete — $(date)" | tee -a "$LOGFILE"
echo "=====================================================" | tee -a "$LOGFILE"
