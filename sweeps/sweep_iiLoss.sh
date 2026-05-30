#!/bin/bash
# sweep_iiLoss.sh
# Runs iiLoss λ sweep from 0.0 to 1.0 in 0.1 steps (11 runs × 5 folds).
# Usage:  bash sweep_iiLoss.sh [Ef|Ed]   (default: Ef)
# Monitor: tail -f logs/sweep_iiLoss.log

TARGET=${1:-Ef}
LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Force CPU to avoid NVML driver mismatch crash
# Remove this line once admin fixes the nvidia kernel module
export CUDA_VISIBLE_DEVICES=""

# Find python — works on cluster (conda) and local (python3)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate mlstab 2>/dev/null
PYTHON=$(which python 2>/dev/null || which python3)
echo "Using python: $PYTHON ($($PYTHON --version))"
TOTAL=${#LAMBDAS[@]}
LOGDIR="logs"
LOGFILE="$LOGDIR/sweep_iiLoss_${TARGET}.log"

mkdir -p "$LOGDIR"

echo "=====================================================" | tee -a "$LOGFILE"
echo "  iiLoss sweep — target: $TARGET" | tee -a "$LOGFILE"
echo "  λ values: ${LAMBDAS[*]}" | tee -a "$LOGFILE"
echo "  Started: $(date)" | tee -a "$LOGFILE"
echo "=====================================================" | tee -a "$LOGFILE"

for i in "${!LAMBDAS[@]}"; do
    LAM=${LAMBDAS[$i]}
    RUN=$((i + 1))
    MODEL="iiLoss_${LAM}"

    echo "" | tee -a "$LOGFILE"
    echo "-----------------------------------------------------" | tee -a "$LOGFILE"
    echo "  [$RUN/$TOTAL]  λ = $LAM  →  $MODEL" | tee -a "$LOGFILE"
    echo "  $(date)" | tee -a "$LOGFILE"
    echo "-----------------------------------------------------" | tee -a "$LOGFILE"

    $PYTHON -u mlstabilitytest/train_models.py allMP_current "$TARGET" "$MODEL" \
        2>&1 | tee -a "$LOGFILE"

    STATUS=${PIPESTATUS[0]}
    if [ $STATUS -ne 0 ]; then
        echo "  [ERROR] Run $MODEL exited with status $STATUS — stopping." | tee -a "$LOGFILE"
        exit $STATUS
    fi

    echo "  [DONE] λ = $LAM finished at $(date)" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "=====================================================" | tee -a "$LOGFILE"
echo "  iiLoss sweep complete — $(date)" | tee -a "$LOGFILE"
echo "=====================================================" | tee -a "$LOGFILE"
