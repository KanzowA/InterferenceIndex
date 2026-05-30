#!/bin/bash
# sweep_iiLoss_pcgrad.sh
# ─────────────────────────────────────────────────────────────────────────────
# Fine-tune with gradient surgery (PCGrad): ξ² gradient projected orthogonal
# to MSE gradient at every step, so ξ reduction cannot hurt Ef accuracy.
#
# Reuses stage-1 checkpoints from sweep_iiLoss_finetune.sh — run that first.
#
# Usage:
#   bash sweep_iiLoss_pcgrad.sh          # target = Ef (default)
#   bash sweep_iiLoss_pcgrad.sh Ed
# ─────────────────────────────────────────────────────────────────────────────

TARGET=${1:-Ef}
LAMBDAS=(0.05 0.1 0.2 0.3 0.5)

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate mlstab 2>/dev/null
PYTHON=$(which python 2>/dev/null || which python3)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  PCGrad iiLoss sweep — target: $TARGET"
echo "  $(date)"
echo "============================================================"

# ── Check stage-1 checkpoints exist ──────────────────────────────────────────
CKPT_DIR="checkpoints/$TARGET"
for fold in 0 1 2 3 4; do
    if [ ! -f "$CKPT_DIR/${TARGET}_fold${fold}.pt" ]; then
        echo "ERROR: checkpoint $CKPT_DIR/${TARGET}_fold${fold}.pt not found."
        echo "Run sweep_iiLoss_finetune.sh first to generate stage-1 checkpoints."
        exit 1
    fi
done
echo "Stage-1 checkpoints found — running PCGrad fine-tuning."

# ── Fine-tune with PCGrad for each λ ─────────────────────────────────────────
N=${#LAMBDAS[@]}
for i in "${!LAMBDAS[@]}"; do
    LAM="${LAMBDAS[$i]}"
    MODEL="iiLoss_pcgrad_${LAM}"
    LOG="$LOG_DIR/pcgrad_${TARGET}_lam${LAM}.log"

    echo ""
    echo "[$((i+1))/$N]  PCGrad λ=${LAM}  →  ${MODEL}"
    echo "  $(date)"
    echo "  Log: $LOG"
    echo "-----------------------------------------------------"

    $PYTHON -u mlstabilitytest/train_models.py allMP_current_finetune \
        "$TARGET" "$MODEL" 2>&1 | tee "$LOG"
done

echo ""
echo "============================================================"
echo "  All done — $(date)"
echo "  Run: python interference_score.py allMP_current_finetune \\"
echo "         --hullout hullout_current.json"
echo "============================================================"
