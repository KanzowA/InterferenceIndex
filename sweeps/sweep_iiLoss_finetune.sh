#!/bin/bash
# sweep_iiLoss_finetune.sh
# ─────────────────────────────────────────────────────────────────────────────
# Two-stage iiLoss sweep with proper reference normalisation.
#
# Stage 1: Train λ=0 model (pure MSE) on all 5 folds, save checkpoints to
#          checkpoints/<target>/Ef_fold{0..4}.pt
#
# Stage 2: For each λ in LAMBDAS, load the stage-1 checkpoints, set reference
#          values from the converged model, then fine-tune with iiLoss.
#
# Usage:
#   bash sweep_iiLoss_finetune.sh          # target = Ef (default)
#   bash sweep_iiLoss_finetune.sh Ed
# ─────────────────────────────────────────────────────────────────────────────

TARGET=${1:-Ef}
LAMBDAS=(0.05 0.1 0.2 0.3 0.4 0.5 0.7 1.0)

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate mlstab 2>/dev/null
PYTHON=$(which python 2>/dev/null || which python3)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Two-stage iiLoss finetune sweep — target: $TARGET"
echo "  $(date)"
echo "============================================================"

# ── Stage 1: save checkpoints ─────────────────────────────────────────────────
CKPT_DIR="checkpoints/$TARGET"
STAGE1_DONE=true
for fold in 0 1 2 3 4; do
    if [ ! -f "$CKPT_DIR/${TARGET}_fold${fold}.pt" ]; then
        STAGE1_DONE=false
        break
    fi
done

if $STAGE1_DONE; then
    echo ""
    echo "Stage 1 checkpoints already exist — skipping."
else
    echo ""
    echo "Stage 1: Training λ=0 (pure MSE) and saving checkpoints..."
    MODEL="iiLoss_save_0.0"
    LOG="$LOG_DIR/finetune_stage1_${TARGET}.log"
    echo "  Log: $LOG"
    $PYTHON -u mlstabilitytest/train_models.py allMP_current "$TARGET" "$MODEL" \
        2>&1 | tee "$LOG"
    echo "  Stage 1 done."
fi

# ── Stage 2: fine-tune for each λ ────────────────────────────────────────────
N=${#LAMBDAS[@]}
for i in "${!LAMBDAS[@]}"; do
    LAM="${LAMBDAS[$i]}"
    MODEL="iiLoss_finetune_${LAM}"
    LOG="$LOG_DIR/finetune_stage2_${TARGET}_lam${LAM}.log"

    echo ""
    echo "[$((i+1))/$N]  Fine-tuning λ=${LAM}  →  ${MODEL}"
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
