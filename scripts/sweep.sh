#!/bin/bash
# Full training sweep over lam and model variant, for one or more seeds.
#
# Per seed:
#   [1] iiLoss_save_0.0      stage 1, pure MSE, writes the per-fold checkpoints
#                            that every stage-2 run for this seed reloads.
#                            Output is copied to iiLoss_0.0 and HdLoss_0.0.
#   [2] iiLoss_finetune_0.0  matched lam=0 control (500+200 epochs). This is the
#                            anchor the lam>0 fine-tuned runs must be compared
#                            against; the stage-1 baseline is not, because it
#                            has seen 200 fewer epochs.
#   [3..] lam = 0.1 .. 1.0 crossed with VARIANTS.
#
# Usage:
#   bash scripts/sweep.sh                # target Hf, seed 0
#   bash scripts/sweep.sh Hf 0 1 2       # target Hf, three seeds
#
# Requires the 'interference' conda environment to be active.

set -u

TARGET=${1:-Hf}
shift || true
SEEDS=("$@")
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(0)

SPLIT="allMP_2026"
VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Two-stage variants only, so surgery-vs-no-surgery is a controlled comparison:
# *_pcgrad and *_finetune differ solely in the gradient projection.
# Add iiLoss / HdLoss here to include the single-stage runs as well.
VARIANTS=(iiLoss_pcgrad iiLoss_finetune HdLoss_pcgrad HdLoss_finetune)

PYTHON=$(command -v python)
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: no python on PATH. Activate the environment first:"
    echo "    conda activate interference"
    exit 1
fi
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    echo "ERROR: torch not importable. Activate the environment first:"
    echo "    conda activate interference"
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PER_SEED=$(( 2 + ${#VALUES[@]} * ${#VARIANTS[@]} ))
TOTAL=$(( PER_SEED * ${#SEEDS[@]} ))

mkdir -p logs
LOGFILE="logs/sweep_${TARGET}_$(date +%Y%m%d_%H%M%S).log"
ML_DIR="data/2026/ml/${TARGET}"

log() { echo "$@" | tee -a "$LOGFILE"; }

log "============================================================"
log "  Sweep: target ${TARGET}, seeds ${SEEDS[*]}"
log "  ${PER_SEED} runs per seed, ${TOTAL} total"
log "  Python: $PYTHON ($($PYTHON --version 2>&1))"
log "  Started: $(date)"
log "============================================================"

IDX=1

run() {   # run <model> <seed>
    local model=$1 seed=$2
    log ""
    log "[${IDX}/${TOTAL}]  ${model}  seed ${seed}  ($(date))"
    log "-----------------------------------------------------"
    "$PYTHON" -u scripts/train_models.py "$SPLIT" "$TARGET" "$model" "$seed" 2>&1 | tee -a "$LOGFILE"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log "ERROR: ${model} seed ${seed} failed - stopping."
        exit 1
    fi
    log "[DONE] ${model} seed ${seed} at $(date)"
    IDX=$(( IDX + 1 ))
}

for SEED in "${SEEDS[@]}"; do

    # Stage 1. Checkpoints are written per seed, so seeds do not clobber
    # each other and may be run in any order.
    run iiLoss_save_0.0 "$SEED"
    for DIR in iiLoss_0.0 HdLoss_0.0; do
        mkdir -p "${ML_DIR}/${DIR}_s${SEED}"
        cp "${ML_DIR}/iiLoss_save_0.0_s${SEED}/ml_input.json" \
           "${ML_DIR}/${DIR}_s${SEED}/ml_input.json"
        log "  copied stage-1 predictions to ${DIR}_s${SEED}"
    done
    rm -rf "${ML_DIR}/iiLoss_save_0.0_s${SEED}"

    # Matched lam=0 control.
    run iiLoss_finetune_0.0 "$SEED"
    mkdir -p "${ML_DIR}/HdLoss_finetune_0.0_s${SEED}"
    cp "${ML_DIR}/iiLoss_finetune_0.0_s${SEED}/ml_input.json" \
       "${ML_DIR}/HdLoss_finetune_0.0_s${SEED}/ml_input.json"
    log "  copied control predictions to HdLoss_finetune_0.0_s${SEED}"

    for VAL in "${VALUES[@]}"; do
        for VARIANT in "${VARIANTS[@]}"; do
            run "${VARIANT}_${VAL}" "$SEED"
        done
    done
done

log ""
log "============================================================"
log "  Sweep complete - $(date)"
log "  Log: $LOGFILE"
log "  Evaluate with:"
log "    python scripts/interference_score.py 2026"
log "============================================================"
