#!/bin/bash
# sweep.sh — Run iiLoss or HdLoss hyperparameter sweep (single-stage or PCGrad).
#
# Usage:
#   bash sweep.sh iiLoss             # iiLoss λ sweep,        single-stage
#   bash sweep.sh HdLoss             # HdLoss α sweep,        single-stage
#   bash sweep.sh iiLoss pcgrad      # iiLoss+PCGrad,         two-stage
#   bash sweep.sh HdLoss pcgrad      # HdLoss+PCGrad,         two-stage
#
# Optional third argument overrides the training target (default: Hf):
#   bash sweep.sh iiLoss single Hd
#
# Prerequisites:
#   - conda environment 'interference' (see environment.yml)
#   - GPU recommended; set CUDA_VISIBLE_DEVICES="" to force CPU
#   - For pcgrad runs: stage-1 checkpoints are generated automatically

# ── Arguments ─────────────────────────────────────────────────────────────────
MODEL_TYPE=${1:-iiLoss}   # iiLoss | HdLoss
MODE=${2:-single}         # single | pcgrad
TARGET=${3:-Hf}           # Hf | Hd

if [[ "$MODEL_TYPE" != "iiLoss" && "$MODEL_TYPE" != "HdLoss" ]]; then
    echo "ERROR: MODEL_TYPE must be 'iiLoss' or 'HdLoss', got '$MODEL_TYPE'"
    exit 1
fi

# ── Hyperparameter values ──────────────────────────────────────────────────────
if [[ "$MODE" == "pcgrad" ]]; then
    VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
else
    VALUES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
fi

# ── Environment ────────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate interference 2>/dev/null
PYTHON=$(which python 2>/dev/null || which python3)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

LOGFILE="logs/sweep_${MODEL_TYPE}_${MODE}_${TARGET}.log"

echo "============================================================" | tee "$LOGFILE"
echo "  ${MODEL_TYPE} sweep [${MODE}] — target: ${TARGET}"        | tee -a "$LOGFILE"
echo "  Values: ${VALUES[*]}"                                      | tee -a "$LOGFILE"
echo "  Python: $PYTHON ($($PYTHON --version 2>&1))"              | tee -a "$LOGFILE"
echo "  Started: $(date)"                                          | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# ── Stage 1: save MSE checkpoints (PCGrad only) ────────────────────────────────
if [[ "$MODE" == "pcgrad" ]]; then
    CKPT_DIR="checkpoints/${TARGET}"
    NEED_STAGE1=false
    for fold in 0 1 2 3 4; do
        [[ ! -f "${CKPT_DIR}/${TARGET}_fold${fold}.pt" ]] && NEED_STAGE1=true && break
    done

    if $NEED_STAGE1; then
        echo "" | tee -a "$LOGFILE"
        echo "Stage 1: training iiLoss_save_0.0 (pure MSE) to generate checkpoints..." | tee -a "$LOGFILE"
        $PYTHON -u scripts/train_models.py allMP_2026 "$TARGET" iiLoss_save_0.0 \
            2>&1 | tee -a "$LOGFILE"
        [[ ${PIPESTATUS[0]} -ne 0 ]] && echo "Stage 1 failed — aborting." && exit 1
        echo "Stage 1 done." | tee -a "$LOGFILE"
    else
        echo "Stage-1 checkpoints found — skipping." | tee -a "$LOGFILE"
    fi
fi

# ── Main sweep ─────────────────────────────────────────────────────────────────
SPLIT="allMP_2026"
N=${#VALUES[@]}

for i in "${!VALUES[@]}"; do
    VAL="${VALUES[$i]}"
    MODEL=$( [[ "$MODE" == "pcgrad" ]] \
        && echo "${MODEL_TYPE}_pcgrad_${VAL}" \
        || echo "${MODEL_TYPE}_${VAL}" )

    echo "" | tee -a "$LOGFILE"
    echo "[$((i+1))/$N]  ${MODEL}  ($(date))" | tee -a "$LOGFILE"
    echo "-----------------------------------------------------" | tee -a "$LOGFILE"

    $PYTHON -u scripts/train_models.py "$SPLIT" "$TARGET" "$MODEL" \
        2>&1 | tee -a "$LOGFILE"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "ERROR: ${MODEL} failed — stopping." | tee -a "$LOGFILE"
        exit 1
    fi
    echo "[DONE] ${MODEL} at $(date)" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo "  Sweep complete — $(date)"                                  | tee -a "$LOGFILE"
echo "  Score results:"                                            | tee -a "$LOGFILE"
echo "    python scripts/interference_score.py 2026"           | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
