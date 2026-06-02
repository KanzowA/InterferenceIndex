#!/bin/bash
# sweep.sh — Full hyperparameter sweep over all 4 model variants.
#
# Structure (42 runs total):
#   [1/42]  λ=0.0  →  iiLoss_save_0.0  (MSE baseline + checkpoints)
#                      output copied to iiLoss_0.0 and HdLoss_0.0
#   [2/42]  λ=0.0  →  iiLoss_finetune_0.0  (MSE fine-tune control, 500+200 epochs)
#                      output copied to HdLoss_finetune_0.0
#                      serves as λ=0 anchor for PCGrad curves in Fig. 6
#   [3..42] λ=0.1..1.0  ×  {iiLoss, iiLoss_pcgrad, HdLoss, HdLoss_pcgrad}
#
# Usage:
#   bash sweep.sh        # target: Hf (default)
#   bash sweep.sh Hd     # target: Hd
#
# Prerequisites:
#   - conda environment 'interference' (see environment.yml)
#   - GPU recommended; set CUDA_VISIBLE_DEVICES=0 if needed

TARGET=${1:-Hf}
SPLIT="allMP_2026"
VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
VARIANTS=(iiLoss iiLoss_pcgrad HdLoss HdLoss_pcgrad)

# ── Environment ───────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate interference 2>/dev/null
PYTHON=$(which python 2>/dev/null || which python3)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs
LOGFILE="logs/sweep_full_${TARGET}.log"
ML_DIR="data/2026/ml/${TARGET}"

echo "============================================================" | tee "$LOGFILE"
echo "  Full sweep — target: ${TARGET} — 42 runs"                  | tee -a "$LOGFILE"
echo "  Python: $PYTHON ($($PYTHON --version 2>&1))"               | tee -a "$LOGFILE"
echo "  Started: $(date)"                                           | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# ── [1/42] λ=0.0 — MSE baseline, saves checkpoints ──────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[1/42]  iiLoss_save_0.0  (λ=0, MSE baseline + checkpoints)  ($(date))" | tee -a "$LOGFILE"
echo "-----------------------------------------------------" | tee -a "$LOGFILE"

$PYTHON -u scripts/train_models.py "$SPLIT" "$TARGET" iiLoss_save_0.0 \
    2>&1 | tee -a "$LOGFILE"

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "ERROR: λ=0 baseline failed — aborting." | tee -a "$LOGFILE"
    exit 1
fi

# Copy λ=0 predictions to iiLoss_0.0 and HdLoss_0.0 (identical: pure MSE)
for DIR in iiLoss_0.0 HdLoss_0.0; do
    mkdir -p "${ML_DIR}/${DIR}"
    cp "${ML_DIR}/iiLoss_save_0.0/ml_input.json" "${ML_DIR}/${DIR}/ml_input.json"
    echo "  → copied λ=0 predictions to ${DIR}" | tee -a "$LOGFILE"
done
rm -rf "${ML_DIR}/iiLoss_save_0.0"
echo "  → removed iiLoss_save_0.0 from ML_DIR (checkpoints kept)" | tee -a "$LOGFILE"
echo "[DONE] λ=0 baseline at $(date)" | tee -a "$LOGFILE"

# ── [2/42] λ=0.0 fine-tune control — λ=0 anchor for PCGrad curves ────────────
echo "" | tee -a "$LOGFILE"
echo "[2/42]  iiLoss_finetune_0.0  (λ=0, MSE fine-tune control, 500+200 epochs)  ($(date))" | tee -a "$LOGFILE"
echo "-----------------------------------------------------" | tee -a "$LOGFILE"

$PYTHON -u scripts/train_models.py "$SPLIT" "$TARGET" iiLoss_finetune_0.0 \
    2>&1 | tee -a "$LOGFILE"

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "ERROR: iiLoss_finetune_0.0 failed — aborting." | tee -a "$LOGFILE"
    exit 1
fi

# Copy to HdLoss_finetune_0.0 (lam=0 → identical regardless of model class)
mkdir -p "${ML_DIR}/HdLoss_finetune_0.0"
cp "${ML_DIR}/iiLoss_finetune_0.0/ml_input.json" \
   "${ML_DIR}/HdLoss_finetune_0.0/ml_input.json"
echo "  → copied finetune λ=0 predictions to HdLoss_finetune_0.0" | tee -a "$LOGFILE"
echo "[DONE] finetune λ=0 control at $(date)" | tee -a "$LOGFILE"

# ── [3..42] λ=0.1..1.0 × 4 variants ─────────────────────────────────────────
IDX=3

for VAL in "${VALUES[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do

        MODEL="${VARIANT}_${VAL}"   # e.g. iiLoss_0.1, iiLoss_pcgrad_0.1

        echo "" | tee -a "$LOGFILE"
        echo "[${IDX}/42]  ${MODEL}  ($(date))" | tee -a "$LOGFILE"
        echo "-----------------------------------------------------" | tee -a "$LOGFILE"

        $PYTHON -u scripts/train_models.py "$SPLIT" "$TARGET" "$MODEL" \
            2>&1 | tee -a "$LOGFILE"

        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            echo "ERROR: ${MODEL} failed — stopping." | tee -a "$LOGFILE"
            exit 1
        fi
        echo "[DONE] ${MODEL} at $(date)" | tee -a "$LOGFILE"
        ((IDX++))
    done
done

echo "" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
echo "  Sweep complete — $(date)"                                   | tee -a "$LOGFILE"
echo "  Evaluate with:"                                             | tee -a "$LOGFILE"
echo "    python scripts/interference_score.py 2026"               | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
