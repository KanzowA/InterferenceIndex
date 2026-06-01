#!/bin/bash
# sweep.sh — Full hyperparameter sweep over all 4 model variants.
#
# Structure (41 runs total):
#   [1/41]  λ=0.0  →  iiLoss_save_0.0  (MSE baseline + checkpoints)
#                      output copied to iiLoss_0.0 and HdLoss_0.0
#   [2..41] λ=0.1..1.0  ×  {iiLoss, iiLoss_pcgrad, HdLoss, HdLoss_pcgrad}
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
echo "  Full sweep — target: ${TARGET} — 41 runs"                  | tee -a "$LOGFILE"
echo "  Python: $PYTHON ($($PYTHON --version 2>&1))"               | tee -a "$LOGFILE"
echo "  Started: $(date)"                                           | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"

# ── [1/41] λ=0.0 — MSE baseline, shared across all variants ──────────────────
echo "" | tee -a "$LOGFILE"
echo "[1/41]  iiLoss_save_0.0  (λ=0, MSE baseline + checkpoints)  ($(date))" | tee -a "$LOGFILE"
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
# Remove iiLoss_save_0.0 from ML_DIR so it does not appear as a duplicate
# entry in interference_score.py auto-detection. Checkpoints remain in
# checkpoints/${TARGET}/ and are unaffected.
rm -rf "${ML_DIR}/iiLoss_save_0.0"
echo "  → removed iiLoss_save_0.0 from ML_DIR (checkpoints kept)" | tee -a "$LOGFILE"
echo "[DONE] λ=0 baseline at $(date)" | tee -a "$LOGFILE"

# ── [2..41] λ=0.1..1.0 × 4 variants ─────────────────────────────────────────
IDX=2

for VAL in "${VALUES[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do

        if [[ "$VARIANT" == *"pcgrad"* ]]; then
            MODEL="${VARIANT}_${VAL}"       # e.g. iiLoss_pcgrad_0.1
        else
            MODEL="${VARIANT}_${VAL}"       # e.g. iiLoss_0.1
        fi

        echo "" | tee -a "$LOGFILE"
        echo "[${IDX}/41]  ${MODEL}  ($(date))" | tee -a "$LOGFILE"
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
