#!/bin/bash
# Nested cross-validation on the 2025 MP hull: lam chosen on a validation split.
#
# Per seed and outer fold (scripts/nested_cv.py, protocol in its docstring):
#   inner   stage 1, the lam = 0 control, and iiLoss, iiLossRaw and HdLoss at
#           every lam of the grid, scored on the validation split
#   outer   stage 1 again, then the control, each objective at its selected
#           lam and at the preprint's fixed lam, and iiLoss at two values of
#           the denominator stabiliser
# then the outer folds are merged into data/2025/ml/Hf/<run>_s<seed>/.
#
# Usage:
#   bash scripts/nested.sh               # seeds 0 to 4, as in the paper
#   bash scripts/nested.sh 0 1           # a subset; finished folds are skipped
#
# Requires the 'interference' conda environment to be active.

set -u

SEEDS=("$@")
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(0 1 2 3 4)

source "$(dirname "$0")/_job_common.sh"

LOGFILE="logs/nested_$(date +%Y%m%d_%H%M%S).log"
log "============================================================"
log "  Nested CV: seeds ${SEEDS[*]}, 5 outer folds each"
log "  Python: $PYTHON ($($PYTHON --version 2>&1))"
check_hull
log "  Started: $(date)"
log "============================================================"

for SEED in "${SEEDS[@]}"; do
    for FOLD in 0 1 2 3 4; do
        run_step "seed ${SEED} fold ${FOLD}" \
            "$PYTHON" -u scripts/nested_cv.py run "$SEED" "$FOLD"
    done
done
run_step "merge" "$PYTHON" scripts/nested_cv.py merge

log ""
log "============================================================"
log "  Nested CV complete - $(date)"
log "  Evaluate with:"
log "    python scripts/interference_score.py 2025 --summary-only"
log "    python scripts/spectral_score.py 2025"
log "    python scripts/summarise_revision.py"
log "============================================================"
