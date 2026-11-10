#!/bin/bash
# Capacity ladder: the nested protocol at smaller base widths.
#
# The ladder of the preprint (base widths 32 to 512, three seeds), run with the
# nested protocol for iiLoss and HdLoss. Width 1024 is the nested run itself
# (scripts/nested.sh), with the same seeds and the same per-fold seeding, so it
# is not repeated here.
#
# Usage:
#   bash scripts/capacity.sh                       # widths 32 .. 512, seeds 0 1 2
#   WIDTHS="32 256" bash scripts/capacity.sh 0 1   # any subset; resumes
#
# Requires the 'interference' conda environment to be active.

set -u

SEEDS=("$@")
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(0 1 2)
WIDTHS=${WIDTHS:-"32 64 128 256 512"}

source "$(dirname "$0")/_job_common.sh"

LOGFILE="logs/capacity_$(date +%Y%m%d_%H%M%S).log"
log "============================================================"
log "  Capacity ladder: widths ${WIDTHS}, seeds ${SEEDS[*]}"
log "  Python: $PYTHON ($($PYTHON --version 2>&1))"
check_hull
log "  Started: $(date)"
log "============================================================"

for W in $WIDTHS; do
    for SEED in "${SEEDS[@]}"; do
        for FOLD in 0 1 2 3 4; do
            run_step "width ${W} seed ${SEED} fold ${FOLD}" \
                "$PYTHON" -u scripts/nested_cv.py run "$SEED" "$FOLD" "$W" \
                --objectives iiLoss,HdLoss --no-eps
        done
    done
    run_step "merge width ${W}" "$PYTHON" scripts/nested_cv.py merge "$W"
done

log ""
log "============================================================"
log "  Capacity ladder complete - $(date)"
log "============================================================"
