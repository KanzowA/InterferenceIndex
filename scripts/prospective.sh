#!/bin/bash
# Prospective test: train on all of MP, predict the WBM discovery set.
#
# Per seed: stage one, then the lam = 0 control and the three objectives at
# their preregistered lam. Predictions go to data/wbm/ml/.
#
# Usage:
#   bash scripts/prospective.sh               # seeds 0 to 4, as in the paper
#   bash scripts/prospective.sh 0             # a subset; finished runs are skipped
#
# Requires the 'interference' conda environment to be active.

set -u

SEEDS=("$@")
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(0 1 2 3 4)
WBM=data/wbm/2023-12-13-wbm-summary.csv.gz
WBM_URL=https://github.com/janosh/matbench-discovery/raw/main/data/wbm/2023-12-13-wbm-summary.csv.gz

source "$(dirname "$0")/_job_common.sh"

LOGFILE="logs/prospective_$(date +%Y%m%d_%H%M%S).log"
log "============================================================"
log "  Prospective WBM test: seeds ${SEEDS[*]}"
log "  Python: $PYTHON ($($PYTHON --version 2>&1))"
check_hull
if [[ ! -f "$WBM" ]]; then
    mkdir -p data/wbm
    curl -sSL -o "$WBM" "$WBM_URL"
    gzip -t "$WBM" || { log "ERROR: WBM summary download failed"; rm -f "$WBM"; exit 1; }
fi
log "  Started: $(date)"
log "============================================================"

for SEED in "${SEEDS[@]}"; do
    for MODEL in iiLoss_save_0.0 iiLoss_finetune_0.0 iiLoss_finetune_0.2 \
                 iiLossRaw_finetune_0.2 HdLoss_finetune_0.1; do
        run_step "${MODEL} seed ${SEED}" \
            "$PYTHON" -u scripts/prospective.py "$MODEL" "$SEED"
    done
done

log ""
log "============================================================"
log "  Prospective runs complete - $(date)"
log "  Evaluate with:  python scripts/prospective_score.py"
log "============================================================"
