#!/bin/bash
# Prospective test: train on all of MP, predict the WBM discovery set.
#
# Per seed: stage one, then the lam = 0 control and the objectives. Predictions
# go to data/wbm/ml/, training curves to data/wbm/ml/curves/.
#
# lam is the value the nested cross-validation selects most often, which is also
# the median of its 25 seed/fold choices (results/2025/nested/selection_*.json):
# 0.3 for iiLoss and HdLoss, 0.2 for the naive iiLossRaw. The first runs, at the
# lam of the preprint (iiLoss 0.2, HdLoss 0.1), are documented in
# docs/PREREGISTRATION.md; their predictions are in the Zenodo archive.
#
# Every stage-two run of a seed fine-tunes the same stage-one checkpoint, so the
# arms are paired with their control. Stage one is therefore skipped whenever its
# checkpoint exists, and the job stops rather than retrain stage one under
# stage-two results that were fine-tuned from an earlier checkpoint.
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
WBM_MD5=ae071b0918fe011e3ef63303f10338be   # of the unpacked content

source "$(dirname "$0")/_job_common.sh"

LOGFILE="logs/prospective_$(date +%Y%m%d_%H%M%S).log"
log "============================================================"
log "  Prospective WBM test: seeds ${SEEDS[*]}"
log "  Python: $PYTHON ($($PYTHON --version 2>&1))"
check_hull
if [[ ! -f "$WBM" ]] || [[ "$(zcat "$WBM" 2>/dev/null | md5sum | cut -d' ' -f1)" != "$WBM_MD5" ]]; then
    log "ERROR: $WBM is missing or not the Matbench Discovery WBM summary."
    log "       Get it from https://figshare.com/articles/dataset/22715158 (see README)."
    exit 1
fi
log "  Started: $(date)"
log "============================================================"

ARMS=(iiLoss_finetune_0.0 iiLoss_finetune_0.3 HdLoss_finetune_0.3 iiLossRaw_finetune_0.2)

for SEED in "${SEEDS[@]}"; do
    CKPT="checkpoints/prospective/s${SEED}/Hf_fold0_w1024_seed${SEED}.pt"
    if [[ -f "$CKPT" ]]; then
        log ""
        log "  stage one seed ${SEED}: checkpoint exists, skipping"
    elif compgen -G "data/wbm/ml/*_finetune_*_s${SEED}.csv.gz" > /dev/null; then
        log "ERROR: seed ${SEED} has stage-two predictions but no stage-one checkpoint."
        log "       Restore ${CKPT} (Zenodo, runs_wbm/ckpt/s${SEED}/) instead of retraining it."
        exit 1
    else
        run_step "iiLoss_save_0.0 seed ${SEED}" \
            "$PYTHON" -u scripts/prospective.py iiLoss_save_0.0 "$SEED"
    fi
    for MODEL in "${ARMS[@]}"; do
        run_step "${MODEL} seed ${SEED}" \
            "$PYTHON" -u scripts/prospective.py "$MODEL" "$SEED"
    done
done

log ""
log "============================================================"
log "  Prospective runs complete - $(date)"
log "  Evaluate with:  python scripts/prospective_score.py"
log "============================================================"
