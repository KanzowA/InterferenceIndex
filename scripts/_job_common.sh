# Shared by nested.sh, capacity.sh and prospective.sh; sourced, not run.

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
mkdir -p logs

log() { echo "$@" | tee -a "$LOGFILE"; }

# The runs of the revision were trained on one hull file; stop rather than
# train on another.
check_hull() {
    local line
    line=$("$PYTHON" - <<'PYEOF'
import json, sys
sys.path.insert(0, "scripts")
from build_hull import REFERENCE_HASH, content_hash
with open("data/2025/hullout_2025.json") as f:
    digest = content_hash(json.load(f))
print(digest, "matches" if digest == REFERENCE_HASH["mp"] else "DIFFERS")
PYEOF
)
    log "  Hull: data/2025/hullout_2025.json, content hash ${line}"
    if [[ "$line" != *matches ]]; then
        log "ERROR: hullout_2025.json is not the file the revision runs used - stopping."
        exit 1
    fi
}

run_step() {   # run_step <label> <command ...>
    local label=$1
    shift
    log ""
    log "[$(date +%T)]  ${label}"
    "$@" 2>&1 | tee -a "$LOGFILE"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log "ERROR: ${label} failed - stopping."
        exit 1
    fi
}
