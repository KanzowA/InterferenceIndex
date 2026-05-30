"""
analyze_litmno.py
-----------------
Runs StabilityAnalysis for iiLoss_* models on the LiMnTMO experiment,
generating ml_results.json files comparable to the existing Bartel et al. models.

Usage (from TestStabilityMl dir):
    python analyze_litmno.py

Add more models to MODELS below if needed (e.g. iiLoss_0.2).
"""

import os
import sys

here      = os.path.abspath(os.path.dirname(__file__))
repo_dir  = os.path.join(here, "mlstabilitytest")
ml_dir    = os.path.join(repo_dir, "ml_data")

sys.path.insert(0, here)

from mlstabilitytest.stability.StabilityAnalysis import StabilityAnalysis
from mlstabilitytest.analyze_models import process

# ── Models and experiments to evaluate ────────────────────────────────────────
MODELS     = ["iiLoss_0.0", "iiLoss_0.1"]
EXPERIMENT = "LiMnTMO"
PROP       = "Ef"

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for model in MODELS:
        data_dir = os.path.join(ml_dir, PROP, EXPERIMENT, model)
        pred_file = os.path.join(data_dir, "ml_input.json")

        if not os.path.exists(pred_file):
            print(f"[SKIP] {model}: ml_input.json not found — train first with:")
            print(f"       python mlstabilitytest/train_models.py {EXPERIMENT} {PROP} {model}")
            continue

        print(f"\nAnalysing {model} on {EXPERIMENT} ...")
        process(PROP, model, EXPERIMENT, ml_dir)
        print(f"  → ml_results.json written to {data_dir}")

    print("\nDone. Compare results in:")
    print(f"  {os.path.join(ml_dir, PROP, EXPERIMENT)}")
