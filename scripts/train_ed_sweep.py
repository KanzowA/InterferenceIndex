"""
train_ed_sweep.py
-----------------
Trains EdLoss_<alpha> models using 5-fold CV on allMP/Ef, then prints
the command to score and plot them.

Why large alpha values?
  EdLossNN uses an *unnormalized* Ed MSE term:
      L = MSE_Ef/MSE_init  +  alpha * mean_rxn( Ed_error² )
  At epoch 0, Ed_MSE ≈ 0.02–0.03 eV²/atom², so alpha must be large
  (≥ 5) to contribute meaningfully relative to the Ef term (≈ 1.0).
  The small existing models (EdLoss_0.05, 0.1, 0.5) are effectively
  identical to pure MSE — use those as the low-alpha anchor points.

Usage (from TestStabilityMl dir):
    python train_ed_sweep.py
    python train_ed_sweep.py --alphas 1 5 10 25   # custom alpha values
"""

import subprocess
import sys
import os

# ── Alpha values to train ──────────────────────────────────────────────────────
# Default sweep: covers the range where EdLoss diverges from pure MSE.
# The existing EdLoss_0.05, 0.1, 0.5 act as near-zero anchors.
DEFAULT_ALPHAS = [0.01, 0.1, 1, 10, 100]

def parse_alphas():
    alphas = DEFAULT_ALPHAS
    if "--alphas" in sys.argv:
        idx = sys.argv.index("--alphas")
        alphas = [float(a) for a in sys.argv[idx + 1:]]
    return alphas

EXPERIMENT = "allMP"
PROP       = "Ef"

def alpha_to_str(a):
    """Format float alpha without trailing zeros: 5.0 → '5', 0.5 → '0.5'"""
    s = str(a)
    if s.endswith(".0"):
        s = s[:-2]
    return s

if __name__ == "__main__":
    alphas = parse_alphas()

    train_script = os.path.join("mlstabilitytest", "train_models.py")

    for alpha in alphas:
        model = f"EdLoss_{alpha_to_str(alpha)}"
        data_dir  = os.path.join("mlstabilitytest", "ml_data", PROP, EXPERIMENT, model)
        pred_file = os.path.join(data_dir, "ml_input.json")

        if os.path.exists(pred_file):
            print(f"[SKIP] {model}: ml_input.json already exists — delete to retrain")
            continue

        print(f"\n{'='*60}")
        print(f"Training {model} ({EXPERIMENT}, {PROP}) ...")
        print(f"{'='*60}")
        cmd = [sys.executable, train_script, EXPERIMENT, PROP, model]
        subprocess.run(cmd, check=True)
        print(f"  → predictions written to {pred_file}")

    print("\n" + "="*60)
    print("All EdLoss models trained (or already present).")
    print("\nNext steps:")
    print("  1. python interference_score.py          # re-score everything")
    print("  2. python plot_combined_sweep.py         # generate combined figure")
