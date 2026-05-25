"""
eval_ed_direct.py
-----------------
Evaluates models trained directly on the Ed (decomposition enthalpy) target
and compares them to Ef-trained baselines.

For Ed-trained models the predictions are Ed values per compound, so the Ed MAE
is computed directly:

    Ed MAE = mean_i |Ed_pred_i − Ed_DFT_i|

This is conceptually the same quantity reported for Ef-trained models in
interference_summary.csv, where Ed is derived by propagating Ef errors through
decomposition reactions.  Both equal mean|Ed_ML − Ed_DFT|; they just arrive
at Ed_ML differently.

Note: ξ cannot be computed for Ed-trained models because they don't produce
Ef predictions (needed to evaluate the reaction interference pattern).

Usage
-----
    # Run after:  python train_models.py allMP Ed iiLoss_0.0
    python eval_ed_direct.py

Output
------
  Console table comparing Ed-trained and Ef-trained models on Ed MAE.
"""

import os
import sys
import json
import math
import csv

HERE     = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(HERE, "mlstabilitytest")
DATA_DIR = os.path.join(REPO_DIR, "mp_data", "data")
ML_DIR   = os.path.join(REPO_DIR, "ml_data")

HULLOUT     = os.path.join(DATA_DIR, "hullout.json")
SUMMARY_CSV = os.path.join(HERE, "interference_summary.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_ed_mae_rmse(ed_pred, ed_dft):
    """
    Direct Ed MAE and RMSE.

    Parameters
    ----------
    ed_pred : dict  {formula: Ed_ML}
    ed_dft  : dict  {formula: Ed_DFT}

    Returns
    -------
    mae, rmse, n
    """
    errors = []
    for formula, pred in ed_pred.items():
        if formula in ed_dft:
            errors.append(pred - ed_dft[formula])
    if not errors:
        return float("nan"), float("nan"), 0
    n    = len(errors)
    mae  = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    return mae, rmse, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "allMP"
    if split not in ("allMP", "allMP_single"):
        print("Usage: python eval_ed_direct.py [allMP|allMP_single]")
        sys.exit(1)

    ed_ml_dir = os.path.join(ML_DIR, "Ed", split)
    print(f"Ed-trained model predictions: {ed_ml_dir}\n")

    # ── Load DFT data ────────────────────────────────────────────────────────
    print("Loading hullout.json …")
    hullout = load_json(HULLOUT)
    ed_dft  = {formula: entry["Ed"]
               for formula, entry in hullout.items()
               if isinstance(entry, dict) and "Ed" in entry}
    print(f"  {len(ed_dft)} compounds with DFT Ed values\n")

    # ── Discover Ed-trained models ────────────────────────────────────────────
    ed_model_results = {}

    if os.path.isdir(ed_ml_dir):
        model_names = sorted(os.listdir(ed_ml_dir))
        for model_name in model_names:
            pred_path = os.path.join(ed_ml_dir, model_name, "ml_input.json")
            if not os.path.exists(pred_path):
                continue
            print(f"Evaluating Ed-trained model: {model_name}")
            ed_pred = load_json(pred_path)
            mae, rmse, n = compute_ed_mae_rmse(ed_pred, ed_dft)
            ed_model_results[model_name] = {
                "mae": mae, "rmse": rmse, "n": n,
                "label": f"{model_name} (Ed-direct)"
            }
            print(f"  n={n}  Ed MAE={mae:.4f}  Ed RMSE={rmse:.4f}")
    else:
        print(f"  [WARNING] No Ed model directory found at {ed_ml_dir}")
        print("  Run:  python train_models.py allMP Ed iiLoss_0.0")

    # ── Load Ef-trained baselines from interference_summary.csv ─────────────
    baseline_results = {}
    if os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                baseline_results[row["model"]] = {
                    "mae":  float(row["Ef_MAE"]),
                    "rmse": float(row["Ef_RMSE"]),
                    "ed_mae":  float(row["Ed_MAE"]),
                    "ed_rmse": float(row["Ed_RMSE"]),
                    "rms_xi":    float(row["rms_xi"]),
                    "median_xi": float(row["median_xi"]),
                    "n": int(row["n"]),
                }
    else:
        print(f"\n[WARNING] {SUMMARY_CSV} not found — run interference_score.py first")

    # ── Print comparison table ───────────────────────────────────────────────
    sep = "─" * 90
    hdr = "{:<25}  {:>7}  {:>8}  {:>8}  {:>8}"
    print(f"\n{sep}")
    print(f"  Ed MAE Comparison  ({split})")
    print(f"  'reaction-based' = Ed computed via Ef-error propagation through reactions")
    print(f"  'direct'         = Ed predicted directly by an Ed-trained model")
    print(sep)
    print(hdr.format("Model", "N", "Ed MAE", "Ed RMSE", "ξ (RMS)"))
    print(sep)

    # --- Ed-trained models (direct Ed prediction)
    for name, res in ed_model_results.items():
        print(hdr.format(
            f"{name} [direct]",
            res["n"],
            f"{res['mae']:.4f}",
            f"{res['rmse']:.4f}",
            "  n/a  ",
        ))

    if ed_model_results:
        print()  # separator

    # --- Ef-trained baselines (reaction-based Ed)
    SHOW_MODELS = [
        "ElFrac", "Meredig", "Magpie", "AutoMat", "ElemNet", "Roost", "CGCNN",
        "iiLoss_0.0",
    ]
    # Also show any iiLoss/EdLoss variants present
    for key in sorted(baseline_results):
        if key.startswith(("iiLoss_", "EdLoss_")) and key not in SHOW_MODELS:
            SHOW_MODELS.append(key)

    for model in SHOW_MODELS:
        if model not in baseline_results:
            continue
        res = baseline_results[model]
        print(hdr.format(
            f"{model} [Ef→Ed rxn]",
            res["n"],
            f"{res['ed_mae']:.4f}",
            f"{res['ed_rmse']:.4f}",
            f"{res['rms_xi']:.4f}",
        ))

    print(sep)
    print()

    # ── Summary observation ──────────────────────────────────────────────────
    if ed_model_results and "iiLoss_0.0" in baseline_results:
        best_direct = min(ed_model_results.values(), key=lambda r: r["mae"])
        baseline_ed = baseline_results["iiLoss_0.0"]["ed_mae"]
        best_name   = min(ed_model_results, key=lambda k: ed_model_results[k]["mae"])
        print(
            f"Best Ed-direct model ({best_name}): Ed MAE = {best_direct['mae']:.4f} eV/atom"
        )
        print(
            f"iiLoss_0.0 Ef-trained baseline:  Ed MAE = {baseline_ed:.4f} eV/atom"
            f"  (reaction-based)"
        )
        diff = best_direct["mae"] - baseline_ed
        sign = "+" if diff > 0 else ""
        print(f"Δ Ed MAE (direct − baseline) = {sign}{diff:.4f} eV/atom")
        if diff < 0:
            print(
                "  → Ed-direct training reduces Ed MAE vs Ef-trained baseline "
                "(expected, since Ed is the direct training target)."
            )
        else:
            print(
                "  → Ef-trained baseline matches or beats Ed-direct training on Ed MAE "
                "(not unusual: Ef covers all reaction members, providing implicit "
                "regularisation)."
            )


if __name__ == "__main__":
    main()
