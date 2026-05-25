"""
interference_score.py
---------------------
Computes the interference score for ML formation energy models evaluated
in Bartel et al. (npj Comput. Mater. 6, 97, 2020).

Score definition
----------------
For a decomposition reaction  X → Σ_k v_k · P_k  (per atom of X):

    c_i  = v_i · δ_i          where δ_i = ΔHf^ML_i − ΔHf^DFT_i

    score = |Σ_i c_i| / √(Σ_i c_i²)  =  √N · |cos θ|

    θ = angle between the all-ones vector and the vector c = (c_i)

    score → 0   : strong error cancellation  (good)
    score → √N  : no cancellation, errors add constructively  (bad)

Note: elements always have ΔHf = 0 by definition, so δ_element = 0 and
they never contribute to the score. Reactions that decompose entirely into
elements are assigned score = NaN.

Usage
-----
Run from the TestStabilityMl directory:

    python interference_score.py                  # uses allMP (5-fold, all compounds)
    python interference_score.py allMP_single     # uses allMP_single (80/20 split)

Output
------
  interference_scores.csv   — per-compound scores for every model
  interference_summary.csv  — MAE, RMSE, mean/median γ per model
"""

import os
import sys
import json
import math
import csv
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(HERE, "mlstabilitytest")
DATA_DIR = os.path.join(REPO_DIR, "mp_data", "data")

HULLOUT = os.path.join(DATA_DIR, "hullout.json")
EF_DFT  = os.path.join(DATA_DIR, "Ef.json")

# Base Bartel-et-al. models (fixed order)
MODELS_BASE = ["ElFrac", "Meredig", "Magpie", "AutoMat", "ElemNet",
               "Roost", "CGCNN"]

# GammaLoss_* variants are auto-detected at runtime from the ml_data folder.
# MODELS is rebuilt in main() — do not edit manually.
MODELS = list(MODELS_BASE)

EXCLUDE_N1 = True   # set False to keep single-participant reactions

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_rxn(rxn_str):
    """
    Parse a reaction string such as '0.6667_Ac + 0.3333_Ac1Ag3'.
    Returns list of (amount: float, formula: str)
    """
    products = []
    for token in rxn_str.split(" + "):
        token = token.strip()
        amt_str, formula = token.split("_", 1)
        products.append((float(amt_str), formula))
    return products


def is_element(formula):
    """True only for single-element formulas (Fe, Li, O, Fe2, S8, ...).
    The old implementation (no digits → elemental) incorrectly flagged
    binary 1:1 compounds like LiF, NaCl, AcAg as elemental — causing
    ~2,100 compounds to be silently excluded from interference scoring.
    """
    import re
    symbols = re.findall(r'[A-Z][a-z]?', formula)
    return len(set(symbols)) == 1


def num_atoms(formula):
    import re
    counts = re.findall(r'[A-Z][a-z]?(\d+)', formula)
    if not counts:
        return 1
    return sum(int(n) for n in counts)


def interference_score(compound, rxn_str, ml_ef, dft_ef, mode='error'):
    products = parse_rxn(rxn_str)
    N_c = num_atoms(compound)
    c = []

    if compound not in ml_ef or compound not in dft_ef:
        return None
    if not is_element(compound):
        delta_c = ml_ef[compound] - dft_ef[compound] if mode == 'error' else ml_ef[compound]
        c.append(-delta_c)

    for amt_k, formula_k in products:
        if is_element(formula_k):
            continue
        if formula_k not in ml_ef or formula_k not in dft_ef:
            return None
        delta_k = ml_ef[formula_k] - dft_ef[formula_k] if mode == 'error' else ml_ef[formula_k]
        N_k = num_atoms(formula_k)
        c.append(amt_k * N_k * delta_k / N_c)

    if len(c) == 0:
        return None

    signed_sum  = sum(c)           # = Ed_DFT - Ed_ML  (for mode='error')
    numerator   = abs(signed_sum)
    denominator = math.sqrt(sum(x**2 for x in c))

    if denominator < 1e-12:
        return None

    return numerator / denominator, len(c), signed_sum


def compute_mae_rmse(ml_ef, dft_ef, reference_compounds=None):
    """MAE and RMSE over a defined reference set.

    Parameters
    ----------
    ml_ef : dict  {formula: predicted_Ef}
    dft_ef : dict {formula: dft_Ef}
    reference_compounds : iterable, optional
        The full set of compounds over which MAE is defined.
        If given, the denominator is len(reference_compounds) and compounds
        missing from ml_ef are counted as errors of NaN (excluded from sum
        but the denominator reflects the full set size).
        If None, iterates over ml_ef keys (legacy behaviour).
    """
    errors = []
    n_missing = 0
    compounds = reference_compounds if reference_compounds is not None else ml_ef.keys()
    for formula in compounds:
        if formula not in dft_ef:
            continue
        if formula not in ml_ef:
            n_missing += 1
            continue
        errors.append(ml_ef[formula] - dft_ef[formula])
    if n_missing:
        print(f"    [MAE] {n_missing} compounds in reference set have no ML prediction — excluded from MAE")
    if not errors:
        return float('nan'), float('nan')
    mae  = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
    return mae, rmse


# ── Plotting (unchanged) ───────────────────────────────────────────────────────

def plot_propagation_circles(results):
    import matplotlib.pyplot as plt
    import numpy as np

    by_model = defaultdict(list)
    for row in results:
        by_model[row["model"]].append(row)

    colors = ["steelblue", "darkorange", "forestgreen",
              "crimson", "mediumpurple", "sienna", "deeppink", "teal",
              "goldenrod", "slateblue", "coral", "mediumseagreen"]
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(MODELS)}

    models_present = [m for m in MODELS if m in by_model]
    n_models = len(models_present)

    fig, axes = plt.subplots(n_models, 2, figsize=(14, 6 * n_models))
    if n_models == 1:
        axes = [axes]

    for row_idx, model_name in enumerate(models_present):
        rows   = by_model[model_name]
        color  = model_colors.get(model_name, "gray")
        Ns     = [r["N"] for r in rows]
        xis = [r["gamma_err"] for r in rows]
        deltas = [r["delta_err"] for r in rows]
        thetas = [r["theta_err"] for r in rows]
        xis_mean = []
        deltas_mean = []
        xis_average = []
        deltas_average = []
        for i in range(2, 11):
            thetas_i = [r["theta_err"] for r in rows if r["N"] == i]
            if not thetas_i:
                continue
            mean_theta_i = sum(thetas_i) / len(thetas_i)
            xis_mean.append(np.sqrt(i) * np.cos(np.radians(mean_theta_i)))
            deltas_mean.append(np.sqrt(i) * np.sin(np.radians(mean_theta_i)))

            xis_i = [r["gamma_err"] for r in rows if r["N"] == i]
            average_gamma = sum(xis_i) / len(xis_i)
            xis_average.append(average_gamma)
            deltas_average.append(np.sqrt(i) * np.sin(np.arccos(average_gamma / np.sqrt(i))))

        arc = np.linspace(0, np.pi / 2, 300)

        ax = axes[row_idx][0]
        for N_val in sorted(set(Ns)):
            r = math.sqrt(N_val)
            ax.plot(r * np.cos(arc), r * np.sin(arc),
                    color="lightgray", lw=1.0, zorder=0)
            ax.text(r * np.cos(0.24) + 0.04 - r * 0.003, r * np.sin(0.24),
                    f"N={N_val}", color="gray", fontsize=7,
                    va="top", ha="left", rotation=-80)

        ax.scatter(xis, deltas, c=color, s=6, alpha=0.3, label=model_name)
        ax.scatter(xis_mean, deltas_mean, c='k', s=24, marker='x',
                   label=r'$\langle\vartheta_N\rangle$ (mean)')
        ax.scatter(xis_average, deltas_average, c='k', s=24, marker='p',
                   label=r'$\langle\gamma\rangle$ (mean)')
        median_theta = np.radians(np.median(np.array(thetas)))
        max_r = math.sqrt(max(Ns))
        ax.plot([0, max_r * math.cos(median_theta)],
                [0, max_r * math.sin(median_theta)],
                color="gray", lw=1.0, ls=":",
                label=rf"$\langle\vartheta\rangle$ (median) $= {math.degrees(median_theta):.1f}°$")
        ax.set_xlabel(r"$\gamma = \sqrt{N}\,|\cos\vartheta|$  (propagation)")
        ax.set_ylabel(r"$\delta = \sqrt{N}\,|\sin\vartheta|$  (compensation)")
        ax.set_title(rf"{model_name} — Error geometry ($\gamma^2 + \delta^2 = N$)")
        ax.set_xlim(-0.05, None)
        ax.set_ylim(-0.05, None)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, markerscale=2)

        ax = axes[row_idx][1]
        rms_xi    = np.sqrt(np.mean(np.array(xis) ** 2))
        median_xi = np.median(np.array(xis))
        xi_max    = math.sqrt(max(Ns))
        ax.hist(xis, bins=30, range=(0, xi_max),
                color=color, alpha=0.7, density=True)
        ax.axvline(1.0, color="gray", lw=1.0, ls=":", label=r"$\xi = 1$ (null)")
        ax.axvline(rms_xi, color="black", lw=1.2, ls="--",
                   label=rf"$\sqrt{{\langle\xi^2\rangle}} = {rms_xi:.3f}$")
        ax.axvline(median_xi, color="black", lw=1.0, ls="-.",
                   label=rf"median $\xi = {median_xi:.3f}$")
        ax.set_xlabel(r"$\xi$", fontsize=12)
        ax.set_ylabel(r"$\rho(\xi)$", fontsize=11)
        ax.set_title(f"{model_name} — $\\xi$ distribution")
        ax.set_xlim(0, xi_max)
        ax.legend(fontsize=8)

    plt.suptitle("Propagation Score — all models", fontsize=14, y=1.001)
    plt.tight_layout()
    plt.savefig("propagation_circles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure saved to propagation_circles.png")


def plot_score_vs_error(results, hullout):
    import matplotlib.pyplot as plt
    import numpy as np

    by_model = defaultdict(list)
    for row in results:
        by_model[row["model"]].append(row)

    colors = ["steelblue", "darkorange", "forestgreen",
              "crimson", "mediumpurple", "sienna", "deeppink", "teal",
              "goldenrod", "slateblue", "coral", "mediumseagreen"]
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(MODELS)}
    models_present = [m for m in MODELS if m in by_model]
    n_models = len(models_present)

    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    for row_idx, model_name in enumerate(models_present):
        rows  = by_model[model_name]
        color = model_colors.get(model_name, "gray")
        xis = np.array([r["gamma_err"] for r in rows])
        ed_dft = np.array([r["Ed_DFT"]    for r in rows])
        stable = np.array([r["stability"]  for r in rows])

        ax = axes[row_idx][0]
        for is_stable, marker, label in [(True, 'o', 'stable'), (False, 'x', 'unstable')]:
            mask = stable == is_stable
            ax.scatter(xis[mask], np.abs(ed_dft[mask]),
                       c=color, marker=marker, s=10, alpha=0.3, label=label)
        ax.set_xlabel(r"$\gamma$ (interference score, error-based)")
        ax.set_ylabel(r"$|\Delta H_d^{DFT}|$ (eV/atom)")
        ax.set_title(f"{model_name} — score vs. |Ed|")
        ax.legend(fontsize=8, markerscale=2)

        ax = axes[row_idx][1]
        bins = np.linspace(0, np.percentile(np.abs(ed_dft), 95), 20)
        centers = 0.5 * (bins[:-1] + bins[1:])
        for is_stable, ls, label in [(True, '-', 'stable'), (False, '--', 'unstable')]:
            mask = stable == is_stable
            g_sub  = xis[mask]
            ed_sub = np.abs(ed_dft[mask])
            means  = [g_sub[(ed_sub >= bins[i]) & (ed_sub < bins[i+1])].mean()
                      for i in range(len(bins) - 1)]
            means  = [m if not np.isnan(m) else None for m in means]
            ax.plot(centers, means, color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel(r"$|\Delta H_d^{DFT}|$ (eV/atom)")
        ax.set_ylabel(r"mean $\gamma$")
        ax.set_title(f"{model_name} — mean score vs. |Ed| (binned)")
        ax.legend(fontsize=8)

    plt.suptitle("Interference score vs. decomposition energy", fontsize=14, y=1.001)
    plt.tight_layout()
    plt.savefig("score_vs_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure saved to score_vs_error.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Parse optional split argument ─────────────────────────────────────────
    split = sys.argv[1] if len(sys.argv) > 1 else "allMP"
    if split not in ("allMP", "allMP_single"):
        print("Usage: python interference_score.py [allMP|allMP_single]")
        sys.exit(1)

    ML_DIR = os.path.join(REPO_DIR, "ml_data", "Ef", split)
    print(f"Using split: {split}  →  {ML_DIR}\n")

    # ── Auto-detect GammaLoss_* and EdLoss_* variants ─────────────────────────
    global MODELS

    def _detect_variants(prefix, ml_dir):
        variants = []
        if os.path.isdir(ml_dir):
            for name in os.listdir(ml_dir):
                if name.startswith(prefix) and os.path.isdir(os.path.join(ml_dir, name)):
                    try:
                        float(name.split("_", 1)[1])
                        variants.append(name)
                    except ValueError:
                        pass
        variants.sort(key=lambda n: float(n.split("_", 1)[1]))
        return variants

    gamma_variants = _detect_variants("GammaLoss_", ML_DIR)
    ed_variants    = _detect_variants("EdLoss_",    ML_DIR)

    MODELS = list(MODELS_BASE) + gamma_variants + ed_variants

    if gamma_variants:
        print(f"Auto-detected GammaLoss variants: {gamma_variants}")
    if ed_variants:
        print(f"Auto-detected EdLoss variants:    {ed_variants}")
    if not gamma_variants and not ed_variants:
        print("No GammaLoss_* or EdLoss_* variants found in ml_data folder.")

    print("Loading DFT reference data …")
    hullout = load_json(HULLOUT)

    # Full DFT reference (85,014): used for Ef MAE — includes elements (Ef=0)
    dft_ef_all = {f: v["Ef"] for f, v in hullout.items() if isinstance(v, dict) and "Ef" in v}

    # Non-elemental compounds with a reaction string: used for interference scoring
    valid_compounds = [
        f for f, v in hullout.items()
        if isinstance(v, dict) and "rxn" in v and not is_element(f)
    ]
    # Ef reference restricted to valid_compounds (for interference score internals)
    dft_ef = {f: hullout[f]["Ef"] for f in valid_compounds}
    print(f"  {len(dft_ef_all)} total compounds in hullout (MAE reference)")
    print(f"  {len(valid_compounds)} non-elemental compounds for interference scoring\n")

    results = []
    # Per-model accumulators for the summary table
    summary = {}   # model → {"scores": [], "mae": float, "rmse": float, "n": int}

    for model in MODELS:
        ml_input_path = os.path.join(ML_DIR, model, "ml_input.json")
        if not os.path.exists(ml_input_path):
            print(f"  [SKIP] {model}: ml_input.json not found at {ml_input_path}")
            continue

        print(f"Processing {model} …")
        ml_ef = load_json(ml_input_path)

        # ── MAE / RMSE over all 85,014 compounds in hullout ──────────────
        mae, rmse = compute_mae_rmse(ml_ef, dft_ef_all, reference_compounds=list(dft_ef_all.keys()))

        # ── Per-compound γ scores and Ed errors ───────────────────────────
        scores    = []
        ed_errors = []   # Ed_ML - Ed_DFT  per compound
        n_ok, n_skip = 0, 0

        for compound in valid_compounds:
            entry   = hullout[compound]
            rxn_str = entry["rxn"]
            result_err = interference_score(compound, rxn_str, ml_ef, dft_ef, mode='error')
            result_ml  = interference_score(compound, rxn_str, ml_ef, dft_ef, mode='ml')

            if result_err is None or result_ml is None:
                n_skip += 1
                continue

            gamma_err, N, signed_sum = result_err
            if EXCLUDE_N1 and N == 1:
                n_skip += 1
                continue

            # Ed_ML - Ed_DFT = -signed_sum  (since signed_sum = Ed_DFT - Ed_ML)
            ed_err = -signed_sum

            sqrt_N    = math.sqrt(N)
            theta_err = math.acos(min(gamma_err / sqrt_N, 1.0))
            delta_err = sqrt_N * math.sin(theta_err)

            gamma_ml  = result_ml[0]
            theta_ml  = math.acos(min(gamma_ml / sqrt_N, 1.0))
            delta_ml  = sqrt_N * math.sin(theta_ml)

            n_ok += 1
            scores.append(gamma_err)
            ed_errors.append(ed_err)
            results.append({
                "model":     model,
                "compound":  compound,
                "stability": entry["stability"],
                "Ef_DFT":    entry["Ef"],
                "Ed_DFT":    entry["Ed"],
                "Ed_err":    round(ed_err, 6),
                "gamma_err": round(gamma_err, 6),
                "delta_err": round(delta_err, 6),
                "theta_err": round(math.degrees(theta_err), 4),
                "gamma_ml":  round(gamma_ml, 6),
                "delta_ml":  round(delta_ml, 6),
                "theta_ml":  round(math.degrees(theta_ml), 4),
                "N":         N,
                "sqrt_N":    round(sqrt_N, 4),
            })

        # Ed MAE / RMSE over scored compounds
        ed_mae  = sum(abs(e) for e in ed_errors) / len(ed_errors) if ed_errors else float('nan')
        ed_rmse = math.sqrt(sum(e**2 for e in ed_errors) / len(ed_errors)) if ed_errors else float('nan')

        print(f"  scored {n_ok} compounds, skipped {n_skip}")
        summary[model] = {"scores": scores, "mae": mae, "rmse": rmse,
                          "ed_mae": ed_mae, "ed_rmse": ed_rmse, "n": n_ok}

    # ── Write per-compound CSV ─────────────────────────────────────────────────
    out_csv = "interference_scores.csv"
    fieldnames = ["model", "compound", "stability", "Ef_DFT", "Ed_DFT", "Ed_err",
                  "gamma_err", "delta_err", "theta_err",
                  "gamma_ml",  "delta_ml",  "theta_ml",
                  "N", "sqrt_N"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nPer-compound scores written to  {out_csv}")

    # ── Write summary CSV ──────────────────────────────────────────────────────
    sum_csv = "interference_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "Ef_MAE", "Ef_RMSE",
                    "Ed_MAE", "Ed_RMSE", "rms_xi", "median_xi"])
        for model, s in summary.items():
            sc      = sorted(s["scores"])
            n       = len(sc)
            rms_g   = math.sqrt(sum(x**2 for x in sc) / n) if n else float('nan')
            med_g   = sc[n // 2] if n else float('nan')
            w.writerow([model, s["n"],
                        round(s["mae"],     4),
                        round(s["rmse"],    4),
                        round(s["ed_mae"],  4),
                        round(s["ed_rmse"], 4),
                        round(rms_g,        4),
                        round(med_g,        4)])
    print(f"Summary written to              {sum_csv}")

    # ── Print comparison table ─────────────────────────────────────────────────
    col = "{:<12}  {:>7}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}"
    sep = "─" * 82
    print(f"\n{sep}")
    print(f"  Interference Score Summary  ({split})")
    print(f"  lower ξ = more error cancellation  |  Ed = derived decomposition enthalpy")
    print(f"  RMS ξ = sqrt(<ξ²>); null hypothesis: RMS ξ = 1 for i.i.d. residuals")
    print(sep)
    print(col.format("Model", "N", "Ef MAE", "Ef RMSE", "Ed MAE", "Ed RMSE", "RMS ξ", "Median ξ"))
    print(sep)
    for model, s in summary.items():
        sc    = sorted(s["scores"])
        n     = len(sc)
        rms_g = math.sqrt(sum(x**2 for x in sc) / n) if n else float('nan')
        med_g = sc[n // 2] if n else float('nan')
        print(col.format(
            model, s["n"],
            f"{s['mae']:.4f}",
            f"{s['rmse']:.4f}",
            f"{s['ed_mae']:.4f}",
            f"{s['ed_rmse']:.4f}",
            f"{rms_g:.4f}",
            f"{med_g:.4f}",
        ))
    print(sep)

    plot_propagation_circles(results)


if __name__ == "__main__":
    main()
