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
  interference_summary.csv  — MAE, RMSE, mean/median ξ per model
"""

import os
import sys
import json
import math
import csv
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

DATA_ROOT = os.path.join(REPO_ROOT, "data")

HULLOUT_MAP = {
    "2020": os.path.join(DATA_ROOT, "2020", "hullout_2020.json"),
    "2026": os.path.join(DATA_ROOT, "2026", "hullout_2026.json"),
    "2026_single": os.path.join(DATA_ROOT, "2026", "hullout_2026.json"),
}

HF_DFT = os.path.join(DATA_ROOT, "2020", "Hf.json")

# Base Bartel-et-al. models (fixed order)
MODELS_BASE = ["ElFrac", "Meredig", "Magpie", "AutoMat", "ElemNet",
               "Roost", "CGCNN"]

# iiLoss_* variants are auto-detected at runtime from the ml_data folder.
# MODELS is rebuilt in main() — do not edit manually.
MODELS = list(MODELS_BASE)


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
        if "_" not in token:
            # bare formula with no coefficient — treat as weight 1.0
            products.append((1.0, token))
            continue
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


def interference_score(compound, rxn_str, ml_hf, dft_hf, mode='error'):
    products = parse_rxn(rxn_str)
    N_c = num_atoms(compound)
    c = []

    if compound not in ml_hf or compound not in dft_hf:
        return None
    if not is_element(compound):
        delta_c = ml_hf[compound] - dft_hf[compound] if mode == 'error' else ml_hf[compound]
        c.append(-delta_c)

    for amt_k, formula_k in products:
        if is_element(formula_k):
            continue
        if formula_k not in ml_hf or formula_k not in dft_hf:
            return None
        delta_k = ml_hf[formula_k] - dft_hf[formula_k] if mode == 'error' else ml_hf[formula_k]
        N_k = num_atoms(formula_k)
        c.append(amt_k * N_k * delta_k / N_c)

    if len(c) == 0:
        return None

    signed_sum  = sum(c)           # = Hd_DFT - Hd_ML  (for mode='error')
    numerator   = abs(signed_sum)
    denominator = math.sqrt(sum(x**2 for x in c))

    if denominator < 1e-12:
        return None

    return numerator / denominator, len(c), signed_sum


def compute_mae_rmse(ml_hf, dft_hf, reference_compounds=None):
    """MAE and RMSE over a defined reference set.

    Parameters
    ----------
    ml_hf : dict  {formula: predicted_Hf}
    dft_hf : dict {formula: dft_Hf}
    reference_compounds : iterable, optional
        The full set of compounds over which MAE is defined.
        If given, the denominator is len(reference_compounds) and compounds
        missing from ml_hf are counted as errors of NaN (excluded from sum
        but the denominator reflects the full set size).
        If None, iterates over ml_hf keys (legacy behaviour).
    """
    errors = []
    n_missing = 0
    compounds = reference_compounds if reference_compounds is not None else ml_hf.keys()
    for formula in compounds:
        if formula not in dft_hf:
            continue
        if formula not in ml_hf:
            n_missing += 1
            continue
        errors.append(ml_hf[formula] - dft_hf[formula])
    if n_missing:
        print(f"    [MAE] {n_missing} compounds in reference set have no ML prediction — excluded from MAE")
    if not errors:
        return float('nan'), float('nan')
    mae  = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
    return mae, rmse


# ── Plotting ──────────────────────────────────────────────────────────────────
# Style constants matching figure2_circles.py
_CMAP    = 'RdBu_r'
_COL_REF = '#6B7280'   # ξ = 1 reference line

def plot_interference_circles(results, models):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                         'axes.linewidth': 0.8})

    by_model = defaultdict(list)
    for row in results:
        by_model[row["model"]].append(row)

    models_present = [m for m in MODELS if m in by_model]
    models_filtered = [m for m in models_present if m in models]
    n_models = len(models_filtered)
    if n_models == 0:
        return

    # Grid layout: up to 4 columns
    n_cols = min(n_models, 4)
    n_rows = math.ceil(n_models / n_cols)

    # Shared geometry: common lim across all models so panels are comparable
    all_N = [r["N"] for m in models_filtered for r in by_model[m]]
    lim   = math.sqrt(max(all_N)) + 0.2   # top of circle frame = base of histogram

    # Histogram height in data coords (40 % of circle region)
    HIST_H = lim * 0.40
    FULL_H = lim + HIST_H

    # Fixed density scale: compute global max across all models
    hist_max = 0.0
    for model_name in models_filtered:
        xis = np.array([r["xi_err"] for r in by_model[model_name]])
        counts, _ = np.histogram(xis, bins=35, range=(0, lim), density=True)
        if counts.size:
            hist_max = max(hist_max, float(counts.max()))
    HIST_SCALE = HIST_H / hist_max if hist_max > 0 else 1.0

    norm     = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=math.sqrt(10))
    cmap_obj = plt.get_cmap(_CMAP)
    arc_theta = np.linspace(0, np.pi / 2, 300)

    tick_vals = [t for t in [0, 1, 2, 3] if t <= lim]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols + 0.8, 5.5 * n_rows),
                             squeeze=False)

    for idx, model_name in enumerate(models_filtered):
        ax = axes[idx // n_cols, idx % n_cols]

        rows   = by_model[model_name]
        Ns     = np.array([r["N"]         for r in rows])
        xis    = np.array([r["xi_err"]    for r in rows])
        deltas = np.array([r["eta_err"] for r in rows])

        # Concentric arcs + N labels
        for N_val in sorted(set(Ns)):
            R = math.sqrt(N_val)
            ax.plot(R * np.cos(arc_theta), R * np.sin(arc_theta),
                    color='lightgray', lw=1.0, zorder=0)
            ax.text(R * np.cos(0.24) + 0.04 - R * 0.003,
                    R * np.sin(0.24),
                    f'N={N_val}', color='gray', fontsize=7,
                    va='top', ha='left', rotation=-80)

        # Scatter coloured by xi
        ax.scatter(xis, deltas, c=xis, cmap=_CMAP, norm=norm,
                   s=7, alpha=0.45, zorder=3)

        # Per-N RMS-xi diamond
        first = True
        for N_val in sorted(set(Ns)):
            xis_N = xis[Ns == N_val]
            rms_N = math.sqrt(float(np.mean(xis_N ** 2)))
            eta_N = math.sqrt(max(N_val - rms_N ** 2, 0))
            label = r'$\sqrt{\langle\xi^2\rangle}_N$' if first else None
            ax.scatter([rms_N], [eta_N], color='black', s=30,
                       marker='D', zorder=5, label=label)
            first = False

        # Histogram bars floating above the frame (clip_on=False)
        rms_xi = math.sqrt(float(np.mean(xis ** 2)))
        counts, edges = np.histogram(xis, bins=35, range=(0, lim), density=True)
        bar_tops = lim + counts * HIST_SCALE   # top y of each bar in data coords
        for left, right, h, bar_top in zip(edges[:-1], edges[1:], counts, bar_tops):
            xi_mid = (left + right) / 2
            bar_h  = h * HIST_SCALE
            ax.bar(left, bar_h, width=right - left, bottom=lim,
                   color=cmap_obj(norm(xi_mid)),
                   align='edge', edgecolor='none', alpha=0.85,
                   zorder=3, clip_on=False)

        def _vline_top(x_val):
            """Return the top of the histogram bar that x_val falls in (or lim if none)."""
            for i in range(len(counts)):
                if edges[i] <= x_val < edges[i + 1]:
                    return float(lim + counts[i] * HIST_SCALE)
            return float(lim)

        # Reference lines — extend only to the top of their overlapping histogram bar
        xi1_top = _vline_top(1.0)
        rms_top = _vline_top(rms_xi)
        ax.plot([1.0, 1.0], [0, xi1_top], color=_COL_REF, lw=0.9, ls=':',
                alpha=0.7, zorder=4, clip_on=False, label=r'$\xi = 1$')
        ax.plot([rms_xi, rms_xi], [0, rms_top], color='black', lw=1.2, ls='--',
                zorder=5, clip_on=False,
                label=r'$\sqrt{\langle\xi^2\rangle} = ' + rf'{rms_xi:.3f}$')

        # Full box frame; ticks and labels on bottom and left only
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        
        ax.set_xlabel(r'$\xi = \sqrt{N}\cos\theta$', fontsize=11)
        ax.set_ylabel(r'$\eta = \sqrt{N}\sin\theta$', fontsize=11)
        ax.set_aspect('equal')
        ax.tick_params(top=False, right=False)
        ax.legend(fontsize=8, loc='upper right')

        # Model name above the histogram (no letter label)
        ax.text(1.0, 1.1, model_name,
                transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='bottom',
                clip_on=False)

    # Hide any unused axes in the grid
    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if c != 0:
                ax.tick_params(labelleft=False)
                ax.set_ylabel('')
            if r != n_rows - 1:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel('')
        
        
    plt.tight_layout(w_pad=4.0)
    fig.subplots_adjust(right=0.88)

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.70])
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'$\xi$', fontsize=13)
    cbar.set_ticks([0, 1, math.sqrt(10)])
    cbar.set_ticklabels(['0', '1', r'$\sqrt{10}$'])

    out_fig = os.path.join(REPO_ROOT, "figures", "interference_circles.png")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_fig}")


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
        xis = np.array([r["xi_err"] for r in rows])
        ed_dft = np.array([r["Hd_DFT"]    for r in rows])
        stable = np.array([r["stability"]  for r in rows])

        ax = axes[row_idx][0]
        for is_stable, marker, label in [(True, 'o', 'stable'), (False, 'x', 'unstable')]:
            mask = stable == is_stable
            ax.scatter(xis[mask], np.abs(ed_dft[mask]),
                       c=color, marker=marker, s=10, alpha=0.3, label=label)
        ax.set_xlabel(r"$\xi$ (interference index, error-based)")
        ax.set_ylabel(r"$|\Delta H_d^{DFT}|$ (eV/atom)")
        ax.set_title(f"{model_name} — score vs. |Ed|")
        ax.legend(fontsize=8, markerscale=2)

        ax = axes[row_idx][1]
        bins = np.linspace(0, np.percentile(np.abs(ed_dft), 95), 20)
        centers = 0.5 * (bins[:-1] + bins[1:])
        for is_stable, ls, label in [(True, '-', 'stable'), (False, '--', 'unstable')]:
            mask = stable == is_stable
            xi_sub = xis[mask]
            ed_sub = np.abs(ed_dft[mask])
            means  = [xi_sub[(ed_sub >= bins[i]) & (ed_sub < bins[i+1])].mean()
                      for i in range(len(bins) - 1)]
            means  = [m if not np.isnan(m) else None for m in means]
            ax.plot(centers, means, color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel(r"$|\Delta H_d^{DFT}|$ (eV/atom)")
        ax.set_ylabel(r"mean $\xi$")
        ax.set_title(f"{model_name} — mean score vs. |Ed| (binned)")
        ax.legend(fontsize=8)

    plt.suptitle("Interference score vs. decomposition energy", fontsize=14, y=1.001)
    plt.tight_layout()
    out_fig = os.path.join(REPO_ROOT, "figures", "score_vs_error.png")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_fig}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Parse arguments ───────────────────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser(description="Compute interference scores for ML models.")
    parser.add_argument("split", nargs="?", default="2020",
                        help="Data split to evaluate (default: 2020).")
    parser.add_argument("--hullout", default=None,
                        help="Path to hullout JSON file (default: data/2020/hullout_2020.json). "
                             "Use hullout_current.json for 2026 MP data.")
    args = parser.parse_args()
    split = args.split

    if split.startswith("2020"):
        ML_DIR = os.path.join(DATA_ROOT, "2020", "ml", "Hf")
    elif split.startswith("2026"):
        ML_DIR = os.path.join(DATA_ROOT, "2026", "ml", "Hf")
    else:
        raise ValueError(f"Unknown split: {split}")
    print(f"Using split: {split}  →  {ML_DIR}\n")

    # Select hullout: auto-map by split, then override with --hullout if given
    HULLOUT = HULLOUT_MAP[split]
    print(f"Using hullout: {HULLOUT}\n")

    # ── Auto-detect iiLoss_* and HdLoss_* variants ────────────────────────────
    global MODELS

    def _detect_variants(prefix, ml_dir):
        variants = []
        if os.path.isdir(ml_dir):
            for name in os.listdir(ml_dir):
                if name.startswith(prefix) and os.path.isdir(os.path.join(ml_dir, name)):
                    try:
                        float(name.rsplit("_", 1)[1])  # handles iiLoss_finetune_0.1 etc.
                        variants.append(name)
                    except ValueError:
                        pass
        variants.sort(key=lambda n: (n.rsplit("_", 1)[0], float(n.rsplit("_", 1)[1])))
        return variants

    ii_variants = _detect_variants("iiLoss_", ML_DIR)
    ed_variants = _detect_variants("HdLoss_", ML_DIR)

    MODELS = list(MODELS_BASE) + ii_variants + ed_variants

    if ii_variants:
        print(f"Auto-detected iiLoss variants: {ii_variants}")
    if ed_variants:
        print(f"Auto-detected HdLoss variants: {ed_variants}")
    if not ii_variants and not ed_variants:
        print("No iiLoss_* or HdLoss_* variants found in ml_data folder.")

    print("Loading DFT reference data …")
    hullout = load_json(HULLOUT)

    # Full DFT reference (85,014): used for Ef MAE — includes elements (Ef=0)
    dft_hf_all = {f: v["Hf"] for f, v in hullout.items() if isinstance(v, dict) and "Hf" in v}

    # Non-elemental compounds with a reaction string: used for interference scoring
    valid_compounds = [
        f for f, v in hullout.items()
        if isinstance(v, dict) and "rxn" in v and not is_element(f)
    ]
    # Ef reference restricted to valid_compounds (for interference score internals)
    dft_hf = {f: hullout[f]["Hf"] for f in valid_compounds}
    print(f"  {len(dft_hf_all)} total compounds in hullout (MAE reference)")
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
        ml_hf = load_json(ml_input_path)

        # ── MAE / RMSE over all 85,014 compounds in hullout ──────────────
        mae, rmse = compute_mae_rmse(ml_hf, dft_hf_all, reference_compounds=list(dft_hf_all.keys()))

        # ── Per-compound ξ scores and Ed errors ───────────────────────────
        scores    = []
        hd_errors = []   # Hd_ML - Hd_DFT per compound (includes N=1)
        n_ok, n_skip = 0, 0

        for compound in valid_compounds:
            entry   = hullout[compound]
            rxn_str = entry["rxn"]
            result_err = interference_score(compound, rxn_str, ml_hf, dft_hf, mode='error')

            if result_err is None:
                n_skip += 1
                continue

            xi_err, N, signed_sum = result_err

            # Ed MAE: include ALL compounds (N=1 and above)
            # Hd_ML - Hd_DFT = -signed_sum  (signed_sum = Hd_DFT - Hd_ML)
            hd_err = -signed_sum
            hd_errors.append(hd_err)

            # xi: exclude single-participant reactions (N=1 is geometrically undefined)
            if N == 1:
                n_skip += 1
                continue

            result_ml = interference_score(compound, rxn_str, ml_hf, dft_hf, mode='ml')
            if result_ml is None:
                n_skip += 1
                continue

            sqrt_N    = math.sqrt(N)
            theta_err = math.acos(min(xi_err / sqrt_N, 1.0))
            eta_err = sqrt_N * math.sin(theta_err)

            xi_ml    = result_ml[0]
            theta_ml = math.acos(min(xi_ml / sqrt_N, 1.0))
            eta_ml = sqrt_N * math.sin(theta_ml)

            n_ok += 1
            scores.append(xi_err)
            results.append({
                "model":     model,
                "compound":  compound,
                "stability": entry["stability"],
                "Hf_DFT":    entry["Hf"],
                "Hd_DFT":    entry["Hd"],
                "Hd_err":    round(hd_err, 6),
                "xi_err":    round(xi_err, 6),
                "eta_err":   round(eta_err, 6),
                "theta_err": round(math.degrees(theta_err), 4),
                "xi_ml":     round(xi_ml, 6),
                "eta_ml":    round(eta_ml, 6),
                "theta_ml":  round(math.degrees(theta_ml), 4),
                "N":         N,
                "sqrt_N":    round(sqrt_N, 4),
            })

        # Ed MAE / RMSE over all compounds with reactions (including N=1)
        hd_mae  = sum(abs(e) for e in hd_errors) / len(hd_errors) if hd_errors else float('nan')
        hd_rmse = math.sqrt(sum(e**2 for e in hd_errors) / len(hd_errors)) if hd_errors else float('nan')

        print(f"  scored {n_ok} for xi  |  {len(hd_errors)} for Hd MAE  |  skipped {n_skip}")
        summary[model] = {"scores": scores, "mae": mae, "rmse": rmse,
                          "hd_mae": hd_mae, "hd_rmse": hd_rmse,
                          "n": n_ok, "n_hd": len(hd_errors)}

    # ── Write per-compound CSV ─────────────────────────────────────────────────
    year = "2020" if split.startswith("2020") else "2026"
    out_dir = os.path.join(REPO_ROOT, "results", year)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"interference_scores_{year}.csv")
    fieldnames = ["model", "compound", "stability", "Hf_DFT", "Hd_DFT", "Hd_err",
                  "xi_err",   "eta_err", "theta_err",
                  "xi_ml",    "eta_ml",  "theta_ml",
                  "N", "sqrt_N"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nPer-compound scores written to  {out_csv}")

    # ── Write summary CSV ──────────────────────────────────────────────────────
    sum_csv = os.path.join(out_dir, f"interference_summary_{year}.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_xi", "n_hd", "Hf_MAE", "Hf_RMSE",
                    "Hd_MAE", "Hd_RMSE", "rms_xi", "median_xi"])
        for model, s in summary.items():
            sc      = sorted(s["scores"])
            n       = len(sc)
            rms_xi  = math.sqrt(sum(x**2 for x in sc) / n) if n else float('nan')
            med_xi  = sc[n // 2] if n else float('nan')
            w.writerow([model, s["n"], s["n_hd"],
                        round(s["mae"],     4),
                        round(s["rmse"],    4),
                        round(s["hd_mae"],  4),
                        round(s["hd_rmse"], 4),
                        round(rms_xi,       4),
                        round(med_xi,       4)])
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
        sc     = sorted(s["scores"])
        n      = len(sc)
        rms_xi = math.sqrt(sum(x**2 for x in sc) / n) if n else float('nan')
        med_xi = sc[n // 2] if n else float('nan')
        print(col.format(
            model, s["n"],
            f"{s['mae']:.4f}",
            f"{s['rmse']:.4f}",
            f"{s['hd_mae']:.4f}",
            f"{s['hd_rmse']:.4f}",
            f"{rms_xi:.4f}",
            f"{med_xi:.4f}",
        ))
    print(sep)

    plot_interference_circles(results, ['ElFrac', 'Meredig', 'Magpie', 'AutoMat', 'ElemNet', 'Roost', 'CGCNN'])


if __name__ == "__main__":
    main()
