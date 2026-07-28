"""Interference scores for machine-learned formation enthalpy predictions.

Usage
-----
    python scripts/interference_score.py 2020
    python scripts/interference_score.py 2026

Output, written to results/<year>/
    interference_scores_<year>.csv    per-compound scores for every model
    interference_summary_<year>.csv   per-model MAE, RMSE, xi and F1
"""

import argparse
import csv
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

DATA_ROOT = os.path.join(REPO_ROOT, "data")

HULLOUT_MAP = {
    "2020": os.path.join(DATA_ROOT, "2020", "hullout_2020.json"),
    "2026": os.path.join(DATA_ROOT, "2026", "hullout_2026.json"),
    "2026_single": os.path.join(DATA_ROOT, "2026", "hullout_2026.json"),
}

MODELS_BASE = ["ElFrac", "Meredig", "Magpie", "AutoMat", "ElemNet",
               "Roost", "CGCNN"]

MODELS = list(MODELS_BASE)


# -- Helpers --------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_rxn(rxn_str):
    """Parse a reaction string into [(amount, formula), ...]."""
    products = []
    for token in rxn_str.split(" + "):
        token = token.strip()
        if "_" not in token:
            products.append((1.0, token))
            continue
        amt_str, formula = token.split("_", 1)
        products.append((float(amt_str), formula))
    return products


def is_element(formula):
    """True only for single-element formulas (Fe, Li, O, Fe2, S8, ...)"""
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    return len(set(symbols)) == 1


def num_atoms(formula):
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    return sum(int(n) if n else 1 for el, n in tokens if el)


def interference_score(compound, rxn_str, ml_hf, dft_hf, mode="error",
                       normalized_coeffs=False):
    """Interference score for one decomposition reaction.

    The 2020 files store raw stoichiometric coefficients, which must be
    rescaled by N_k / N_c; the 2026 files store per-atom weights already.
    Set normalized_coeffs accordingly.
    """
    products = parse_rxn(rxn_str)
    N_c = num_atoms(compound)
    c = []

    if compound not in ml_hf or compound not in dft_hf:
        return None
    if not is_element(compound):
        delta_c = ml_hf[compound] - dft_hf[compound] if mode == "error" else ml_hf[compound]
        c.append(-delta_c)

    for amt_k, formula_k in products:
        if is_element(formula_k):
            continue
        if formula_k not in ml_hf or formula_k not in dft_hf:
            return None
        delta_k = ml_hf[formula_k] - dft_hf[formula_k] if mode == "error" else ml_hf[formula_k]
        if normalized_coeffs:
            weight = amt_k                  # 2026: already per-atom weight
        else:
            N_k    = num_atoms(formula_k)
            weight = amt_k * N_k / N_c     # 2020: raw v_k -> normalise
        c.append(weight * delta_k)

    if len(c) == 0:
        return None

    signed_sum  = sum(c)           # = Hd_DFT - Hd_ML  (for mode='error')
    numerator   = abs(signed_sum)
    denominator = math.sqrt(sum(x**2 for x in c))

    if denominator < 1e-12:
        return None

    return numerator / denominator, len(c), signed_sum


def detect_convention(hullout, n_samples=100):
    """Detect the coefficient convention by reconstructing Hd both ways.

    True if the coefficients are pre-normalised per-atom weights, False if
    they are raw stoichiometric amounts.
    """
    dft_hf = {k: v["Hf"] for k, v in hullout.items()
              if isinstance(v, dict) and "Hf" in v}
    n_C = n_A = 0
    for compound, entry in hullout.items():
        if not isinstance(entry, dict) or "rxn" not in entry or "Hd" not in entry:
            continue
        if is_element(compound):
            continue
        N_c = num_atoms(compound)
        hd_A = hd_C = dft_hf.get(compound, None)
        if hd_A is None:
            continue
        ok = True
        for token in entry["rxn"].split(" + "):
            if "_" not in token:
                continue
            amt_str, fk = token.strip().split("_", 1)
            if is_element(fk):
                continue
            if fk not in dft_hf:
                ok = False
                break
            v_k = float(amt_str)
            N_k = num_atoms(fk)
            hd_A -= v_k * N_k / N_c * dft_hf[fk]
            hd_C -= v_k * dft_hf[fk]
        if not ok:
            continue
        err_A = abs(hd_A - entry["Hd"])
        err_C = abs(hd_C - entry["Hd"])
        if err_C < err_A:
            n_C += 1
        elif err_A < err_C:
            n_A += 1
        if n_C + n_A >= n_samples:
            break
    return n_C > n_A   # True -> normalised (Convention C / 2026)


def compute_mae_rmse(ml_hf, dft_hf, reference_compounds=None):
    """MAE and RMSE over a defined reference set.

    Parameters
    ----------
    ml_hf : dict
        Predicted formation enthalpy per formula.
    dft_hf : dict
        Reference formation enthalpy per formula.
    reference_compounds : iterable, optional
        Compounds the MAE is defined over; defaults to the predicted ones.
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
        print(f"    [MAE] {n_missing} compounds in reference set have no ML prediction - excluded from MAE")
    if not errors:
        return float("nan"), float("nan")
    mae  = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
    return mae, rmse



# -- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute interference scores for ML models.")
    parser.add_argument("split", nargs="?", default="2020",
                        help="Data split to evaluate (default: 2020).")
    parser.add_argument("--hullout", default=None)
    args = parser.parse_args()
    split = args.split

    if split.startswith("2020"):
        ML_DIR = os.path.join(DATA_ROOT, "2020", "ml", "Hf")
    elif split.startswith("2026"):
        ML_DIR = os.path.join(DATA_ROOT, "2026", "ml", "Hf")
    else:
        raise ValueError(f"Unknown split: {split}")
    print(f"Using split: {split}  ->  {ML_DIR}\n")

    HULLOUT = HULLOUT_MAP[split]
    print(f"Using hullout: {HULLOUT}\n")

    global MODELS

    def _split_variant(name):
        """Split a run name into (family, lam, seed).

        Runs are named <family>_<lam> with an optional _s<seed> suffix, so
        both "iiLoss_pcgrad_0.3" and "iiLoss_pcgrad_0.3_s2" are accepted.
        Raises ValueError if the lam field is not numeric.
        """
        stem, seed = name, -1
        head, _, tail = name.rpartition("_")
        if tail.startswith("s") and tail[1:].isdigit():
            stem, seed = head, int(tail[1:])
        family, _, lam = stem.rpartition("_")
        return family, float(lam), seed

    def _detect_variants(prefix, ml_dir):
        variants = []
        if os.path.isdir(ml_dir):
            for name in os.listdir(ml_dir):
                if name.startswith(prefix) and os.path.isdir(os.path.join(ml_dir, name)):
                    try:
                        _split_variant(name)
                        variants.append(name)
                    except ValueError:
                        pass
        variants.sort(key=_split_variant)
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

    print("Loading DFT reference data ...")
    hullout = load_json(HULLOUT)

    normalized_coeffs = detect_convention(hullout)
    print(f"  Convention: {'pre-normalised per-atom (2026-style)' if normalized_coeffs else 'raw stoichiometric (2020-style)'}")

    dft_hf_all = {f: v["Hf"] for f, v in hullout.items() if isinstance(v, dict) and "Hf" in v}
    valid_compounds = [
        f for f, v in hullout.items()
        if isinstance(v, dict) and "rxn" in v and not is_element(f)
    ]
    dft_hf = {f: hullout[f]["Hf"] for f in valid_compounds}
    print(f"  {len(dft_hf_all)} total compounds in hullout (MAE reference)")
    print(f"  {len(valid_compounds)} non-elemental compounds for interference scoring\n")

    results = []
    summary = {}   # model -> {"scores": [], "mae": float, "rmse": float, "n": int}

    for model in MODELS:
        ml_input_path = os.path.join(ML_DIR, model, "ml_input.json")
        if not os.path.exists(ml_input_path):
            print(f"  [SKIP] {model}: ml_input.json not found at {ml_input_path}")
            continue

        print(f"Processing {model} ...")
        ml_hf = load_json(ml_input_path)

        mae, rmse = compute_mae_rmse(ml_hf, dft_hf_all, reference_compounds=list(dft_hf_all.keys()))

        scores    = []
        hd_errors = []   # Hd_ML - Hd_DFT per compound (includes N=1)
        n_ok = 0
        n_skip_missing = 0   # result_err is None (compound or product absent from ml_hf)
        n_skip_N1      = 0   # N == 1 (decomposes only to elements, ii undefined)
        n_skip_ml      = 0   # result_ml is None (ML denominator vanishes)
        tp = fp = fn = tn = 0   # stability classification (Hd_ML <= 0 vs Hd_DFT <= 0)

        for compound in valid_compounds:
            entry   = hullout[compound]
            rxn_str = entry["rxn"]
            result_err = interference_score(compound, rxn_str, ml_hf, dft_hf, mode="error",
                                            normalized_coeffs=normalized_coeffs)

            if result_err is None:
                n_skip_missing += 1
                continue

            xi_err, N, signed_sum = result_err

            # signed_sum is Hd_DFT - Hd_ML, so the ML error is its negative.
            hd_err = -signed_sum
            hd_errors.append(hd_err)

            # Classified for all N, before the N == 1 skip below.
            hd_ml = entry["Hd"] + hd_err
            if entry["Hd"] <= 0:
                if hd_ml <= 0: tp += 1
                else:          fn += 1
            else:
                if hd_ml <= 0: fp += 1
                else:          tn += 1

            # A single participant leaves xi geometrically undefined.
            if N == 1:
                n_skip_N1 += 1
                continue

            result_ml = interference_score(compound, rxn_str, ml_hf, dft_hf, mode="ml",
                                           normalized_coeffs=normalized_coeffs)
            if result_ml is None:
                n_skip_ml += 1
                continue

            sqrt_N  = math.sqrt(N)
            phi_err = math.acos(min(xi_err / sqrt_N, 1.0))
            eta_err = sqrt_N * math.sin(phi_err)

            xi_ml  = result_ml[0]
            phi_ml = math.acos(min(xi_ml / sqrt_N, 1.0))
            eta_ml = sqrt_N * math.sin(phi_ml)

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
                "phi_err": round(math.degrees(phi_err), 4),
                "xi_ml":     round(xi_ml, 6),
                "eta_ml":    round(eta_ml, 6),
                "phi_ml":  round(math.degrees(phi_ml), 4),
                "N":         N,
                "sqrt_N":    round(sqrt_N, 4),
            })

        hd_mae  = sum(abs(e) for e in hd_errors) / len(hd_errors) if hd_errors else float("nan")
        hd_rmse = math.sqrt(sum(e**2 for e in hd_errors) / len(hd_errors)) if hd_errors else float("nan")

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = (2 * prec * rec / (prec + rec)
                if (prec + rec) > 0 else float("nan"))
        n_skip = n_skip_missing + n_skip_N1 + n_skip_ml

        print(f"  scored {n_ok} for xi  |  {len(hd_errors)} for Hd MAE  |  "
              f"skipped {n_skip} (missing: {n_skip_missing}, N=1: {n_skip_N1}, ml_denom: {n_skip_ml})")
        print(f"  stability: tp={tp} fp={fp} fn={fn} tn={tn}  ->  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")

        summary[model] = {"scores": scores, "mae": mae, "rmse": rmse,
                          "hd_mae": hd_mae, "hd_rmse": hd_rmse,
                          "n": n_ok, "n_hd": len(hd_errors),
                          "f1": f1, "precision": prec, "recall": rec,
                          "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    year = "2020" if split.startswith("2020") else "2026"
    out_dir = os.path.join(REPO_ROOT, "results", year)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"interference_scores_{year}.csv")
    fieldnames = ["model", "compound", "stability", "Hf_DFT", "Hd_DFT", "Hd_err",
                  "xi_err",   "eta_err", "phi_err",
                  "xi_ml",    "eta_ml",  "phi_ml",
                  "N", "sqrt_N"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nPer-compound scores written to  {out_csv}")

    sum_csv = os.path.join(out_dir, f"interference_summary_{year}.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_xi", "n_hd", "Hf_MAE", "Hf_RMSE",
                    "Hd_MAE", "Hd_RMSE", "rms_xi", "median_xi",
                    "F1", "Precision", "Recall", "TP", "FP", "FN", "TN"])
        for model, s in summary.items():
            sc      = sorted(s["scores"])
            n       = len(sc)
            rms_xi  = math.sqrt(sum(x**2 for x in sc) / n) if n else float("nan")
            med_xi  = sc[n // 2] if n else float("nan")
            w.writerow([model, s["n"], s["n_hd"],
                        round(s["mae"],     4),
                        round(s["rmse"],    4),
                        round(s["hd_mae"],  4),
                        round(s["hd_rmse"], 4),
                        round(rms_xi,       4),
                        round(med_xi,       4),
                        round(s["f1"],        4),
                        round(s["precision"], 4),
                        round(s["recall"],    4),
                        s["tp"], s["fp"], s["fn"], s["tn"]])
    print(f"Summary written to              {sum_csv}")

    col = "{:<20}  {:>7}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}"
    sep = "-" * 96
    print(f"\n{sep}")
    print(f"  Interference Score Summary  ({split})")
    print("  lower xi = more error cancellation  |  Ed = derived decomposition enthalpy")
    print("  xi_rms = sqrt(<xi^2>); statistical expectation is 1 for i.i.d. residuals")
    print(sep)
    print(col.format("Model", "N", "Ef MAE", "Ef RMSE", "Ed MAE", "Ed RMSE", "xi_rms", "xi_median", "F1"))
    print(sep)
    for model, s in summary.items():
        sc     = sorted(s["scores"])
        n      = len(sc)
        rms_xi = math.sqrt(sum(x**2 for x in sc) / n) if n else float("nan")
        med_xi = sc[n // 2] if n else float("nan")
        print(col.format(
            model, s["n"],
            f"{s['mae']:.4f}",
            f"{s['rmse']:.4f}",
            f"{s['hd_mae']:.4f}",
            f"{s['hd_rmse']:.4f}",
            f"{rms_xi:.4f}",
            f"{med_xi:.4f}",
            f"{s['f1']:.4f}",
        ))
    print(sep)


if __name__ == "__main__":
    main()