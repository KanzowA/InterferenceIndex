"""
perovskite_subset.py
--------------------
Evaluates model performance on the ABO₃ perovskite subset of the MP dataset.

Identifies ABO₃ perovskites from the hullout file by formula, then computes
Ef MAE, Ed MAE, and interference score ξ on that subset for every model
found in ml_data/Ef/<split>/.

Usage (from the TestStabilityMl directory):
    python perovskite_subset.py
    python perovskite_subset.py allMP_current --hullout hullout_current.json

Output:
    perovskite_summary.csv  — per-model metrics on the ABO₃ subset
"""

import os, sys, re, json, math, csv, argparse
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(HERE, "mlstabilitytest")
DATA_DIR = os.path.join(REPO_DIR, "mp_data", "data")
HULLOUT  = os.path.join(DATA_DIR, "hullout.json")


# ── Formula helpers ────────────────────────────────────────────────────────────

def parse_formula(formula: str) -> dict:
    """Return {element: count} for a formula string."""
    tokens = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', formula)
    comp = {}
    for el, n in tokens:
        if not el:
            continue
        count = float(n) if n else 1.0
        comp[el] = comp.get(el, 0) + count
    return comp


def is_perovskite_ABO3(formula: str) -> bool:
    """
    True if the formula is an ABO₃ perovskite (or a rational multiple thereof).

    Criteria:
      - Exactly 3 distinct elements, one of which is O
      - O count = 3 × count of each non-O element (within floating-point tolerance)
      - e.g. BaTiO3, LaAlO3, Ba2Ti2O6, SrVO3 → True
             La2NiO4, TiO2, BaO → False
    """
    comp = parse_formula(formula)
    if 'O' not in comp or len(comp) != 3:
        return False
    o_count = comp['O']
    non_o   = {el: v for el, v in comp.items() if el != 'O'}
    if len(non_o) != 2:
        return False
    a, b = non_o.values()
    # A:B:O should be n:n:3n
    if abs(a - b) > 1e-3:
        return False
    ratio = o_count / a
    return abs(ratio - 3.0) < 1e-3


# ── Score helpers ──────────────────────────────────────────────────────────────

def parse_rxn(rxn_str: str):
    products = []
    for token in rxn_str.split(" + "):
        token = token.strip()
        if "_" not in token:
            products.append((1.0, token))
        else:
            amt_str, formula = token.split("_", 1)
            products.append((float(amt_str), formula))
    return products


def is_element(formula: str) -> bool:
    symbols = re.findall(r'[A-Z][a-z]?', formula)
    return len(set(symbols)) == 1


def num_atoms(formula: str) -> int:
    counts = re.findall(r'[A-Z][a-z]?(\d+)', formula)
    return sum(int(n) for n in counts) if counts else 1


def interference_score(compound, rxn_str, ml_ef, dft_ef):
    if compound not in ml_ef or compound not in dft_ef:
        return None
    products = parse_rxn(rxn_str)
    N_c = num_atoms(compound)
    c   = []

    if not is_element(compound):
        c.append(-(ml_ef[compound] - dft_ef[compound]))

    for amt_k, fk in products:
        if is_element(fk):
            continue
        if fk not in ml_ef or fk not in dft_ef:
            return None
        N_k = num_atoms(fk)
        c.append(amt_k * N_k * (ml_ef[fk] - dft_ef[fk]) / N_c)

    if not c:
        return None
    denom = math.sqrt(sum(x**2 for x in c))
    if denom < 1e-12:
        return None
    return abs(sum(c)) / denom


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", nargs="?", default="allMP")
    parser.add_argument("--hullout", default=None)
    args = parser.parse_args()

    hullout_path = HULLOUT
    if args.hullout:
        candidate = args.hullout
        if not os.path.isabs(candidate):
            candidate = os.path.join(HERE, candidate)
        hullout_path = candidate

    ML_DIR = os.path.join(REPO_DIR, "ml_data", "Ef", args.split)
    print(f"Split   : {args.split}")
    print(f"Hullout : {hullout_path}")
    print(f"ML dir  : {ML_DIR}\n")

    # ── Load hullout & identify perovskites ────────────────────────────────────
    hullout = load_json(hullout_path)

    perovskites = {
        f for f, v in hullout.items()
        if isinstance(v, dict) and is_perovskite_ABO3(f)
    }
    perov_with_rxn = {
        f for f in perovskites
        if isinstance(hullout[f], dict)
        and "rxn"   in hullout[f]
        and "Ef"    in hullout[f]
        and not is_element(f)
    }

    dft_ef = {f: hullout[f]["Ef"] for f in perov_with_rxn}
    print(f"ABO₃ perovskites in hullout : {len(perovskites)}")
    print(f"  with reaction + Ef        : {len(perov_with_rxn)}\n")

    if not perov_with_rxn:
        print("No perovskites found — check formula parsing or hullout file.")
        sys.exit(1)

    # ── Auto-detect models ─────────────────────────────────────────────────────
    all_models = []
    if os.path.isdir(ML_DIR):
        for name in sorted(os.listdir(ML_DIR)):
            if os.path.isfile(os.path.join(ML_DIR, name, "ml_input.json")):
                all_models.append(name)

    if not all_models:
        print(f"No ml_input.json files found under {ML_DIR}")
        sys.exit(1)

    print(f"Models found: {all_models}\n")

    # ── Compute metrics ────────────────────────────────────────────────────────
    rows = []
    header = ["model", "n_perov", "Ef_MAE", "Ef_RMSE",
              "Ed_MAE", "Ed_RMSE", "mean_xi", "median_xi"]

    for model in all_models:
        ml_path = os.path.join(ML_DIR, model, "ml_input.json")
        ml_ef   = load_json(ml_path)

        # Ef MAE / RMSE on perovskite subset
        ef_errors = []
        for f in perov_with_rxn:
            if f in ml_ef and f in dft_ef:
                ef_errors.append(ml_ef[f] - dft_ef[f])

        if not ef_errors:
            print(f"  [SKIP] {model}: no predictions for perovskite subset")
            continue

        ef_mae  = sum(abs(e) for e in ef_errors) / len(ef_errors)
        ef_rmse = math.sqrt(sum(e**2 for e in ef_errors) / len(ef_errors))

        # Ed MAE / RMSE — reconstruct Ed from hullout reactions
        ed_errors = []
        xis       = []
        for f in perov_with_rxn:
            if f not in ml_ef:
                continue
            rxn_str = hullout[f].get("rxn", "")
            if not rxn_str:
                continue

            # Ed_ML = sum of weighted ML Ef values in the reaction
            products  = parse_rxn(rxn_str)
            N_c       = num_atoms(f)

            # Ed = Ef(compound) - sum_k(amt_k * N_k/N_c * Ef(phase_k))
            ed_dft = hullout[f].get("Ed")
            if ed_dft is None:
                continue

            # ML Ed prediction
            if f not in ml_ef:
                continue
            ed_ml_num = ml_ef[f]
            ed_ml_den = 0.0
            ok        = True
            for amt_k, fk in products:
                if fk not in ml_ef:
                    ok = False; break
                N_k      = num_atoms(fk)
                ed_ml_den += amt_k * N_k / N_c * ml_ef[fk]
            if not ok:
                continue
            ed_ml = ed_ml_num - ed_ml_den
            ed_errors.append(ed_ml - ed_dft)

            # ξ
            xi = interference_score(f, rxn_str, ml_ef, dft_ef)
            if xi is not None:
                xis.append(xi)

        n = len(xis)
        if not ed_errors or not xis:
            print(f"  [SKIP] {model}: insufficient Ed/ξ data")
            continue

        ed_mae   = sum(abs(e) for e in ed_errors) / len(ed_errors)
        ed_rmse  = math.sqrt(sum(e**2 for e in ed_errors) / len(ed_errors))
        mean_xi  = sum(xis) / n
        xis_s    = sorted(xis)
        med_xi   = xis_s[n // 2] if n % 2 else (xis_s[n//2-1] + xis_s[n//2]) / 2

        rows.append([model, n, ef_mae, ef_rmse, ed_mae, ed_rmse, mean_xi, med_xi])
        print(f"  {model:<30s}  n={n:5d}  "
              f"Ef={ef_mae:.4f}  Ed={ed_mae:.4f}  "
              f"ξ_mean={mean_xi:.4f}  ξ_med={med_xi:.4f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_path = os.path.join(HERE, "perovskite_summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([row[0], row[1]] + [f"{v:.4f}" for v in row[2:]])

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
