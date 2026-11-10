"""Score WBM predictions two ways: the Matbench Discovery protocol and a closed hull.

Usage
-----
    python scripts/prospective_score.py                     # data/wbm/ml/*.csv.gz
    python scripts/prospective_score.py --check-leaderboard data/wbm/mbd

Matbench Discovery protocol (mixed)
    The candidate is screened against the DFT hull of MP: predicted hull distance
    = DFT e_above_hull + (predicted - DFT formation energy), stable if <= 0, so a
    candidate's error is its own residual and nothing can cancel. Implemented
    line by line from matbench_discovery/metrics/discovery.py: predictions more
    than 5 eV/atom off are masked, predictions and references rounded to 3
    decimals, missing predictions count as unstable, and the DAF of the unique
    prototypes uses their unrounded prevalence. --check-leaderboard rescores the
    published model predictions and compares with the leaderboard.

Closed hull
    data/wbm/hullout_wbm.json: every reaction participant is a WBM entry predicted
    by the same model, so decomposition errors can cancel. Scored with
    spectral.score_run.

Robustness: compositions outside MP
    17.8 % of WBM entries share a reduced formula (as another polymorph) with an
    MP training compound. The "novel" columns repeat the mixed protocol on the
    entries whose formula is not in hullout_2025.json, and the closed MAE(dH_d)
    on the reactions whose every non-elemental participant has such a formula.

Output: results/wbm/prospective_summary.csv, one row per prediction file.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from spectral import HULLOUT_MAP, load_hull, reaction_sums, residual, score_run

WBM_SUMMARY = os.path.join(REPO_ROOT, "data", "wbm", "2023-12-13-wbm-summary.csv.gz")
ML_DIR = os.path.join(REPO_ROOT, "data", "wbm", "ml")
OUT = os.path.join(REPO_ROOT, "results", "wbm", "prospective_summary.csv")


# -- Matbench Discovery metrics --------------------------------------------

def discovery_metrics(ref, pred_eform, subsets):
    """Leaderboard metrics of one prediction series, for each boolean subset."""
    from sklearn.metrics import r2_score
    cols = ["e_above_hull_mp2020_corrected_ppd_mp", "e_form_per_atom_mp2020_corrected"]
    p = pd.to_numeric(pred_eform.reindex(ref.index), errors="coerce")
    p = p.mask((p - ref.e_form_per_atom_mp2020_corrected).abs() > 5.0).round(3)
    r = ref[cols].round(3)
    each_true = r.e_above_hull_mp2020_corrected_ppd_mp
    each_pred = p.add(each_true).sub(r.e_form_per_atom_mp2020_corrected)
    uniq = ref.unique_prototype.astype(bool)
    prev_uniq = float((ref.e_above_hull_mp2020_corrected_ppd_mp[uniq] <= 0).mean())
    out = {}
    for name, m in subsets.items():
        t, q = each_true[m], each_pred[m]
        apos = t <= 0
        mpos = (q <= 0) & q.notna()
        tp = int((apos & mpos).sum()); fp = int((~apos & mpos).sum())
        fn = int((apos & ~mpos).sum()); tn = int((~apos & ~mpos).sum())
        prev = (tp + fn) / len(t)
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn)
        ok = q.notna()
        out.update({f"{name}_F1": 2 * prec * rec / (prec + rec),
                    f"{name}_DAF": prec / (prev_uniq if name == "uniq" else prev),
                    f"{name}_precision": prec, f"{name}_recall": rec,
                    f"{name}_accuracy": (tp + tn) / len(t),
                    f"{name}_MAE_hull": float((t[ok] - q[ok]).abs().mean()),
                    f"{name}_RMSE_hull": float(np.sqrt(((t[ok] - q[ok]) ** 2).mean())),
                    f"{name}_R2": float(r2_score(t[ok], q[ok]))})
    return out


def ranking_metrics(ref, pred_eform, subsets):
    """ROC-AUC and PR-AUC, entries ranked by the unrounded predicted hull distance.

    Not leaderboard metrics; added because a stability threshold at 0 hides how
    well the predictions rank. Missing predictions rank last.
    """
    p = pd.to_numeric(pred_eform.reindex(ref.index), errors="coerce")
    err = p - ref.e_form_per_atom_mp2020_corrected
    p = p.mask(err.abs() > 5.0)
    true = ref.e_above_hull_mp2020_corrected_ppd_mp
    pred = true + (p - ref.e_form_per_atom_mp2020_corrected)
    out = {}
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for name, m in subsets.items():
        tr, pr = true[m].to_numpy(), pred[m].to_numpy()
        score = np.where(np.isfinite(pr), -pr, -np.inf)
        t = (tr <= 0)[np.argsort(-score)]
        tp = np.cumsum(t); fp = np.cumsum(~t)
        out[f"{name}_ROC_AUC"] = float(trapz(np.r_[0, tp / t.sum()], np.r_[0, fp / (~t).sum()]))
        out[f"{name}_PR_AUC"] = float((tp / (tp + fp))[t].sum() / t.sum())
    return out


def load_predictions(path):
    p = pd.read_csv(path).set_index("material_id")["e_form_per_atom"]
    return p[~p.index.duplicated()]


def check_leaderboard(ref, subsets, mbd_dir):
    """Rescore the published predictions; compare with the leaderboard values."""
    manifest = pd.read_csv(os.path.join(REPO_ROOT, "data", "wbm", "mbd_manifest.csv"))
    worst = 0.0
    n = 0
    for _, row in manifest.iterrows():
        path = os.path.join(mbd_dir, f"{row.model_key}.csv.gz")
        if not os.path.exists(path):
            continue
        m = discovery_metrics(ref, load_predictions(path), {"full": subsets["full"]})
        for ours, theirs in (("full_F1", "F1"), ("full_DAF", "DAF"), ("full_MAE_hull", "MAE"),
                             ("full_RMSE_hull", "RMSE")):
            worst = max(worst, abs(round(m[ours], 3) - row[theirs]))
        n += 1
    print(f"Leaderboard check: {n} models, max |difference| {worst:.4f} "
          f"(published values carry 3 decimals)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="default: data/wbm/ml/*.csv.gz")
    parser.add_argument("--check-leaderboard", default=None, metavar="MBD_DIR",
                        help="directory of the published Matbench Discovery predictions")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    from pymatgen.core import Composition
    ref = pd.read_csv(WBM_SUMMARY).set_index("material_id")
    with open(HULLOUT_MAP["2025"]) as f:
        mp_formulas = set(json.load(f))
    reduced = ref.formula.map(lambda f: Composition(f).reduced_formula)
    novel = ~reduced.isin(mp_formulas).to_numpy()
    uniq = ref.unique_prototype.to_numpy(bool)
    subsets = {"full": np.ones(len(ref), bool), "uniq": uniq,
               "novel": novel, "uniq_novel": uniq & novel}
    print(f"WBM: {len(ref)} entries, {100 * (1 - novel.mean()):.1f} % share a formula with an "
          f"MP training compound ({100 * (1 - novel[uniq].mean()):.1f} % of unique prototypes)")

    if args.check_leaderboard:
        check_leaderboard(ref, subsets, args.check_leaderboard)
        if not args.files and not glob.glob(os.path.join(ML_DIR, "*.csv.gz")):
            return

    H = load_hull(HULLOUT_MAP["wbm"])
    id_of = dict(zip(H["comps"], H["material_id"]))
    col_novel = np.zeros(H["M"], bool)
    col_novel[:H["ncomp"]] = [f not in mp_formulas for f in H["comps"]]
    col_novel[H["ncomp"]:] = True                     # elements carry no residual
    bad = np.zeros(H["R"], bool)
    np.logical_or.at(bad, H["rows"], ~col_novel[H["cols"]])
    rows_novel = ~bad
    print(f"Closed WBM hull: {H['ncomp']} compounds, {H['R']} reactions, "
          f"{100 * rows_novel.mean():.1f} % with every participant outside MP")

    files = args.files or sorted(glob.glob(os.path.join(ML_DIR, "*.csv.gz")))
    rows = []
    for path in files:
        run = os.path.basename(path).replace(".csv.gz", "")
        p = load_predictions(path)
        err = (p.reindex(ref.index) - ref.e_form_per_atom_mp2020_corrected)
        err = err[err.abs() <= 5]
        row = {"model": run, "MAE_f_wbm": float(err.abs().mean())}
        row.update(discovery_metrics(ref, p, subsets))
        row.update(ranking_metrics(ref, p, {k: subsets[k] for k in ("full", "uniq")}))

        by_id = p.reindex(list(id_of.values()))
        by_id = by_id.mask((by_id - pd.Series(H["hf"], index=by_id.index)).abs() > 5)
        pred = {f: v for f, v in zip(H["comps"], by_id.to_numpy(float))}
        closed = score_run(H, pred)
        row.update({f"closed_{k}": closed[k] for k in
                    ("MAE_f", "MAE_d", "xi_rms", "Xi2_Q", "p_gauge", "F1", "precision", "recall")})
        d, have = residual(H, pred)
        y, _ = reaction_sums(H, d)
        row["closed_novel_MAE_d"] = float(np.abs(y[rows_novel]).mean())
        rows.append(row)
        print(f"  {run:30s} MAE_f {row['MAE_f_wbm']:.4f}  uniq F1 {row['uniq_F1']:.4f}  "
              f"uniq DAF {row['uniq_DAF']:.3f}  closed MAE_d {row['closed_MAE_d']:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.6g")
    print(f"Summary written to {args.out}")


if __name__ == "__main__":
    main()
