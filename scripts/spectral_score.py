"""Revision metrics for machine-learned formation enthalpy predictions.

Usage
-----
    python scripts/spectral_score.py 2020
    python scripts/spectral_score.py 2025
    python scripts/spectral_score.py 2025 --mlhull my_model

Output, written to results/<year>/spectral_summary_<year>.csv, one row per run:

    MAE_f, MAE_d, xi_rms            as interference_score.py computes them
    frac_xi_lt1                     share of N >= 2 reactions with xi < 1
    Xi2, Xi2_Q, p_gauge             pooled index without and with the gauge fixed,
                                    and the gauge share of the residual
    F1, precision, recall, MCC      at the dH_d = 0 threshold
    maxF1, ROC_AUC, PR_AUC          threshold-free, compounds ranked by predicted dH_d
    frac_pred_stable                share of compounds predicted stable

and, for the four nested runs at base width 1024 and any run stem passed to
--mlhull, the hull rebuilt from the model's own out-of-fold formation
enthalpies (Bartel et al. 2020): every compound is classified by its distance
to that hull, against the DFT label,

    mlhull_F1, mlhull_precision, mlhull_recall, mlhull_DAF, mlhull_MAE_hd

Before any model is scored the LP is run on the DFT energies and must reproduce
the stored hull distances; it stops otherwise.
"""

import argparse
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from spectral import HULLOUT_MAP, HullSolver, load_hull, score_run

DEFAULT_MLHULL = ["iiLoss_finetune_0.0", "iiLoss_nested", "iiLossRaw_nested",
                  "HdLoss_nested"]

# Set in the parent before each pool is forked, so the workers inherit them
# instead of receiving the energy vector with every task.
_SOLVER, _ENERGIES = None, None


def _distance(i):
    return _SOLVER.distance(_ENERGIES, i)


def _stem_of(name):
    """Drop a trailing _s<seed> so all seeds of one run pool together."""
    head, _, tail = name.rpartition("_s")
    return head if head and tail.isdigit() else name


class MLHull:
    """Hull distances from a set of formation enthalpies, in parallel."""

    def __init__(self, H, procs):
        global _SOLVER
        self.H = H
        self.procs = procs
        _SOLVER = HullSolver(H["comps"], H["elements"])
        self.nE = _SOLVER.nE
        self.truth_hd = np.full(H["ncomp"], np.nan)
        self.truth_hd[H["rc"]] = H["hd"]
        self.ok = np.isfinite(self.truth_hd)

    def check(self):
        """The LP on the DFT energies must reproduce the stored distances."""
        hd = self._run(self.H["hf"])
        worst = np.nanmax(np.abs(hd[self.ok] - self.truth_hd[self.ok]))
        if not worst < 1e-6:
            raise SystemExit(f"hull rebuild does not reproduce the DFT hull ({worst:.1e} eV/atom)")
        print(f"  ML hull: the LP reproduces the DFT hull distances (max |diff| {worst:.1e})")

    def _run(self, hf):
        global _ENERGIES
        _ENERGIES = np.r_[hf, np.zeros(self.nE)]
        with Pool(self.procs) as pool:
            return np.array(pool.map(_distance, range(len(hf)), chunksize=500))

    def score(self, pred):
        hf = np.array([pred.get(f, np.nan) for f in self.H["comps"]], dtype=float)
        if not np.all(np.isfinite(hf)):
            raise SystemExit("the ML hull needs a prediction for every compound")
        hd = self._run(hf)[self.ok]
        truth = self.truth_hd[self.ok] <= 0
        pos = hd <= 0
        tp = (pos & truth).sum()
        fp = (pos & ~truth).sum()
        fn = (~pos & truth).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return dict(mlhull_F1=2 * prec * rec / max(prec + rec, 1e-12),
                    mlhull_precision=prec, mlhull_recall=rec,
                    mlhull_DAF=prec / truth.mean(),
                    mlhull_MAE_hd=float(np.nanmean(np.abs(hd - self.truth_hd[self.ok]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("year", choices=["2020", "2025", "2026"])
    parser.add_argument("--mlhull", nargs="*", default=[],
                        help="run stems (no _s<seed>) to rebuild the hull for, "
                             "in addition to the four nested runs (2025)")
    parser.add_argument("--procs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    ml_dir = os.path.join(REPO_ROOT, "data", args.year, "ml", "Hf")
    runs = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(ml_dir, "*", "ml_input.json")))
    mlhull = set(args.mlhull) | (set(DEFAULT_MLHULL) if args.year == "2025" else set())

    print(f"Loading {HULLOUT_MAP[args.year]}")
    H = load_hull(HULLOUT_MAP[args.year])
    print(f"  {H['ncomp']} compounds, {H['R']} reactions, "
          f"{'per-atom' if H['per_atom'] else 'formula-unit (rescaled)'} coefficients")

    hull = None
    if any(_stem_of(r) in mlhull for r in runs):
        hull = MLHull(H, args.procs)
        hull.check()

    rows = []
    for run in runs:
        with open(os.path.join(ml_dir, run, "ml_input.json")) as f:
            pred = json.load(f)
        row = {"model": run, **score_run(H, pred)}
        if hull is not None and _stem_of(run) in mlhull:
            row.update(hull.score(pred))
        rows.append(row)
        print(f"  {run:36s} MAE_f {row['MAE_f']:.4f}  MAE_d {row['MAE_d']:.4f}  "
              f"xi_rms {row['xi_rms']:.4f}  Xi2_Q {row['Xi2_Q']:.4f}  F1 {row['F1']:.4f}"
              + (f"  ML-hull F1 {row['mlhull_F1']:.4f}" if "mlhull_F1" in row else ""),
              flush=True)

    fields = ["model"] + [k for k in rows[0] if k != "model"]
    for extra in ("mlhull_F1", "mlhull_precision", "mlhull_recall", "mlhull_DAF", "mlhull_MAE_hd"):
        if any(extra in r for r in rows) and extra not in fields:
            fields.append(extra)
    out_dir = os.path.join(REPO_ROOT, "results", args.year)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"spectral_summary_{args.year}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()
