"""The interference diagnostic for the Matbench Discovery models, on a closed hull.

Usage
-----
    python scripts/mlip_score.py                 # predictions in data/wbm/mbd/

The published predictions of each model (data/wbm/mbd_manifest.csv lists them,
with their figshare links) are scored on data/wbm/hullout_wbm.json, in which
every reaction participant is a WBM entry predicted by the same model, so errors
can cancel. Elements are fixed at 0.

One common evaluation set per comparison, never a model's own subset:

    compounds   predicted (finite, within 5 eV/atom of DFT) by every model of
                the group
    reactions   those whose participants are all such compounds; D, the gauge
                projector and every metric are built on this sub-hull, and no
                missing value is filled in

The 57 machine-learned interatomic potentials (task IS2RE-SR) are scored on
their own common set; the 10 earlier energy models, which solve other tasks,
on the common set of all 67, and are never pooled with the potentials.
95 % intervals of Xi2_Q: 200 bootstrap resamples of chemical systems (the
element set of the reactant), the same resamples for every model.

Output
    results/wbm/mlip_closed.csv   one row per model
    printed                       group medians and intervals, the shift in
                                  MAE(dH_d) at equal MAE(dfH) between the two
                                  training-data groups, rank inversions
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from spectral import (HULLOUT_MAP, complete_rows, free_sizes, gauge_split,
                      load_hull, reaction_sums)

MANIFEST = os.path.join(REPO_ROOT, "data", "wbm", "mbd_manifest.csv")
PRED_DIR = os.path.join(REPO_ROOT, "data", "wbm", "mbd")
OUT = os.path.join(REPO_ROOT, "results", "wbm", "mlip_closed.csv")

PRE_MLIP = {"MP 2022", "MP Graphs"}
MP_ONLY = {"MPtrj", "MPF", "MPtrj+MDR-MP PBE ω_q"}
N_BOOT = 200


def group_of(row):
    if row.training_sets in PRE_MLIP:
        return "pre-MLIP"
    return "MP-only MLIP" if row.training_sets in MP_ONLY else "extended-data MLIP"


def residuals(H, keys):
    """delta per model (compounds, NaN where unusable; elements 0)."""
    ids = pd.Index(H["material_id"])
    out = {}
    for key in keys:
        path = os.path.join(PRED_DIR, f"{key}.csv.gz")
        p = pd.read_csv(path).set_index("material_id")["e_form_per_atom"]
        p = p[~p.index.duplicated()]
        v = p.reindex(ids).to_numpy(float) - H["hf"]
        v[np.abs(v) > 5] = np.nan                      # Matbench Discovery's filter
        d = np.zeros(H["M"])
        d[:H["ncomp"]] = v
        out[key] = d
    return out


def score_common(H, deltas, manifest):
    """Every model of `deltas` on the strictly common sub-hull."""
    nc = H["ncomp"]
    have = np.ones(H["M"], bool)
    for d in deltas.values():
        have &= np.isfinite(d)
    rows = complete_rows(H, have)
    in_rows = np.zeros(H["M"], bool)
    in_rows[H["cols"][rows[H["rows"]]]] = True
    comps = in_rows.copy()
    comps[nc:] = False
    nfree = free_sizes(H)
    hd = H["hd"][rows]

    # Chemical systems of the reactants, for the bootstrap.
    chem = np.array(["-".join(map(str, np.flatnonzero(H["X"][i]))) for i in H["rc"][rows]])
    _, inv = np.unique(chem, return_inverse=True)
    rng = np.random.default_rng(0)
    n_sys = inv.max() + 1
    weights = [np.bincount(rng.integers(0, n_sys, n_sys), minlength=n_sys)[inv]
               for _ in range(N_BOOT)]

    res = []
    for key, d in deltas.items():
        d = np.where(have, d, 0.0)
        y, q = reaction_sums(H, d)
        y, q = y[rows], q[rows]
        m2 = (nfree[rows] >= 2) & (q > 0)
        deg, dq = gauge_split(H, d, rows)
        _, b = reaction_sums(H, dq)
        b = b[rows]
        boot = [(w * y ** 2).sum() / (w * b).sum() for w in weights]
        truth = hd <= 0
        pos = (hd - y) <= 0
        tp = (pos & truth).sum()
        fp = (pos & ~truth).sum()
        fn = (~pos & truth).sum()
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        res.append(dict(
            model=key, training_sets=manifest.loc[key, "training_sets"],
            test_task=manifest.loc[key, "test_task"], group=group_of(manifest.loc[key]),
            MAE_f=np.abs(d[comps]).mean(), MAE_d_closed=np.abs(y).mean(),
            xi_rms=np.sqrt(np.mean(y[m2] ** 2 / q[m2])),
            Xi2=(y ** 2).sum() / q.sum(), Xi2_Q=(y ** 2).sum() / b.sum(),
            Xi2_Q_lo=np.percentile(boot, 2.5), Xi2_Q_hi=np.percentile(boot, 97.5),
            p_gauge=1 - (deg * dq * dq).sum() / (deg * d * d).sum(),
            F1_closed=2 * prec * rec / (prec + rec),
            n_compounds=int(comps.sum()), n_reactions=int(rows.sum())))
    print(f"  {len(deltas)} models: {int(comps.sum())} common compounds, "
          f"{int(rows.sum())} of {H['R']} reactions")
    return res


def report(d):
    ml = d[d.group != "pre-MLIP"].copy()
    print("\nGauge-fixed index by group (median, models with the 95 % interval below / above 1)")
    for g, sub in d.groupby("group"):
        print(f"  {g:20s} n={len(sub):2d}  Xi2_Q {sub.Xi2_Q.median():.3f}  "
              f"below 1: {(sub.Xi2_Q_hi < 1).sum():2d}  above 1: {(sub.Xi2_Q_lo > 1).sum():2d}  "
              f"MAE_d/MAE_f {(sub.MAE_d_closed / sub.MAE_f).median():.2f}")

    # Does the training data shift MAE(dH_d) at equal MAE(dfH)?
    X = np.c_[np.ones(len(ml)), np.log(ml.MAE_f), (ml.group == "extended-data MLIP").astype(float)]
    yv = np.log(ml.MAE_d_closed)
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    s2 = resid @ resid / (len(ml) - 3)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(abs(coef[2] / se[2]), len(ml) - 3)
    print(f"\nlog MAE_d = {coef[0]:.3f} + {coef[1]:.3f} log MAE_f "
          f"+ {coef[2]:+.4f} [extended data]")
    print(f"  at equal MAE(dfH), extended-data potentials have "
          f"{100 * (np.exp(coef[2]) - 1):+.1f} % MAE(dH_d)  "
          f"(t = {coef[2] / se[2]:.2f}, p = {p:.3f})")

    a, b = ml.MAE_f.to_numpy(), ml.MAE_d_closed.to_numpy()
    n = len(a)
    flips = sum((a[i] - a[j]) * (b[i] - b[j]) < 0 for i in range(n) for j in range(i + 1, n))
    print(f"  pairs ranked differently by MAE(dfH) and MAE(dH_d): {flips} of {n * (n - 1) // 2}")
    r_all = stats.pearsonr(np.log(ml.Xi2_Q), np.log(ml.MAE_d_closed / ml.MAE_f))[0]
    within = {g: stats.pearsonr(np.log(s.Xi2_Q), np.log(s.MAE_d_closed / s.MAE_f))[0]
              for g, s in ml.groupby("group")}
    print(f"  Pearson r(log Xi2_Q, log MAE_d/MAE_f): all potentials {r_all:.2f}; "
          + "; ".join(f"{g} {r:.2f}" for g, r in within.items()))


def main():
    manifest = pd.read_csv(MANIFEST).set_index("model_key")
    keys = [k for k in manifest.index if os.path.exists(os.path.join(PRED_DIR, f"{k}.csv.gz"))]
    missing = [k for k in manifest.index if k not in keys]
    print(f"{len(keys)} prediction files in {PRED_DIR}"
          + (f"; not available: {', '.join(missing)}" if missing else ""))

    H = load_hull(HULLOUT_MAP["wbm"])
    deltas = residuals(H, keys)
    mlips = [k for k in keys if manifest.loc[k, "test_task"] == "IS2RE-SR"
             and manifest.loc[k, "training_sets"] not in PRE_MLIP]
    earlier = [k for k in keys if manifest.loc[k, "training_sets"] in PRE_MLIP]

    print("Potentials, on their common set")
    rows = score_common(H, {k: deltas[k] for k in mlips}, manifest)
    print("All models, on the common set of all of them (rows kept: earlier energy models)")
    rows += [r for r in score_common(H, deltas, manifest) if r["model"] in earlier]

    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False, float_format="%.6g")
    print(f"Written to {OUT}")
    report(d)


if __name__ == "__main__":
    main()
