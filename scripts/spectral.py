"""Shared machinery for the revision's scoring scripts.

A hullout file is turned into its structure matrix S, one row per decomposition
reaction, with -1 on the reactant and per-atom weights on the products, so that

    S dfH = -dH_d     and     (S delta)_j = dH_d,ML - dH_d,DFT

for a residual field delta = dfH,ML - dfH,DFT. Elemental phases are columns of S
pinned at delta = 0, as in interference_score.py. Everything here is numpy and
runs on a laptop.

What is computed from it
    xi_rms      paper 1's statistic, the unweighted mean of xi^2 over reactions
                with at least two non-elemental participants
    Xi2, Xi2_Q  the pooled index sum_j (S delta)_j^2 / delta^T D delta, without
                and with the composition-linear (gauge) direction removed from
                the denominator; D = diag(sum_j S_jc^2)
    p_gauge     the share of delta^T D delta that lies in that direction
    MAE_f, MAE_d, F1, precision, recall, MCC at the dH_d = 0 threshold, and the
                threshold-free max-F1, ROC-AUC and PR-AUC
    ML hull     the hull rebuilt from the model's own formation enthalpies, by LP,
                and each compound classified by its distance to it

The 2020 hullout stores formula-unit amounts; they are rescaled to per-atom
weights by N_k / N_c, detected from the data as interference_score.py does.
"""

import itertools
import json
import os
import re

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

DATA_ROOT = os.path.join(REPO_ROOT, "data")

HULLOUT_MAP = {
    "2020": os.path.join(DATA_ROOT, "2020", "hullout_2020.json"),
    "2026": os.path.join(DATA_ROOT, "2026", "hullout_2026.json"),
    "2025": os.path.join(DATA_ROOT, "2025", "hullout_2025.json"),
    "wbm":  os.path.join(DATA_ROOT, "wbm", "hullout_wbm.json"),
}


# -- Formula helpers ------------------------------------------------------

def parse_formula(formula):
    """Atomic fractions of a formula, brackets and fractional amounts allowed."""
    def group(s, i):
        out = {}
        while i < len(s) and s[i] != ")":
            if s[i] == "(":
                sub, i = group(s, i + 1)
                i += 1
                m = re.match(r"[\d.]*", s[i:])
                n = float(m.group()) if m.group() else 1.0
                i += len(m.group())
                for k, v in sub.items():
                    out[k] = out.get(k, 0) + v * n
            else:
                m = re.match(r"([A-Z][a-z]?)([\d.]*)", s[i:])
                if not m:
                    raise ValueError(f"cannot parse {formula!r} at {s[i:]!r}")
                out[m.group(1)] = (out.get(m.group(1), 0)
                                   + (float(m.group(2)) if m.group(2) else 1.0))
                i += len(m.group())
        return out, i
    counts, _ = group(formula.replace(" ", ""), 0)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def is_element(formula):
    """True only for single-element formulas (Fe, Li, O, Fe2, S8, ...)"""
    return len(set(re.findall(r"[A-Z][a-z]?", formula))) == 1


def num_atoms(formula):
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    return sum(int(n) if n else 1 for el, n in tokens if el)


# -- The structure matrix -------------------------------------------------

def load_hull(path):
    """Structure matrix of a hullout file in COO form, plus what goes with it.

    Returns a dict with
        comps, idx      non-elemental compounds and their column index
        rows, cols, vals
                        the non-zeros of S; element columns follow the compounds
        R, M, ncomp     reactions, columns, compounds
        rc, hd          reactant column and stored dH_d of every reaction
        hf              stored dfH of every compound
        X               composition matrix (atomic fractions), M x n_elements
        per_atom        False if the coefficients had to be rescaled (2020 files)
    """
    with open(path) as f:
        hull = json.load(f)
    comps = [f for f, v in hull.items()
             if isinstance(v, dict) and "Hf" in v and not is_element(f)]
    idx = {f: i for i, f in enumerate(comps)}

    parsed = []
    for f in comps:
        entry = hull[f]
        if "rxn" not in entry:
            continue
        parts, ok = [], True
        for token in entry["rxn"].split(" + "):
            token = token.strip()
            amt, prod = token.split("_", 1) if "_" in token else ("1", token)
            if not is_element(prod) and prod not in idx:
                ok = False
                break
            parts.append((prod, float(amt)))
        if ok:
            parsed.append((f, parts))

    # Coefficient convention, from the atom balance of the first reactions.
    def balance(f, parts, rescale):
        comp = parse_formula(f)
        tot = {k: -v for k, v in comp.items()}
        for prod, amt in parts:
            w = amt * num_atoms(prod) / num_atoms(f) if rescale else amt
            for k, v in parse_formula(prod).items():
                tot[k] = tot.get(k, 0) + w * v
        return max(abs(v) for v in tot.values())
    probe = parsed[:200]
    per_atom = (np.median([balance(f, p, False) for f, p in probe])
                <= np.median([balance(f, p, True) for f, p in probe]))

    elements = sorted({el for f in comps for el in parse_formula(f)}
                      | {re.findall(r"[A-Z][a-z]?", p)[0]
                         for _, parts in parsed for p, _ in parts if is_element(p)})
    eidx = {el: len(comps) + j for j, el in enumerate(elements)}

    rows, cols, vals, rc, hd = [], [], [], [], []
    for f, parts in parsed:
        r = len(rc)
        rows.append(r); cols.append(idx[f]); vals.append(-1.0)
        for prod, amt in parts:
            w = amt if per_atom else amt * num_atoms(prod) / num_atoms(f)
            col = (eidx[re.findall(r"[A-Z][a-z]?", prod)[0]] if is_element(prod)
                   else idx[prod])
            rows.append(r); cols.append(col); vals.append(w)
        rc.append(idx[f]); hd.append(hull[f]["Hd"])

    M = len(comps) + len(elements)
    X = np.zeros((M, len(elements)))
    for f, i in idx.items():
        for el, x in parse_formula(f).items():
            X[i, eidx[el] - len(comps)] = x
    X[len(comps) + np.arange(len(elements)), np.arange(len(elements))] = 1.0

    return dict(comps=comps, idx=idx, rows=np.array(rows), cols=np.array(cols),
                vals=np.array(vals), R=len(rc), M=M, ncomp=len(comps),
                rc=np.array(rc), hd=np.array(hd), X=X, elements=elements,
                hf=np.array([hull[f]["Hf"] for f in comps]), per_atom=bool(per_atom),
                material_id=[hull[f].get("material_id") for f in comps])


def residual(H, pred):
    """delta over all columns (elements 0) and the mask of columns that have one."""
    nc = H["ncomp"]
    d = np.zeros(H["M"])
    have = np.zeros(H["M"], bool)
    have[nc:] = True
    for f, v in pred.items():
        i = H["idx"].get(f)
        if i is not None and v is not None and np.isfinite(v):
            d[i] = v - H["hf"][i]
            have[i] = True
    return d, have


def reaction_sums(H, d):
    """(S delta)_j and sum_k (S_jk delta_k)^2 for every reaction."""
    rows, cols, vals = H["rows"], H["cols"], H["vals"]
    y = np.zeros(H["R"]); np.add.at(y, rows, vals * d[cols])
    q = np.zeros(H["R"]); np.add.at(q, rows, (vals * d[cols]) ** 2)
    return y, q


def free_sizes(H):
    """Non-elemental participants per reaction, reactant included."""
    n = np.zeros(H["R"], int)
    np.add.at(n, H["rows"], (H["cols"] < H["ncomp"]).astype(int))
    return n


def complete_rows(H, have):
    """Reactions whose every participant has a residual."""
    bad = np.zeros(H["R"], bool)
    np.logical_or.at(bad, H["rows"], ~have[H["cols"]])
    return ~bad


# -- The gauge ------------------------------------------------------------

def gauge_split(H, d, rows_ok, rcond=1e-10):
    """D-orthogonal split of delta into its composition-linear part and the rest.

    D = diag(sum_j S_jc^2) over the reactions in `rows_ok`; the composition
    matrix carries the elements as columns, so the projector acts on element
    columns too and every balanced reaction sum is left unchanged.
    Returns (deg, delta_Q) with delta_Q = delta - X mu.
    """
    keep = rows_ok[H["rows"]]
    deg = np.zeros(H["M"])
    np.add.at(deg, H["cols"][keep], H["vals"][keep] ** 2)
    X = H["X"]
    G = X.T @ (deg[:, None] * X)
    w, V = np.linalg.eigh(G)
    inv = np.where(w > rcond * max(w.max(), 1), 1 / np.where(w > 0, w, 1), 0)
    mu = (V * inv) @ V.T @ (X.T @ (deg * d))
    return deg, d - X @ mu


# -- Scores ---------------------------------------------------------------

def classification(pred_stable, truth):
    tp = int((pred_stable & truth).sum()); fp = int((pred_stable & ~truth).sum())
    fn = int((~pred_stable & truth).sum()); tn = int((~pred_stable & ~truth).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    # in floats: the product of four counts overflows a 64-bit integer
    mcc = (float(tp) * tn - float(fp) * fn) / np.sqrt(
        max(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    return dict(F1=f1, precision=prec, recall=rec, MCC=mcc, TP=tp, FP=fp, FN=fn, TN=tn)


def threshold_free(score, truth):
    """max-F1, ROC-AUC and PR-AUC, ranking compounds by predicted dH_d."""
    o = np.argsort(score)
    t = truth[o]
    tp = np.cumsum(t); fp = np.cumsum(~t); npos = t.sum()
    prec = tp / (tp + fp); rec = tp / npos
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    roc = trapz(np.r_[0, tp / npos], np.r_[0, fp / (~t).sum()])
    return dict(maxF1=float(f1.max()), ROC_AUC=float(roc),
                PR_AUC=float(np.sum(prec[t]) / npos))


def score_run(H, pred):
    """Every revision metric for one prediction set {formula: dfH}."""
    nc = H["ncomp"]
    d, have = residual(H, pred)
    ok = complete_rows(H, have)
    y, q = reaction_sums(H, d)
    nfree = free_sizes(H)
    mxi = ok & (nfree >= 2) & (q > 0)

    deg, dq = gauge_split(H, d, ok)
    num = float((y[ok] ** 2).sum())
    den_raw = float((deg * d * d).sum())
    den_q = float((deg * dq * dq).sum())

    hd = H["hd"][ok]
    p = hd - y[ok]
    truth = hd <= 0
    out = dict(MAE_f=float(np.abs(d[:nc][have[:nc]]).mean()),
               MAE_d=float(np.abs(y[ok]).mean()),
               xi_rms=float(np.sqrt(np.mean(y[mxi] ** 2 / q[mxi]))),
               frac_xi_lt1=float(np.mean(y[mxi] ** 2 / q[mxi] < 1)),
               Xi2=num / den_raw, Xi2_Q=num / den_q, p_gauge=1 - den_q / den_raw)
    out.update(classification(p <= 0, truth))
    out.update(threshold_free(p, truth))
    out.update(frac_pred_stable=float((p <= 0).mean()),
               frac_true_stable=float(truth.mean()),
               n_reactions=int(ok.sum()), n_xi=int(mxi.sum()))
    return out


# -- The hull rebuilt from predictions ------------------------------------

class HullSolver:
    """Distance of every compound to the hull of all other phases, by LP.

    Candidates for a compound are the phases whose elements are a subset of its
    own, elements included at dfH = 0; the decomposition energy is the LP
    optimum of sum_k x_k E_k subject to the compound's atomic fractions, so
    dH_d = E_c - optimum, negative for stable compounds (Bartel convention).

    Parameters
    ----------
    formulas : list of str
        The compounds, in column order.
    elements : list of str or None
        Element order; defaults to the sorted elements of `formulas`.
    """

    def __init__(self, formulas, elements=None):
        fracs = [parse_formula(f) for f in formulas]
        elements = elements or sorted({e for c in fracs for e in c})
        self.elements = list(elements)
        self.nC, self.nE = len(formulas), len(elements)
        eidx = {e: i for i, e in enumerate(elements)}
        X = np.zeros((self.nC + self.nE, self.nE))
        for i, c in enumerate(fracs):
            for e, v in c.items():
                X[i, eidx[e]] = v
        X[self.nC + np.arange(self.nE), np.arange(self.nE)] = 1.0
        self.X = X
        self.esets = [frozenset(np.flatnonzero(X[i])) for i in range(len(X))]
        self._by_set = {}
        for i, s in enumerate(self.esets):
            self._by_set.setdefault(s, []).append(i)

    def candidates(self, i):
        s, c = sorted(self.esets[i]), []
        for r in range(1, len(s) + 1):
            for sub in itertools.combinations(s, r):
                c += self._by_set.get(frozenset(sub), [])
        return np.array([j for j in c if j != i])

    def decompose(self, energies, i):
        """(products, per-atom weights, dH_d) of compound i, or Nones."""
        c, s = self.candidates(i), sorted(self.esets[i])
        res = linprog(energies[c], A_eq=self.X[c][:, s].T, b_eq=self.X[i, s],
                      bounds=(0, None), method="highs")
        if res.status != 0:
            return None, None, np.nan
        keep = res.x > 1e-9
        return c[keep], res.x[keep], energies[i] - res.fun

    def distance(self, energies, i):
        return self.decompose(energies, i)[2]
