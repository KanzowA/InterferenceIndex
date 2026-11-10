"""Every number of the revised manuscript, next to its value in the first preprint.

Usage
-----
    python scripts/paper_numbers.py

Writes results/paper_numbers.md and results/paper_numbers.csv: one row per
quantitative statement, with where it stands in the preprint (chemRxiv
10.26434/chemrxiv.15008797/v1), its value there, its value in the revision, and
how the revision value is defined where that is not obvious. Nothing here is
typed in by hand except the preprint's values.

Reads what the other scripts write, and recomputes from the predictions only
what no summary holds (the gauge share in the preprint's definition, the
gauge-fixed xi distribution):

    results/2020/spectral_summary_2020.csv      spectral_score.py 2020
    results/2025/spectral_summary_2025.csv      spectral_score.py 2025
    results/2025/nested/                        nested_cv.py (selection logs)
    results/wbm/prospective_summary.csv         prospective_score.py
    results/wbm/mlip_closed.csv                 mlip_score.py
    data/2025/ml/Hf/                            the predictions

Values are means over seeds; a bracketed number is the sample standard
deviation in units of the last digit, as in the preprint.
"""

import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

from spectral import (HULLOUT_MAP, error_split, free_sizes, gauge_split, load_hull,
                      product_counts, reaction_sums, residual)

RESULTS = os.path.join(REPO_ROOT, "results")
ML_2025 = os.path.join(REPO_ROOT, "data", "2025", "ml", "Hf")
OUT = os.path.join(RESULTS, "paper_numbers")

ROWS = []


# -- Helpers --------------------------------------------------------------

def add(key, where, quantity, old, new, note=""):
    ROWS.append(dict(key=key, preprint_location=where, quantity=quantity,
                     preprint=old, revision=new, note=note))
    print(f"  {key:30s} {quantity[:58]:58s} {str(old):>12s} -> {new}")


def fmt(values, nd=3):
    v = np.asarray(values, float)
    if len(v) < 2:
        return f"{v.mean():.{nd}f}"
    return f"{v.mean():.{nd}f}({int(round(v.std(ddof=1) * 10 ** nd))})"


def pct(x):
    return f"{100 * float(np.mean(x)):.1f}%"


def seeds_of(df, stem):
    return df[df.stem == stem].sort_values("seed")


def rel(df, run, control, col):
    """Mean relative change in %, paired by seed."""
    a = seeds_of(df, run).set_index("seed")[col]
    c = seeds_of(df, control).set_index("seed")[col]
    common = a.index.intersection(c.index)
    return float((100 * (a[common] / c[common] - 1)).mean())


def load_logs(pattern):
    """Selection logs of the nested runs."""
    logs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "2025", "nested", pattern))):
        with open(path) as f:
            logs.append(json.load(f))
    return logs


def split_seed(df):
    parts = df["model"].str.extract(r"^(.*)_s(\d+)$")
    return df.assign(stem=parts[0], seed=parts[1].astype(int))


# -- Sections -------------------------------------------------------------

def section_data():
    print("\nData")
    H = load_hull(HULLOUT_MAP["2025"])
    nfree = free_sizes(H)
    add("n_compounds", "Methods", "non-elemental compounds", "101 491",
        f"{H['ncomp']:,}".replace(",", " "), "MP GGA/GGA+U, MP2020-corrected (data/2025)")
    add("n_reactions_N2", "Fig. 5 caption", "reactions with N >= 2", "99 306",
        f"{int((nfree >= 2).sum()):,}".replace(",", " "))
    add("frac_stable", "Results, false positives", "fraction stable (dH_d <= 0)", "33.8%",
        pct(H["hd"] <= 0))
    logs = load_logs("selection_s*.json")
    add("n_train_fold", "Sec. iiLoss", "training compounds per outer fold", "81 193",
        str(sorted({log["n_train"] for log in logs})))
    add("n_test_fold", "Sec. iiLoss", "test compounds per outer fold", "20 298",
        str(sorted({log["n_test"] for log in logs})))
    add("n_val_fold", "new", "validation compounds per fold (lam selection)", "-",
        str(sorted({log["n_val"] for log in logs})), "10 % of the training fold")

    # Do the stored reactions reproduce the stored dH_d?
    for year, old in (("2020", "-"), ("2026", "-"), ("2025", "-")):
        Hy = load_hull(HULLOUT_MAP[year])
        d = np.r_[Hy["hf"], np.zeros(Hy["M"] - Hy["ncomp"])]
        y, _ = reaction_sums(Hy, d)
        gap = np.abs(y + Hy["hd"])
        flip = ((-y) <= 0) != (Hy["hd"] <= 0)
        add(f"hull_consistency_{year}", "new (SI, data)",
            f"{year} hull: reactions off by > 10 meV / labels inconsistent", old,
            f"{pct(gap > 0.01)} / {pct(flip)}", "S dfH should equal -dH_d")

    from models.iiLossNN import iiLossNN
    from reaction_coverage import coverage
    with open(HULLOUT_MAP["2025"]) as f:
        data = json.load(f)
    features, _, labels = iiLossNN().preprocess(data)
    n_with = n_kept = 0
    fracs = []
    for train, _ in KFold(n_splits=5, shuffle=True, random_state=10).split(features):
        a, b, fr = coverage(data, labels, train)
        n_with += a
        n_kept += b
        fracs += fr
    add("train_rxn_retained", "Sec. iiLoss", "training reactions retained", "92.6%",
        pct(n_kept / n_with), "reaction_coverage.py; reproduces 92.6 % on the 2026 file")
    add("train_rxn_complete", "Sec. iiLoss", "retained reactions with every product kept", "60.5%",
        pct(np.array(fracs) == 1.0))
    add("train_rxn_kept_products", "Sec. iiLoss", "products kept, mean over retained reactions",
        "83.4%", pct(fracs), "complete reactions count 1")
    add("train_rxn_kept_incomplete", "new (wording)", "products kept, incomplete reactions only",
        "-", pct([f for f in fracs if f < 1.0]))


def section_table1():
    print("\nTable 1 (2020 benchmark)")
    s = pd.read_csv(os.path.join(RESULTS, "2020", "spectral_summary_2020.csv")).set_index("model")
    preprint = {"ElFrac": (0.23, 0.18, 0.75), "Meredig": (0.17, 0.18, 0.92),
                "Magpie": (0.13, 0.16, 0.99), "AutoMat": (0.12, 0.15, 1.00),
                "ElemNet": (0.10, 0.12, 1.03), "Roost": (0.06, 0.08, 1.10),
                "CGCNN": (0.03, 0.04, 1.02)}
    for m, (f, d, x) in preprint.items():
        add(f"T1_MAEf_{m}", "Tab. 1", f"{m}: MAE(dH_f)", f"{f:.2f}", f"{s.loc[m, 'MAE_f']:.2f}")
        add(f"T1_MAEd_{m}", "Tab. 1", f"{m}: MAE(dH_d)", f"{d:.2f}", f"{s.loc[m, 'MAE_d']:.2f}")
        add(f"T1_xi_{m}", "Tab. 1", f"{m}: xi_rms", f"{x:.2f}", f"{s.loc[m, 'xi_rms']:.2f}")
        add(f"T1_Xi2Q_{m}", "new Tab. 1 column", f"{m}: Xi2_Q", "-", f"{s.loc[m, 'Xi2_Q']:.3f}")
        add(f"T1_pg_{m}", "new Tab. 1 column", f"{m}: p_gauge", "-", pct(s.loc[m, "p_gauge"]))


def gauge_extras(H):
    """Per run and seed: the preprint's gauge share and remainder index, and
    the share of reactions whose gauge-fixed xi is below 1."""
    nc, nfree = H["ncomp"], free_sizes(H)
    rows, cols, vals = H["rows"], H["cols"], H["vals"]
    # The preprint's definition (eq. 18): compounds only, W from N >= 2 rows.
    m2 = (nfree >= 2)[rows]
    w = np.zeros(H["M"])
    np.add.at(w, cols[m2], vals[m2] ** 2)
    w = w[:nc]
    keep = np.flatnonzero(w > 0)
    sw = np.sqrt(w[keep])
    Q, sv, _ = np.linalg.svd(sw[:, None] * H["X"][:nc][keep], full_matrices=False)
    Q = Q[:, sv > 1e-10 * sv.max()]
    out = {}
    stems = ("iiLoss_finetune_0.0", "iiLossRaw_finetune_0.2", "iiLoss_nested", "HdLoss_nested")
    for stem in stems:
        acc = {"share18": [], "rem18": [], "xi_lt1_Q": []}
        for path in sorted(glob.glob(os.path.join(ML_2025, f"{stem}_s*", "ml_input.json"))):
            with open(path) as f:
                d, have = residual(H, json.load(f))
            y, q = reaction_sums(H, d)
            m = (nfree >= 2) & (q > 0)
            _, dq = gauge_split(H, d, np.ones(H["R"], bool))
            _, qq = reaction_sums(H, dq)
            acc["xi_lt1_Q"].append(np.mean(y[m] ** 2 / qq[m] < 1))
            u = d[keep] * sw
            a = Q.T @ u
            acc["share18"].append(a @ a / (u @ u))
            dp = d.copy()
            dp[keep] = d[keep] - (Q @ a) / sw       # elements stay at 0
            yp, qp = reaction_sums(H, dp)
            mp = (nfree >= 2) & (qp > 0)
            acc["rem18"].append(np.sqrt(np.mean(yp[mp] ** 2 / qp[mp])))
        out[stem] = acc
    return out


def section_crossvalidation():
    print("\nCross-validation on the 2025 hull")
    df = split_seed(pd.read_csv(os.path.join(RESULTS, "2025", "spectral_summary_2025.csv")))
    c, raw, hd, ii = ("iiLoss_finetune_0.0", "iiLossRaw_finetune_0.2", "HdLoss_finetune_0.1",
                      "iiLoss_nested")
    def col(stem, k):
        return seeds_of(df, stem)[k].to_numpy()

    # control, iiLossRaw at lam = 0.2, HdLoss at lam = 0.1
    preprint = {"MAE_f":     ("0.097(1)", "0.195(2)", "0.101(1)"),
                "MAE_d":     ("0.131(2)", "0.104(1)", "0.111(1)"),
                "precision": ("0.389(2)", "0.492(5)", "0.449(5)"),
                "recall":    ("0.800(5)", "0.647(13)", "0.643(10)"),
                "F1":        ("0.523(1)", "0.559(2)", "0.529(1)"),
                "xi_rms":    ("1.107(2)", "0.561(3)", "0.974(5)")}
    labels = ("control", "iiLossRaw l=0.2", "HdLoss l=0.1")
    for k, olds in preprint.items():
        for stem, label, old in zip((c, raw, hd), labels, olds):
            add(f"T2_{k}_{stem}", "Tab. 2", f"{k}, {label}", old, fmt(col(stem, k)))
    for stem in ("iiLoss_nested", "iiLossRaw_nested", "HdLoss_nested"):
        for k in ("MAE_f", "MAE_d", "precision", "recall", "F1", "maxF1", "ROC_AUC",
                  "PR_AUC", "MCC", "xi_rms", "Xi2_Q", "mlhull_F1", "mlhull_DAF",
                  "mlhull_MAE_hd"):
            add(f"T2new_{k}_{stem}", "new Tab. 2", f"{k}, {stem}", "-", fmt(col(stem, k)))
    for k in ("mlhull_F1", "mlhull_precision", "mlhull_recall", "mlhull_DAF", "mlhull_MAE_hd"):
        add(f"T2new_{k}_control", "new Tab. 2", f"{k}, control", "-", fmt(col(c, k)))

    add("abs_dMAE_d_raw", "Abstract; Results", "iiLossRaw l=0.2: change in MAE(dH_d)", "-21%",
        f"{rel(df, raw, c, 'MAE_d'):+.1f}%")
    add("abs_dF1_raw", "Abstract; Results", "iiLossRaw l=0.2: change in F1", "+6.8%",
        f"{rel(df, raw, c, 'F1'):+.1f}%")
    add("dMAE_d_hdloss", "Results", "HdLoss l=0.1: change in MAE(dH_d)", "-15%",
        f"{rel(df, hd, c, 'MAE_d'):+.1f}%")
    add("dF1_hdloss", "Results", "HdLoss l=0.1: change in F1", "+1.1%",
        f"{rel(df, hd, c, 'F1'):+.1f}%")
    for stem, old in ((c, "1.35"), (raw, "0.53"), (ii, "-")):
        add(f"ratio_{stem}", "Conclusions", f"MAE(dH_d)/MAE(dH_f), {stem}", old,
            f"{col(stem, 'MAE_d').mean() / col(stem, 'MAE_f').mean():.2f}")
    for k in ("MAE_f", "MAE_d", "F1", "maxF1", "ROC_AUC", "PR_AUC", "MCC", "mlhull_F1",
              "mlhull_DAF", "mlhull_MAE_hd"):
        add(f"head_{k}", "new abstract / headline", f"iiLoss nested vs control: {k}", "-",
            f"{rel(df, ii, c, k):+.1f}%")
    for k in ("MAE_f", "MAE_d", "F1", "ROC_AUC", "mlhull_F1", "mlhull_MAE_hd"):
        add(f"vsHd_{k}", "new", f"iiLoss nested vs HdLoss nested: {k}", "-",
            f"{rel(df, ii, 'HdLoss_nested', k):+.1f}%")

    # Fig. 5 and the gauge paragraph
    for stem, old in ((c, "52%"), (raw, "90%"), (ii, "-"), ("HdLoss_nested", "-")):
        add(f"xi_lt1_{stem}", "Fig. 5; Results", f"reactions with xi < 1, {stem}", old,
            pct(col(stem, "frac_xi_lt1")))
    extra = gauge_extras(load_hull(HULLOUT_MAP["2025"]))
    for stem in extra:
        add(f"xi_lt1_Q_{stem}", "new", f"reactions with gauge-fixed xi < 1, {stem}", "-",
            pct(extra[stem]["xi_lt1_Q"]))
    add("share18_control", "SI Tab. gauge", "gauge share (eq. 18), control", "4.5%",
        pct(extra[c]["share18"]))
    add("share18_raw", "Results; SI Tab. gauge", "gauge share (eq. 18), iiLossRaw l=0.2", "54.8%",
        pct(extra[raw]["share18"]))
    add("share18_iiLoss", "new", "gauge share (eq. 18), iiLoss nested", "-",
        pct(extra[ii]["share18"]))
    for stem in (c, raw, ii):
        add(f"p_gauge_{stem}", "new (Xi2_Q convention)", f"p_gauge, {stem}", "-",
            pct(col(stem, "p_gauge")))
    add("rem18_control", "SI Tab. gauge", "xi_rms of the remainder (eq. 18), control", "1.150(2)",
        fmt(extra[c]["rem18"]))
    add("rem18_raw", "SI Tab. gauge", "xi_rms of the remainder (eq. 18), iiLossRaw l=0.2",
        "0.949(8)", fmt(extra[raw]["rem18"]))
    add("xi_red_raw", "Results", "iiLossRaw: reduction of xi_rms", "49%",
        f"{-rel(df, raw, c, 'xi_rms'):.0f}%")
    add("xi_red_rem", "Results", "iiLossRaw: reduction of xi_rms of the remainder", "17%",
        f"{100 * (1 - np.mean(extra[raw]['rem18']) / np.mean(extra[c]['rem18'])):.0f}%")

    # False positives, over the pooled out-of-fold predictions
    def fp(stem):
        return col(stem, "FP").mean()
    add("FP_control", "Results, false positives", "false positives, control", "43 149",
        f"{fp(c):.0f}")
    for stem, old in ((raw, "20 244 (47%)"), (ii, "-")):
        add(f"FP_removed_{stem}", "Results, false positives", f"false positives removed, {stem}",
            old, f"{fp(c) - fp(stem):.0f} ({100 * (1 - fp(stem) / fp(c)):.0f}%)")
    for stem, old in ((c, "69.6%"), (raw, "44.4%"), (ii, "-")):
        add(f"predpos_{stem}", "Results, false positives", f"predicted-stable fraction, {stem}",
            old, pct(col(stem, "frac_pred_stable")))
    prevalence = col(c, "frac_true_stable").mean()
    for stem, old in ((c, "1.15x"), (raw, "1.46x"), (ii, "-")):
        add(f"enrich_{stem}", "Results, false positives", f"enrichment over the base rate, {stem}",
            old, f"{col(stem, 'precision').mean() / prevalence:.2f}x")

    # The denominator stabiliser
    eps = ("iiLoss_finetune_eps1e-05_0.2", "iiLoss_finetune_0.2", "iiLoss_finetune_eps0.001_0.2")
    for k in ("F1", "MAE_d", "MAE_f"):
        v = [col(stem, k).mean() for stem in eps]
        add(f"eps_{k}", "Methods (eps)", f"iiLoss l=0.2 at eps 1e-5 / 1e-4 / 1e-3: {k}",
            "~1% (iiLossRaw)",
            " / ".join(f"{x:.4f}" for x in v)
            + f"  (spread {100 * (max(v) - min(v)) / np.mean(v):.1f}%)")
    for stem in eps:
        add(f"eps_vs_control_{stem}", "Methods (eps)", f"{stem} vs control: MAE(dH_d) / MAE(dH_f)",
            "-", f"{rel(df, stem, c, 'MAE_d'):+.1f}% / {rel(df, stem, c, 'MAE_f'):+.1f}%")


def change(values):
    """Relative change in % over seeds, as mean(sd in the last digit)."""
    v = np.asarray(values, float)
    return f"{v.mean():+.1f}({int(round(v.std(ddof=1) * 10))})%"


def reaction_errors(H, stem, gauge=False):
    """Per seed: y, |c| (gauge-fixed if asked) and |delta| of the compounds."""
    Y, R, A = [], [], []
    for path in sorted(glob.glob(os.path.join(ML_2025, f"{stem}_s*", "ml_input.json"))):
        with open(path) as f:
            d, _ = residual(H, json.load(f))
        y, q = reaction_sums(H, d)
        if gauge:
            _, dq = gauge_split(H, d, np.ones(H["R"], bool))
            _, q = reaction_sums(H, dq)
        Y.append(y)
        R.append(np.sqrt(q))
        A.append(np.abs(d[:H["ncomp"]]))
    return np.array(Y), np.array(R), np.array(A)


def section_fig5():
    print("\nFigure 5: error distribution, N, the xi / |c| split, compound roles")
    H = load_hull(HULLOUT_MAP["2025"])
    nfree, roles = free_sizes(H), product_counts(H)
    c, ii = "iiLoss_finetune_0.0", "iiLoss_nested"
    yc, rc, ac = reaction_errors(H, c)

    # (c) the distribution of dH_d,ML - dH_d,DFT = -y, in eV/atom
    for stem in (c, ii):
        e = -reaction_errors(H, stem)[0]
        q25, q75 = np.percentile(e, [25, 75], axis=1)
        add(f"F5c_MAE_{stem}", "Fig. 5c caption", f"mean |error| of dH_d, {stem}", "-",
            fmt(np.abs(e).mean(axis=1)))
        add(f"F5c_median_{stem}", "Fig. 5c caption", f"median |error| of dH_d, {stem}", "-",
            fmt(np.median(np.abs(e), axis=1)))
        add(f"F5c_bias_{stem}", "Fig. 5c caption", f"mean signed error of dH_d, {stem}", "-",
            fmt(e.mean(axis=1)))
        add(f"F5c_sd_{stem}", "Fig. 5c caption", f"sd of the dH_d error, {stem}", "-",
            fmt(e.std(axis=1)))
        add(f"F5c_IQR_{stem}", "Fig. 5c caption", f"interquartile range of the dH_d error, {stem}",
            "-", fmt(q75 - q25))

    # (d, e) per N, and the split of the change into its xi and |c| parts
    groups = [("1", nfree == 1), ("2", nfree == 2), ("3", nfree == 3), ("4", nfree == 4),
              ("5", nfree == 5), (">=6", nfree >= 6), ("all", nfree >= 1)]
    runs = (("", ii, False), ("Q_", ii, True), ("", "HdLoss_nested", False),
            ("", "iiLossRaw_nested", False), ("", "iiLossRaw_finetune_0.2", False))
    for tag, stem, gauge in runs:
        y0, r0, _ = reaction_errors(H, c, gauge) if gauge else (yc, rc, ac)
        y1, r1, _ = reaction_errors(H, stem, gauge)
        xi_part, c_part = error_split(y0, r0, y1, r1)
        mae0 = np.abs(y0)
        where = "Fig. 5d,e" if stem == ii and not gauge else "SI (error split)"
        for name, m in groups:
            if stem != ii and name != "all":
                continue
            base = mae0[:, m].mean(axis=1)
            label = f"N {name} ({int(m.sum())} reactions), {stem}"
            if not gauge:
                add(f"F5d_{name}_{stem}", where, f"change in MAE(dH_d), {label}", "-",
                    change(100 * (np.abs(y1)[:, m].mean(axis=1) / base - 1)))
            kind = "gauge-fixed " if gauge else ""
            add(f"F5e_{tag}xi_{name}_{stem}", where, f"{kind}xi part of it, {label}", "-",
                change(100 * xi_part[:, m].mean(axis=1) / base))
            add(f"F5e_{tag}c_{name}_{stem}", where, f"{kind}|c| part of it, {label}", "-",
                change(100 * c_part[:, m].mean(axis=1) / base))

    # (f) MAE(dH_f) by the number of reactions a compound enters as a product
    stable = H["hd"][:H["ncomp"]] <= 0
    groups = [("0", "product in 0 reactions", roles == 0),
              ("1-2", "product in 1-2 reactions", (roles >= 1) & (roles <= 2)),
              ("3-9", "product in 3-9 reactions", (roles >= 3) & (roles <= 9)),
              ("10-99", "product in 10-99 reactions", (roles >= 10) & (roles <= 99)),
              (">=100", "product in >=100 reactions", roles >= 100),
              ("stable", "on the hull (dH_d <= 0)", stable),
              ("unstable", "above the hull", ~stable),
              ("all", "all compounds", roles >= 0)]
    for stem in (ii, "HdLoss_nested", "iiLossRaw_nested", "iiLossRaw_finetune_0.2"):
        a1 = reaction_errors(H, stem)[2]
        for name, text, m in groups:
            if stem != ii and name not in ("0", ">=100", "all"):
                continue
            add(f"F5f_{name}_{stem}", "Fig. 5f" if stem == ii else "SI (error split)",
                f"change in MAE(dH_f), {text} ({int(m.sum())}), {stem}",
                "-", change(100 * (a1[:, m].mean(axis=1) / ac[:, m].mean(axis=1) - 1)))


def section_selection():
    print("\nlam selection")
    logs = load_logs("selection*.json")
    for width in sorted({log["width"] for log in logs}):
        sub = [log for log in logs if log["width"] == width]
        for obj in sub[0]["objectives"]:
            count = Counter(log["selected"][obj] for log in sub)
            add(f"lambda_{obj}_w{width}", "new (replaces the test-chosen lam of Tab. 2)",
                f"lam selected, {obj}, width {width} ({len(sub)} seed-folds)", "-",
                ", ".join(f"{k:g}: {v}" for k, v in sorted(count.items())))


def section_capacity():
    print("\nCapacity ladder")
    df = split_seed(pd.read_csv(os.path.join(RESULTS, "2025", "spectral_summary_2025.csv")))
    seeds = sorted(df[df.stem == "iiLoss_finetune_0.0_w32"].seed.unique())
    df = df[df.seed.isin(seeds)]
    old = "+0% (w32) ... +6.8% (w1024), F1"
    for width in (32, 64, 128, 256, 512, 1024):
        tag = "" if width == 1024 else f"_w{width}"
        ctl = "iiLoss_finetune_0.0" + tag
        add(f"cap_xi_w{width}", "Sec. iiLoss (capacity)", f"width {width}: xi_rms of the control",
            "1.06-1.11", fmt(seeds_of(df, ctl).xi_rms))
        for stem in ("iiLoss_nested", "HdLoss_nested"):
            for k in ("MAE_f", "MAE_d", "F1", "ROC_AUC"):
                add(f"cap_{k}_{stem}_w{width}", "Sec. iiLoss (capacity); SI Fig. S3",
                    f"width {width}: {stem} vs control, {k}",
                    old if (k == "F1" and stem == "iiLoss_nested") else "-",
                    f"{rel(df, stem + tag, ctl, k):+.1f}%")


def section_prospective():
    print("\nProspective test on WBM")
    df = split_seed(pd.read_csv(os.path.join(RESULTS, "wbm", "prospective_summary.csv")))
    c = "iiLoss_finetune_0.0"
    for stem in ("iiLoss_finetune_0.2", "iiLossRaw_finetune_0.2", "HdLoss_finetune_0.1"):
        for k in ("closed_MAE_d", "MAE_f_wbm", "uniq_F1", "uniq_DAF", "uniq_precision",
                  "uniq_recall", "uniq_MAE_hull", "uniq_ROC_AUC", "full_F1", "closed_Xi2_Q",
                  "closed_F1", "novel_MAE_hull", "novel_DAF", "novel_F1", "closed_novel_MAE_d"):
            add(f"wbm_{k}_{stem}", "new (prospective)", f"{stem} vs control: {k}", "-",
                f"{rel(df, stem, c, k):+.1f}%")
    ctl = seeds_of(df, c)
    add("wbm_control_uniq_F1", "new (prospective)", "control: official F1 (unique prototypes)",
        "-", fmt(ctl.uniq_F1))
    add("wbm_control_closed_MAE_d", "new (prospective)", "control: closed MAE(dH_d)", "-",
        fmt(ctl.closed_MAE_d))


def section_mlip():
    print("\nMatbench Discovery models")
    d = pd.read_csv(os.path.join(RESULTS, "wbm", "mlip_closed.csv"))
    for group, sub in d.groupby("group"):
        add(f"mlip_Xi2Q_{group}", "new (SI)", f"{group}: median Xi2_Q (n = {len(sub)})", "-",
            f"{sub.Xi2_Q.median():.3f}")
        add(f"mlip_CI_{group}", "new (SI)", f"{group}: interval below / above 1", "-",
            f"{(sub.Xi2_Q_hi < 1).sum()} / {(sub.Xi2_Q_lo > 1).sum()}")
        add(f"mlip_ratio_{group}", "new (SI)", f"{group}: median MAE(dH_d)/MAE(dH_f)", "-",
            f"{(sub.MAE_d_closed / sub.MAE_f).median():.2f}")
    ml = d[d.group != "pre-MLIP"]
    X = np.c_[np.ones(len(ml)), np.log(ml.MAE_f), (ml.group == "extended-data MLIP").astype(float)]
    coef, *_ = np.linalg.lstsq(X, np.log(ml.MAE_d_closed), rcond=None)
    add("mlip_shift", "new (SI)", "extended-data potentials, MAE(dH_d) at equal MAE(dH_f)", "-",
        f"{100 * (np.exp(coef[2]) - 1):+.1f}%",
        "log-log fit with a group term; p in mlip_score.py")
    a, b = ml.MAE_f.to_numpy(), ml.MAE_d_closed.to_numpy()
    n = len(a)
    flips = sum((a[i] - a[j]) * (b[i] - b[j]) < 0 for i in range(n) for j in range(i + 1, n))
    add("mlip_inversions", "new (SI)", "model pairs ranked differently by MAE(dH_f) and MAE(dH_d)",
        "-", f"{flips} of {n * (n - 1) // 2}")
    add("mlip_set", "new (SI)", "common evaluation set of the potentials: compounds / reactions",
        "-", f"{int(ml.n_compounds.iloc[0])} / {int(ml.n_reactions.iloc[0])}")


def write():
    with open(OUT + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    with open(OUT + ".md", "w") as f:
        f.write("# Numbers of the revised manuscript\n\n"
                "Generated by `scripts/paper_numbers.py`; do not edit by hand. Preprint: "
                "chemRxiv 10.26434/chemrxiv.15008797/v1. Revision values are means over seeds, "
                "with the sample standard deviation in the last digits in brackets.\n\n"
                "| key | where | quantity | preprint | revision | note |\n"
                "|---|---|---|---|---|---|\n")
        for r in ROWS:
            f.write(f"| {r['key']} | {r['preprint_location']} | {r['quantity']} | "
                    f"{r['preprint']} | {r['revision']} | {r['note']} |\n")
    print(f"\n{len(ROWS)} numbers written to {OUT}.md and {OUT}.csv")


def main():
    section_data()
    section_table1()
    section_crossvalidation()
    section_fig5()
    section_selection()
    section_capacity()
    section_prospective()
    section_mlip()
    write()


if __name__ == "__main__":
    main()
