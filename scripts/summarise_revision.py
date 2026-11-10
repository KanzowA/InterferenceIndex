"""Paired comparisons behind the revision's tables.

Usage
-----
    python scripts/summarise_revision.py                       # all sections
    python scripts/summarise_revision.py --section nested      # one section

Sections
    nested       every nested run against the matched control (Table 2)
    head         iiLoss against HdLoss, both nested
    eps          the denominator stabiliser at lam = 0.2
    selection    lam chosen on the validation split, per objective and width
    capacity     the nested runs against the control at every base width
    prospective  the preregistered comparisons on the WBM discovery set

Every comparison is paired by seed: the relative change is averaged over seeds,
"k/n" counts the seeds in which the run beats its control, and p is a paired
t-test. Lower is better for errors and the indices, higher for everything else;
p_gauge is reported without a direction.

Reads results/2025/spectral_summary_2025.csv, results/2025/nested/ and
results/wbm/prospective_summary.csv; writes the paired tables beside them as
nested_paired.csv, capacity_paired.csv and prospective_paired.csv.
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

RESULTS_2025 = os.path.join(REPO_ROOT, "results", "2025")
RESULTS_WBM = os.path.join(REPO_ROOT, "results", "wbm")

CONTROL = "iiLoss_finetune_0.0"
LOWER = {"MAE_f", "MAE_d", "xi_rms", "Xi2", "Xi2_Q", "mlhull_MAE_hd", "MAE_f_wbm",
         "full_MAE_hull", "uniq_MAE_hull", "novel_MAE_hull", "uniq_RMSE_hull",
         "closed_MAE_d", "closed_novel_MAE_d", "closed_xi_rms", "closed_Xi2_Q"}
NEUTRAL = {"p_gauge", "closed_p_gauge"}

NESTED_RUNS = ["iiLoss_nested", "iiLossRaw_nested", "HdLoss_nested",
               "iiLoss_finetune_0.2", "iiLossRaw_finetune_0.2", "HdLoss_finetune_0.1"]
EPS_RUNS = ["iiLoss_finetune_eps1e-05_0.2", "iiLoss_finetune_0.2",
            "iiLoss_finetune_eps0.001_0.2"]
NESTED_METRICS = ["MAE_f", "MAE_d", "xi_rms", "Xi2_Q", "p_gauge", "frac_xi_lt1",
                  "F1", "precision", "recall", "maxF1", "ROC_AUC", "PR_AUC", "MCC",
                  "mlhull_F1", "mlhull_precision", "mlhull_recall", "mlhull_DAF",
                  "mlhull_MAE_hd"]
PROSPECTIVE_RUNS = ["iiLoss_finetune_0.2", "iiLossRaw_finetune_0.2", "HdLoss_finetune_0.1"]
PROSPECTIVE_METRICS = ["MAE_f_wbm", "uniq_MAE_hull", "uniq_F1", "uniq_DAF",
                       "uniq_precision", "uniq_recall", "uniq_ROC_AUC", "uniq_PR_AUC",
                       "full_F1", "full_DAF", "closed_MAE_d", "closed_xi_rms",
                       "closed_Xi2_Q", "closed_F1", "novel_MAE_hull", "novel_DAF",
                       "novel_F1", "closed_novel_MAE_d"]


# -- Helpers --------------------------------------------------------------

def split_seed(df):
    """Add stem and seed columns from run names of the form <stem>_s<seed>."""
    parts = df["model"].str.extract(r"^(.*)_s(\d+)$")
    return df.assign(stem=parts[0], seed=parts[1].astype(int))


def paired(df, run, control, metrics):
    """Relative change of `run` against `control`, paired by seed."""
    a = df[df.stem == run].set_index("seed")
    c = df[df.stem == control].set_index("seed")
    seeds = sorted(set(a.index) & set(c.index))
    a, c = a.loc[seeds], c.loc[seeds]
    rows = []
    for k in metrics:
        if k not in a or a[k].isna().all():
            continue
        rel = 100 * (a[k] / c[k] - 1)
        better = (rel < 0) if k in LOWER else (rel > 0)
        p = stats.ttest_rel(a[k], c[k]).pvalue if len(seeds) > 1 else np.nan
        rows.append(dict(run=run, control=control, metric=k, n=len(seeds),
                         control_mean=c[k].mean(), run_mean=a[k].mean(),
                         run_sd=a[k].std(ddof=1) if len(seeds) > 1 else np.nan,
                         rel_pct=rel.mean(),
                         n_better=np.nan if k in NEUTRAL else int(better.sum()), p=p))
    return rows


def show(rows, title):
    print(f"\n{title}")
    for r in rows:
        tag = ("   " if np.isnan(r["n_better"])
               else f"{int(r['n_better'])}/{r['n']}")
        print(f"  {r['metric']:18s} {r['control_mean']:9.4f} -> {r['run_mean']:9.4f}  "
              f"{r['rel_pct']:+7.2f} %  {tag}  p={r['p']:.2g}")


# -- Sections -------------------------------------------------------------

def section_nested(df):
    out = []
    for run in NESTED_RUNS:
        rows = paired(df, run, CONTROL, NESTED_METRICS)
        show(rows, f"{run} against {CONTROL}")
        out += rows
    return out


def section_head(df):
    rows = paired(df, "iiLoss_nested", "HdLoss_nested",
                  ["MAE_f", "MAE_d", "F1", "maxF1", "ROC_AUC", "PR_AUC", "MCC",
                   "mlhull_F1", "mlhull_DAF", "mlhull_MAE_hd"])
    show(rows, "iiLoss_nested against HdLoss_nested")
    return rows


def section_eps(df):
    out = []
    for run in EPS_RUNS:
        rows = paired(df, run, CONTROL, ["MAE_f", "MAE_d", "F1", "maxF1", "ROC_AUC"])
        show(rows, f"{run} against {CONTROL}")
        out += rows
    return out


def section_selection():
    print("\nlam chosen on the validation split (seed-folds per value)")
    logs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_2025, "nested", "selection*.json"))):
        with open(path) as f:
            log = json.load(f)
        logs.setdefault(log["width"], []).append(log)
    for width in sorted(logs):
        for obj in logs[width][0]["objectives"]:
            count = Counter(log["selected"][obj] for log in logs[width])
            print(f"  width {width:5d}  {obj:10s} "
                  + "  ".join(f"{lam:g}: {n}" for lam, n in sorted(count.items()))
                  + f"   ({len(logs[width])} seed-folds)")


def section_capacity(df):
    out = []
    widths = sorted({int(w[2:]) for w in df.stem.str.extract(r"(_w\d+)$")[0].dropna()})
    seeds_small = sorted(int(s) for s in df[df.stem.str.endswith(f"_w{widths[0]}")].seed.unique()) if widths else []
    print(f"\nCapacity ladder (seeds {seeds_small}; width 1024 is the nested run, same seeds)")
    for width in widths + [1024]:
        tag = "" if width == 1024 else f"_w{width}"
        sub = df[df.seed.isin(seeds_small)] if width == 1024 else df
        c = sub[sub.stem == CONTROL + tag]
        print(f"  width {width:5d}  control: MAE_f {c.MAE_f.mean():.4f}  MAE_d {c.MAE_d.mean():.4f}  "
              f"xi_rms {c.xi_rms.mean():.3f}  F1 {c.F1.mean():.4f}")
        for run in ("iiLoss_nested", "iiLoss_finetune_0.2", "HdLoss_nested", "HdLoss_finetune_0.1"):
            rows = paired(sub, run + tag, CONTROL + tag,
                          ["MAE_f", "MAE_d", "F1", "maxF1", "ROC_AUC", "PR_AUC"])
            for r in rows:
                r["width"] = width
            out += rows
            print(f"    {run:22s} " + " | ".join(
                f"{r['metric']} {r['rel_pct']:+5.1f}% {int(r['n_better'])}/{r['n']}" for r in rows))
    return out


def section_prospective(df):
    out = []
    for run in PROSPECTIVE_RUNS:
        rows = paired(df, run, CONTROL, PROSPECTIVE_METRICS)
        show(rows, f"WBM: {run} against {CONTROL}")
        out += rows
    rows = paired(df, "iiLoss_finetune_0.2", "HdLoss_finetune_0.1",
                  ["closed_MAE_d", "MAE_f_wbm", "uniq_F1", "uniq_MAE_hull", "uniq_ROC_AUC"])
    show(rows, "WBM: iiLoss_finetune_0.2 against HdLoss_finetune_0.1 (H4)")
    return out + rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", default="all",
                        choices=["all", "nested", "head", "eps", "selection",
                                 "capacity", "prospective"])
    args = parser.parse_args()
    want = lambda s: args.section in ("all", s)

    spectral = os.path.join(RESULTS_2025, "spectral_summary_2025.csv")
    if any(want(s) for s in ("nested", "head", "eps", "capacity")):
        if not os.path.exists(spectral):
            raise SystemExit(f"{spectral} not found - run spectral_score.py 2025 first")
        df = split_seed(pd.read_csv(spectral))
        full = df[~df.stem.str.contains(r"_w\d+$")]
        nested = []
        if want("nested"):
            nested += section_nested(full)
        if want("head"):
            nested += section_head(full)
        if want("eps"):
            nested += section_eps(full)
        if nested and args.section == "all":
            pd.DataFrame(nested).to_csv(os.path.join(RESULTS_2025, "nested_paired.csv"),
                                        index=False, float_format="%.6g")
        if want("capacity"):
            cap = section_capacity(df)
            if args.section in ("all", "capacity"):
                pd.DataFrame(cap).to_csv(os.path.join(RESULTS_2025, "capacity_paired.csv"),
                                         index=False, float_format="%.6g")
    if want("selection"):
        section_selection()
    if want("prospective"):
        pros = os.path.join(RESULTS_WBM, "prospective_summary.csv")
        if not os.path.exists(pros):
            raise SystemExit(f"{pros} not found - run prospective_score.py first")
        rows = section_prospective(split_seed(pd.read_csv(pros)))
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_WBM, "prospective_paired.csv"),
                                  index=False, float_format="%.6g")
    print()


if __name__ == "__main__":
    main()
