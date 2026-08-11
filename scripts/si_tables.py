"""Emit the Supporting Information sweep tables as LaTeX.

One table per variant, rows over lam, columns over the reported metrics.
Values are means over seeds with the sample standard deviation in units of
the last quoted digit, matching the convention of Table 2 of the main text.

Usage
-----
    python scripts/si_tables.py                     # four tables, to stdout
    python scripts/si_tables.py > si_tables.tex
    python scripts/si_tables.py --decimals 4
    python scripts/si_tables.py --metrics Hf_MAE Hd_MAE F1 rms_xi Precision Recall
"""

import argparse
import csv
import os
from statistics import mean, stdev

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# -- Table contents -------------------------------------------------------
VARIANTS = [
    ("iiLoss_finetune_", "iiLoss"),
    ("HdLoss_finetune_", "HdLoss"),
    ("iiLoss_pcgrad_",   "iiLoss + PCGrad"),
    ("HdLoss_pcgrad_",   "HdLoss + PCGrad"),
]

HEADERS = {
    "Hf_MAE":    r"$\mathrm{MAE}(\Del_\mathrm{f}H)$",
    "Hd_MAE":    r"$\mathrm{MAE}(\Del_\mathrm{d}H)$",
    "F1":        r"$F_1$",
    "rms_xi":    r"$\xi_\mathrm{rms}$",
    "Precision": "Precision",
    "Recall":    "Recall",
    "median_xi": r"$\tilde{\xi}$",
}

DEFAULT_METRICS = ["Hf_MAE", "Hd_MAE", "F1", "rms_xi"]
BASELINE = "iiLoss_finetune_0.0"


# -- Helpers --------------------------------------------------------------
def load(dataset):
    """Group summary rows by run name with the _s<seed> suffix removed."""
    path = os.path.join(REPO_ROOT, "results", dataset,
                        f"interference_summary_{dataset}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run interference_score.py first")
    groups = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            stem, _, tail = row["model"].rpartition("_s")
            if stem and tail.isdigit():
                groups.setdefault(stem, []).append(row)
    return groups


def cell(rows, metric, decimals):
    """Mean with the sample standard deviation in units of the last digit."""
    vals = [float(r[metric]) for r in rows]
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    unc = max(1, round(s * 10 ** decimals))
    return f"{m:.{decimals}f}({unc})"


def lams_for(groups, prefix):
    """Sorted lam values present for one variant."""
    found = []
    for stem in groups:
        if not stem.startswith(prefix):
            continue
        try:
            found.append(float(stem[len(prefix):]))
        except ValueError:
            continue
    return sorted(v for v in found if v > 0)


# -- Output ---------------------------------------------------------------
def table(groups, prefix, label, metrics, decimals, tag):
    lams = lams_for(groups, prefix)
    if not lams:
        return None

    cols = "l" + "r" * len(metrics)
    head = " & ".join(HEADERS.get(m, m) for m in metrics)

    out = ["\\begin{table}[H]", "  \\centering", "  \\small",
           f"  \\begin{{tabular}}{{{cols}}}\\toprule",
           f"    $\\lambda$ & {head} \\\\\\midrule"]

    if BASELINE in groups:
        cells = " & ".join(cell(groups[BASELINE], m, decimals) for m in metrics)
        out.append(f"    0.0 & {cells} \\\\")
        out.append("    \\midrule")

    for lam in lams:
        stem = f"{prefix}{lam:.1f}"
        if stem not in groups:
            continue
        cells = " & ".join(cell(groups[stem], m, decimals) for m in metrics)
        out.append(f"    {lam:.1f} & {cells} \\\\")

    n_seeds = len(groups.get(f"{prefix}{lams[0]:.1f}", []))
    out += ["    \\bottomrule", "  \\end{tabular}",
            f"  \\caption{{Complete sweep for \\textit{{{label}}}. "
            f"Values are means over {n_seeds} random seeds, with the "
            f"parenthesised digits giving the sample standard deviation in "
            f"units of the last quoted digit. The $\\lambda = 0$ row is the "
            f"matched baseline shared by all variants.}}",
            f"  \\label{{tab:si-sweep-{tag}}}", "\\end{table}", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2026")
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()

    groups = load(args.dataset)
    for tag, (prefix, label) in enumerate(VARIANTS, start=1):
        block = table(groups, prefix, label, args.metrics, args.decimals, tag)
        if block is None:
            print(f"% no runs found for {prefix}")
            continue
        print(block)


if __name__ == "__main__":
    main()
