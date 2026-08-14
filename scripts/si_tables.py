"""Emit the Supporting Information tables as LaTeX.

Values are means over seeds with the sample standard deviation in units of the
last quoted digit, matching the convention of Table 2 of the main text.

    sweep     one table per variant, rows over lam
    eps       the denominator stabiliser check at the operating point
    surgery   conditional against unconditional gradient surgery
    capacity  the lam = 0 ladder over base widths

Usage
-----
    python scripts/si_tables.py                     # all sections, to stdout
    python scripts/si_tables.py --section eps
    python scripts/si_tables.py --section sweep > si_tables.tex
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


def rows_table(groups, rows, metrics, decimals, caption, tag, first_col):
    """A table whose rows are named runs rather than lam values."""
    present = [(label, stem) for label, stem in rows if stem in groups]
    if not present:
        return None

    cols = "l" + "r" * len(metrics)
    head = " & ".join(HEADERS.get(m, m) for m in metrics)

    out = ["\\begin{table}[H]", "  \\centering", "  \\small",
           f"  \\begin{{tabular}}{{{cols}}}\\toprule",
           f"    {first_col} & {head} \\\\\\midrule"]
    for label, stem in present:
        cells = " & ".join(cell(groups[stem], m, decimals) for m in metrics)
        out.append(f"    {label} & {cells} \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}",
            f"  \\caption{{{caption}}}", f"  \\label{{tab:si-{tag}}}",
            "\\end{table}", ""]
    return "\n".join(out)


def section_eps(groups, metrics, decimals):
    rows = [(r"$10^{-3}$", "iiLoss_finetune_eps1e-3_0.2"),
            (r"$10^{-4}$", "iiLoss_finetune_0.2"),
            (r"$10^{-6}$", "iiLoss_finetune_eps1e-6_0.2")]
    caption = (r"Sensitivity of \textit{iiLoss} at $\lambda = 0.2$ to the "
               r"denominator stabiliser $\epsilon$, in eV$^2$\,atom$^{-2}$. "
               r"The value $10^{-4}$ used throughout was fixed a priori. "
               r"Values are means over three random seeds, with the "
               r"parenthesised digits giving the sample standard deviation in "
               r"units of the last quoted digit.")
    return rows_table(groups, rows, metrics, decimals, caption, "eps",
                      r"$\epsilon$")


def section_surgery(groups, metrics, decimals, lams):
    rows = []
    for lam in lams:
        rows.append((f"{lam} unconditional", f"iiLoss_pcgrad_{lam}"))
        rows.append((f"{lam} conditional",   f"iiLoss_pcgradc_{lam}"))
    caption = (r"Gradient surgery applied at every step against the "
               r"conditional rule of the original \textit{PCGrad} "
               r"formulation, in which the projection is applied only where "
               r"the two gradients conflict. Values are means over three "
               r"random seeds, with the parenthesised digits giving the "
               r"sample standard deviation in units of the last quoted digit.")
    return rows_table(groups, rows, metrics, decimals, caption, "surgery",
                      r"$\lambda$")


def section_capacity(groups, metrics, decimals, widths):
    rows = []
    for w in widths:
        suffix = "" if w == 1024 else f"_w{w}"
        rows.append((str(w), f"iiLoss_finetune_0.0{suffix}"))
    caption = (r"Unregularised baseline over the capacity ladder, by base "
               r"width. The interference index is essentially independent of "
               r"model size. Values are means over three random seeds, with "
               r"the parenthesised digits giving the sample standard "
               r"deviation in units of the last quoted digit.")
    return rows_table(groups, rows, metrics, decimals, caption, "capacity",
                      "base width")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2026")
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--section", default="all",
                        choices=["all", "sweep", "eps", "surgery", "capacity"])
    parser.add_argument("--surgery-lams", nargs="+",
                        default=["0.1", "0.2", "0.3", "0.4", "0.5"])
    parser.add_argument("--widths", nargs="+", type=int,
                        default=[32, 64, 128, 256, 512, 1024])
    args = parser.parse_args()

    groups = load(args.dataset)
    blocks = []

    if args.section in ("all", "sweep"):
        for tag, (prefix, label) in enumerate(VARIANTS, start=1):
            blocks.append(table(groups, prefix, label, args.metrics,
                                args.decimals, tag))
    if args.section in ("all", "eps"):
        blocks.append(section_eps(groups, args.metrics, args.decimals))
    if args.section in ("all", "surgery"):
        blocks.append(section_surgery(groups, args.metrics, args.decimals,
                                      args.surgery_lams))
    if args.section in ("all", "capacity"):
        blocks.append(section_capacity(groups, args.metrics, args.decimals,
                                       args.widths))

    for block in blocks:
        print(block if block is not None else "% no runs found for this table")


if __name__ == "__main__":
    main()
