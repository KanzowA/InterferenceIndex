"""Aggregate the seeded sweep into the comparisons the manuscript reports.

Usage
-----
    python scripts/summarise_runs.py                 # all sections
    python scripts/summarise_runs.py --section lam   # one section
    python scripts/summarise_runs.py --width 256     # a different base width

Sections
    lam        lam sweep per loss, against that width's matched lam=0 control
    surgery    no surgery vs unconditional vs conditional PCGrad at one lam
    capacity   lam=0 ladder, and the relative gain at each width
    matched    capacity models paired with iiLoss runs of similar Hf accuracy
    table2     the Table 2 row set, mean +/- sd

Values are mean +/- sample standard deviation over seeds. Runs are read from
results/<dataset>/interference_summary_<dataset>.csv, so run
interference_score.py first.
"""

import argparse
import csv
import os
from statistics import mean, stdev

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

METRICS = ["F1", "Hd_MAE", "rms_xi", "Hf_MAE"]

# Approximate trainable parameter counts for hidden = (w, w/2, w/4, w/8)
# with a 118-dimensional input; used only to report the overparameterisation
# ratio, not in any computation.
PARAMS = {8: 1.1e3, 16: 2.5e3, 32: 5.3e3, 64: 1.3e4, 128: 3.7e4,
          256: 1.18e5, 512: 4.3e5, 1024: 1.50e6, 2048: 5.6e6}
N_TRAIN = 82252


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


def stat(groups, name, metric):
    vals = [float(r[metric]) for r in groups[name]]
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def run_name(loss, lam, width, base_width):
    suffix = "" if width == base_width else f"_w{width}"
    return f"{loss}_{lam}{suffix}"


def fmt(groups, name):
    out = []
    for metric in METRICS:
        m, s, _ = stat(groups, name, metric)
        out.append(f"{m:.4f}+-{s:.4f}")
    return "  ".join(out)


def header(title):
    print(f"\n{title}")
    print(f"  {'run':<30}{'F1':>16}{'Hd_MAE':>16}{'xi_rms':>16}{'Hf_MAE':>16}")


def section_lam(groups, width, lams):
    for loss in ["iiLoss_finetune", "HdLoss_finetune"]:
        control = run_name(loss, "0.0", width, width if width != 1024 else 1024)
        control = f"{loss}_0.0" + ("" if width == 1024 else f"_w{width}")
        if control not in groups:
            continue
        f0 = stat(groups, control, "F1")[0]
        header(f"{loss}  (width {width}, {stat(groups, control, 'F1')[2]} seeds)")
        print(f"  {'lam=0.0 (control)':<30}{fmt(groups, control)}")
        for lam in lams:
            name = f"{loss}_{lam}" + ("" if width == 1024 else f"_w{width}")
            if name not in groups:
                continue
            rel = 100 * (stat(groups, name, "F1")[0] - f0) / f0
            print(f"  {'lam=' + lam:<30}{fmt(groups, name)}   {rel:+.1f}%")


def section_surgery(groups, width, lam):
    header(f"GRADIENT HANDLING  (width {width}, lam {lam})")
    labels = [("iiLoss_finetune", "no surgery"),
              ("iiLoss_pcgrad", "PCGrad (unconditional)"),
              ("iiLoss_pcgradc", "PCGrad (conditional)")]
    for prefix, label in labels:
        name = f"{prefix}_{lam}" + ("" if width == 1024 else f"_w{width}")
        if name not in groups:
            print(f"  {label:<30}(not run)")
            continue
        print(f"  {label:<30}{fmt(groups, name)}")


def section_capacity(groups, lams):
    header("CAPACITY LADDER  (lam=0, no penalty)")
    for w in sorted(PARAMS):
        name = "iiLoss_finetune_0.0" + ("" if w == 1024 else f"_w{w}")
        if name not in groups:
            continue
        ratio = PARAMS[w] / N_TRAIN
        print(f"  {'w=' + str(w) + f'  (ratio {ratio:.2f})':<30}{fmt(groups, name)}")

    print("\n  best relative F1 gain per width")
    print(f"  {'width':>8}{'ratio':>9}{'best lam':>10}{'rel %':>9}{'xi at best':>12}")
    for w in sorted(PARAMS):
        base = "iiLoss_finetune_0.0" + ("" if w == 1024 else f"_w{w}")
        if base not in groups:
            continue
        f0 = stat(groups, base, "F1")[0]
        cands = []
        for lam in lams:
            name = f"iiLoss_finetune_{lam}" + ("" if w == 1024 else f"_w{w}")
            if name in groups:
                cands.append((lam, name))
        if not cands:
            continue
        lam, name = max(cands, key=lambda t: stat(groups, t[1], "F1")[0])
        gain = 100 * (stat(groups, name, "F1")[0] - f0) / f0
        print(f"  {w:>8}{PARAMS[w] / N_TRAIN:>9.2f}{lam:>10}{gain:>+9.1f}"
              f"{stat(groups, name, 'rms_xi')[0]:>12.4f}")


def section_matched(groups, lams, base_width):
    """Pair each iiLoss run with the capacity model of most similar Hf_MAE.

    This is the control for the objection that the gain comes from degraded
    formation-enthalpy accuracy rather than from error cancellation.
    """
    ladder = []
    for w in sorted(PARAMS):
        name = "iiLoss_finetune_0.0" + ("" if w == 1024 else f"_w{w}")
        if name in groups:
            ladder.append((w, name, stat(groups, name, "Hf_MAE")[0]))
    if not ladder:
        return
    header("MATCHED-ACCURACY CONTROL")
    for lam in lams:
        name = f"iiLoss_finetune_{lam}" + ("" if base_width == 1024
                                           else f"_w{base_width}")
        if name not in groups:
            continue
        hf = stat(groups, name, "Hf_MAE")[0]
        w, near, _ = min(ladder, key=lambda t: abs(t[2] - hf))
        print(f"  --- iiLoss lam={lam} vs nearest capacity model ---")
        print(f"  {'capacity w=' + str(w):<30}{fmt(groups, near)}")
        print(f"  {'iiLoss lam=' + lam:<30}{fmt(groups, name)}")


def section_table2(groups, width, lam):
    header(f"TABLE 2 CANDIDATE ROWS  (width {width}, lam {lam})")
    rows = [("baseline (matched control)", "iiLoss_finetune_0.0"),
            ("iiLoss", f"iiLoss_finetune_{lam}"),
            ("HdLoss", f"HdLoss_finetune_{lam}"),
            ("iiLoss + PCGrad", f"iiLoss_pcgrad_{lam}"),
            ("HdLoss + PCGrad", f"HdLoss_pcgrad_{lam}")]
    for label, stem in rows:
        name = stem + ("" if width == 1024 else f"_w{width}")
        if name not in groups:
            continue
        print(f"  {label:<30}{fmt(groups, name)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="2026")
    parser.add_argument("--width", type=int, default=1024,
                        help="base width for the lam, surgery and table2 sections")
    parser.add_argument("--lam", default="0.2",
                        help="operating point for the surgery and table2 sections")
    parser.add_argument("--section", default="all",
                        choices=["all", "lam", "surgery", "capacity",
                                 "matched", "table2"])
    args = parser.parse_args()

    lams = ["0.1", "0.2", "0.3", "0.4", "0.5"]
    groups = load(args.dataset)
    print(f"{len(groups)} run groups in results/{args.dataset}/")

    if args.section in ("all", "lam"):
        section_lam(groups, args.width, lams)
    if args.section in ("all", "surgery"):
        section_surgery(groups, args.width, args.lam)
    if args.section in ("all", "capacity"):
        section_capacity(groups, lams)
    if args.section in ("all", "matched"):
        section_matched(groups, lams, args.width)
    if args.section in ("all", "table2"):
        section_table2(groups, args.width, args.lam)
    print()


if __name__ == "__main__":
    main()
