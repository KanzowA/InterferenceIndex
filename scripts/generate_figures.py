"""Regenerate the paper figures from pre-computed results into figures/.

Several figures read files that are too large to track in git. Those inputs
are listed per figure below, so a missing one is reported with the command
that produces it rather than as a traceback from the figure script.

Usage
-----
    python scripts/generate_figures.py            # main text and SI
    python scripts/generate_figures.py --fig 4    # one figure
    python scripts/generate_figures.py --fig S1   # supporting figure
    python scripts/generate_figures.py --si       # supporting figures only
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# -- Inputs that are not tracked in git -----------------------------------
# Each entry maps a path, or a glob, to the command that regenerates it.
SCORES_2020 = ("results/2020/interference_scores_2020.csv",
               "python scripts/interference_score.py 2020")
SCORES_2026 = ("results/2026/interference_scores_2026.csv",
               "python scripts/interference_score.py 2026")
CURVES = ("data/2026/ml/Hf/*_finetune_*_s*/training_curve.csv",
          "bash scripts/sweep.sh Hf 0 1 2   (or restore data/ from Zenodo)")

# -- Figures ---------------------------------------------------------------
# (key, label, script, extra arguments, required inputs)
FIGURES = [
    ("1", "Interference circles (conceptual)",
     "scripts/figures/fig1.py", [], []),

    ("2", "Model comparison - interference scores on allMP",
     "scripts/figures/fig2.py",
     ["--csv", "results/2020/interference_scores_2020.csv"],
     [SCORES_2020]),

    ("3", "Gradient field panel",
     "scripts/figures/fig3.py", [], []),

    ("4", "Lambda sweep - iiLoss / HdLoss / PCGrad",
     "scripts/figures/fig4.py",
     ["--csv", "results/2026/interference_summary_2026.csv"], []),

    ("5", "Interference index before and after training",
     "scripts/figures/fig5.py", [], [SCORES_2026]),

    ("S1", "Convergence of the fine-tuning stage - iiLoss",
     "scripts/figures/figS1.py", [], [CURVES]),

    ("S2", "Convergence of the fine-tuning stage - HdLoss",
     "scripts/figures/figS2.py", [], [CURVES]),

    ("S3", "Capacity dependence of the regularisation",
     "scripts/figures/figS3.py", [], []),
]


def missing_inputs(requires):
    """Inputs that are absent. Entries may be plain paths or globs."""
    absent = []
    for pattern, hint in requires:
        if not glob.glob(str(REPO_ROOT / pattern)):
            absent.append((pattern, hint))
    return absent


def run(keys):
    ok, failed, skipped = [], [], []

    for key, label, script, extra, requires in FIGURES:
        if keys is not None and key not in keys:
            continue

        if not (REPO_ROOT / script).exists():
            print(f"\n[Fig. {key}] SKIP - {script} not found")
            skipped.append(key)
            continue

        absent = missing_inputs(requires)
        if absent:
            print(f"\n[Fig. {key}] SKIP - required input not present")
            for pattern, hint in absent:
                print(f"           {pattern}")
                print(f"           regenerate with:  {hint}")
            skipped.append(key)
            continue

        print(f"\n{'=' * 60}")
        print(f"  Fig. {key} - {label}")
        print(f"{'=' * 60}", flush=True)   # before the child writes

        cmd = [sys.executable, str(REPO_ROOT / script)] + extra
        ret = subprocess.run(cmd, check=False, cwd=REPO_ROOT)

        if ret.returncode == 0:
            print("  [OK]")
            ok.append(key)
        else:
            print(f"  [FAILED] exit code {ret.returncode}")
            failed.append(key)

    print(f"\n{'=' * 60}")
    print(f"  Done: {len(ok)} OK  |  {len(failed)} failed  |  "
          f"{len(skipped)} skipped")
    if failed:
        print(f"  Failed:  {', '.join(failed)}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")
    print(f"{'=' * 60}")

    return len(failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", default=None,
                        help="one figure, for example 4 or S1 (default: all)")
    parser.add_argument("--si", action="store_true",
                        help="supporting figures only")
    args = parser.parse_args()

    known = [key for key, *_ in FIGURES]
    if args.fig is not None:
        key = args.fig.upper()
        if key not in known:
            raise SystemExit(f"unknown figure {args.fig!r}, "
                             f"choose from {', '.join(known)}")
        keys = {key}
    elif args.si:
        keys = {k for k in known if k.startswith("S")}
    else:
        keys = None

    sys.exit(run(keys))


if __name__ == "__main__":
    main()
