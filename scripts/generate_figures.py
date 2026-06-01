"""
generate_figures.py
-------------------
Reproduces all paper figures from pre-computed results.
Outputs are written to figures/.

Usage:
    python generate_figures.py           # generate all figures
    python generate_figures.py --fig 6   # generate a specific figure
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Figure registry ────────────────────────────────────────────────────────────
# Each entry: (figure number, label, script, extra args)
FIGURES = [
    (1, "Geometry of the interference index",
        "scripts/figures/fig1_geometry.py", []),

    (2, "Interference circles (conceptual)",
        "scripts/figures/fig2_circles.py", []),

    (3, "Model comparison — interference scores on allMP",
        "scripts/figures/fig3_model_comparison.py",
        ["--csv", "results/2020/interference_scores_2020.csv"]),

    # (4, "Bartel model summary graphic",
    #     "scripts/figures/fig4_bartel_summary.py", []),        # TODO

    (5, "Gradient field panel (MSE vs ξ²)",
        "scripts/figures/fig5_gradient_fields.py", []),

    (6, "Lambda sweep — iiLoss / HdLoss / PCGrad",
        "scripts/figures/fig6_lambda_sweep.py",
        ["--csv", "results/2026/interference_summary_2026.csv"]),

    # (7, "ξ distribution evolution",
    #     "scripts/figures/fig7_xi_distribution.py", []),       # TODO
]


def run(fig_num=None):
    ok, failed, skipped = [], [], []

    for num, label, script, extra in FIGURES:
        if fig_num is not None and num != fig_num:
            continue

        path = Path(script)
        if not path.exists():
            print(f"\n[Fig. {num}] SKIP — {script} not found")
            skipped.append(num)
            continue

        print(f"\n{'='*60}")
        print(f"  Fig. {num} — {label}")
        print(f"{'='*60}")

        cmd = [sys.executable, script] + extra
        ret = subprocess.run(cmd, check=False)

        if ret.returncode == 0:
            print(f"  [OK]")
            ok.append(num)
        else:
            print(f"  [FAILED] exit code {ret.returncode}")
            failed.append(num)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Done: {len(ok)} OK  |  {len(failed)} failed  |  {len(skipped)} skipped")
    if failed:
        print(f"  Failed figures: {failed}")
    if skipped:
        print(f"  Skipped (script missing): {skipped}")
    print(f"{'='*60}")

    return len(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=int, default=None,
                        help="Generate only this figure number (default: all)")
    args = parser.parse_args()
    sys.exit(run(args.fig))
