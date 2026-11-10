"""Figure S3: dependence of interference-aware training on capacity.

    (a) MAE(dfH)   (b) MAE(dH_d)   (c) F1   (d) ROC-AUC

each as the change against the lam = 0 control of the same base width, paired
by seed, for iiLoss and HdLoss with lam chosen on the validation split. Points
are means over seeds, error bars the sample standard deviation. Width 1024 is
the nested run, restricted to the seeds of the smaller widths.

The gain grows with width, and from base width 256 upwards iiLoss improves the
formation enthalpies as well; HdLoss pays for its gain in dfH at every width.

Usage
-----
    python scripts/figures/figS3.py
    python scripts/figures/figS3.py --runs iiLoss_finetune_0.2 HdLoss_finetune_0.1
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# -- Figure style ---------------------------------------------------------
# Fonts are scaled so they print at 7 pt (body) and 6 pt (minor) once the
# figure is reduced to the final column width.
_COL_W    = 6.69          # target column width in inches
_FIG_W    = 15.0          # rendering width for font scaling
_FS       = round(7.0 * _FIG_W / _COL_W)   # body / axis labels
_FS_SM    = round(6.0 * _FIG_W / _COL_W)   # minor annotations
_FS_CB    = _FS
_FS_PANEL = 20            # panel labels a-d
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   _FS,
    "axes.linewidth": 0.8,
})

_COL_II  = "#E61D60"
_COL_HD  = "#1E46E4"
_COL_REF = "#000000"

# -- Series and panels ----------------------------------------------------
SERIES = {
    "iiLoss_nested": dict(label="iiLoss", color=_COL_II, marker="s"),
    "HdLoss_nested": dict(label="HdLoss", color=_COL_HD, marker="o"),
    "iiLoss_finetune_0.2": dict(label=r"iiLoss ($\lambda=0.2$)", color=_COL_II, marker="s"),
    "HdLoss_finetune_0.1": dict(label=r"HdLoss ($\lambda=0.1$)", color=_COL_HD, marker="o"),
}
CONTROL = "iiLoss_finetune_0.0"
DEFAULT_WIDTHS = [32, 64, 128, 256, 512, 1024]
DEFAULT_BASE_WIDTH = 1024

METRICS = [
    ("MAE_f",   r"$\Delta\mathrm{MAE}\left(\Delta_\mathrm{f}H\right)$ / %"),
    ("MAE_d",   r"$\Delta\mathrm{MAE}\left(\Delta_\mathrm{d}H\right)$ / %"),
    ("F1",      r"$\Delta F_1$ / %"),
    ("ROC_AUC", r"$\Delta$ROC-AUC / %"),
]


# -- Helpers --------------------------------------------------------------
def _split(df):
    parts = df["model"].str.extract(r"^(.*)_s(\d+)$")
    return df.assign(stem=parts[0], seed=parts[1].astype(int))


def change_points(df, run, widths, col):
    """(width, mean, sd) of the change against the control, paired by seed."""
    seeds = sorted(df[df.stem == f"{CONTROL}_w{min(widths)}"].seed.unique())
    points = []
    for w in widths:
        tag = "" if w == DEFAULT_BASE_WIDTH else f"_w{w}"
        c = df[(df.stem == CONTROL + tag) & df.seed.isin(seeds)].set_index("seed")[col]
        a = df[(df.stem == run + tag) & df.seed.isin(seeds)].set_index("seed")[col]
        if a.empty or c.empty:
            continue
        rel = 100 * (a.loc[c.index] / c - 1)
        points.append((w, rel.mean(), rel.std(ddof=1)))
    return points


# -- Figure ---------------------------------------------------------------
def build_figure(df, runs, widths):
    df = _split(df)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9),
                             gridspec_kw=dict(wspace=0.42))

    for ax_idx, (ax, (col, ylabel)) in enumerate(zip(axes, METRICS)):
        for run in runs:
            cfg = SERIES[run]
            points = change_points(df, run, widths, col)
            if not points:
                continue
            ax.errorbar([p[0] for p in points], [p[1] for p in points],
                        yerr=[p[2] for p in points], color=cfg["color"], lw=1.6,
                        marker=cfg["marker"], markersize=5, markeredgecolor="white",
                        markeredgewidth=0.5, capsize=2, elinewidth=0.8,
                        label=cfg["label"])
        ax.axhline(0.0, color=_COL_REF, ls="-", lw=1.0, alpha=0.8, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths], fontsize=_FS_SM, rotation=45)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel("base width", fontsize=_FS)
        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, which="major", lw=0.4, alpha=0.5)
        ax.text(-0.3, 1.04, "abcd"[ax_idx], transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=_FS, framealpha=0.9, bbox_to_anchor=(0.5, -0.27))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(
        REPO_ROOT, "results", "2025", "spectral_summary_2025.csv"))
    parser.add_argument("--runs", nargs="+", default=["iiLoss_nested", "HdLoss_nested"],
                        choices=list(SERIES))
    parser.add_argument("--widths", nargs="+", type=int, default=DEFAULT_WIDTHS)
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figureS3.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(df, args.runs, args.widths)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
