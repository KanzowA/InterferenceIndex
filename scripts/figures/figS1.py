"""Figures S1 and S2: convergence of the fine-tuning stage.

Both figures use the training curves of the prospective test, in which one
model per seed is fine-tuned on all compounds of the 2025 MP snapshot with
lambda fixed in advance. Curves are means over the five seeds, shaded bands
the sample standard deviation.

Figure S1 (iiLoss, lambda = 0.2)
    (a) MSE(dfH) / Var(dfH_DFT) for the control, iiLoss and the naive penalty
    (b) interference term relative to its value for the initial model
    (c) gauge share of the training residual for iiLoss

Figure S2 (HdLoss, lambda = 0.1)
    (a) MSE(dfH) / Var(dfH_DFT) for the control and HdLoss
    (b) MSE(dHd) relative to its value for the initial model

The curves are written by the prospective runs (scripts/prospective.sh) as
runs_wbm/curves/<loss>_<lam>_s<seed>.csv and are copied to data/wbm/curves.
File stems: mse_0 = control, iiperpfrozen_0.2 = iiLoss (gauge-free, frozen
scales), iiloss_0.2 = naive <xi^2> penalty, hdloss_0.1 = HdLoss.

Usage
-----
    python scripts/figures/figS1.py
    python scripts/figures/figS1.py --objective HdLoss
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
CURVE_DIR = os.path.join(REPO_ROOT, "data", "wbm", "curves")

# -- Figure style ---------------------------------------------------------
# Fonts are scaled so they print at 7 pt (body) and 6 pt (minor) once the
# figure is reduced to the final column width.
_COL_W    = 6.69          # target column width in inches
_FIG_W    = 13.0          # rendering width for font scaling
_FS       = round(7.0 * _FIG_W / _COL_W)   # body / axis labels
_FS_SM    = round(6.0 * _FIG_W / _COL_W)   # minor annotations
_FS_PANEL = 20            # panel labels a/b/c
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   _FS,
    "axes.linewidth": 0.8,
})

_COL_II  = "#E61D60"
_COL_HD  = "#1E46E4"
_COL_REF = "#000000"

# -- Series ---------------------------------------------------------------
# The naive penalty shares the colour of iiLoss with a lighter, dashed line,
# as in Figure S5.
SERIES = {
    "control": dict(stem="mse_0", color=_COL_REF, lw=1.2, ls="-", alpha=0.8,
                    label=r"control ($\lambda\!=\!0$)"),
    "iiLoss": dict(stem="iiperpfrozen_0.2", color=_COL_II, lw=1.8, ls="-",
                   alpha=1.0, label=r"iiLoss ($\lambda\!=\!0.2$)"),
    "naive": dict(stem="iiloss_0.2", color=_COL_II, lw=1.2, ls="--", alpha=0.55,
                  label=r"naive $\langle\xi^2\rangle$ ($\lambda\!=\!0.2$)"),
    "HdLoss": dict(stem="hdloss_0.1", color=_COL_HD, lw=1.8, ls="-", alpha=1.0,
                   label=r"HdLoss ($\lambda\!=\!0.1$)"),
}

# objective -> (series in panel a, series in panel b, ylabel b, gauge panel)
LAYOUT = {
    "iiLoss": (["control", "iiLoss", "naive"], ["iiLoss", "naive"],
               r"$L_\xi\,/\,L_{\xi,0}$", True),
    "HdLoss": (["control", "HdLoss"], ["HdLoss"],
               r"$\mathrm{MSE}(\Delta_\mathrm{d}H)\,/\,\mathrm{MSE}_0(\Delta_\mathrm{d}H)$",
               False),
}

_MSE_LABEL = (r"$\mathrm{MSE}(\Delta_\mathrm{f}H)\,/\,"
              r"\mathrm{Var}(\Delta_\mathrm{f}H^\mathrm{DFT})$")


# -- Helpers --------------------------------------------------------------
def load(stem, curve_dir):
    """Per-epoch mean and standard deviation over the seeds of one run."""
    paths = sorted(glob.glob(os.path.join(curve_dir, f"{stem}_s*.csv")))
    if not paths:
        raise FileNotFoundError(f"no curves for {stem} in {curve_dir}")
    df = pd.concat([pd.read_csv(p) for p in paths])
    g = df.groupby("epoch")
    return g.mean(numeric_only=True), g.std(ddof=1, numeric_only=True), len(paths)


def draw(ax, curves, keys, column):
    for key in keys:
        cfg = SERIES[key]
        mean, sd, _ = curves[key]
        x = mean.index.values
        y, s = mean[column].values, sd[column].values
        ax.fill_between(x, y - s, y + s, color=cfg["color"],
                        alpha=0.15 * cfg["alpha"], lw=0)
        ax.plot(x, y, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
                alpha=cfg["alpha"], label=cfg["label"])
    ax.set_xlabel("epoch", fontsize=_FS)
    ax.set_xlim(0, x.max())
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.tick_params(labelsize=_FS_SM)


# -- Figure ---------------------------------------------------------------
def build_figure(objective, curve_dir):
    keys_a, keys_b, ylabel_b, gauge = LAYOUT[objective]
    curves = {k: load(SERIES[k]["stem"], curve_dir) for k in set(keys_a + keys_b)}

    ncol = 3 if gauge else 2
    fig, axes = plt.subplots(1, ncol, figsize=(_FIG_W, 4.4),
                             gridspec_kw=dict(wspace=0.32))

    draw(axes[0], curves, keys_a, "mse_over_ref")
    axes[0].set_ylabel(_MSE_LABEL, fontsize=_FS)
    if objective == "iiLoss":
        axes[0].set_yscale("log")
        axes[0].set_ylim(0.02, 0.11)
        axes[0].yaxis.set_major_locator(ticker.FixedLocator([0.02, 0.03, 0.05, 0.1]))
        axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())

    draw(axes[1], curves, keys_b, "penalty_over_ref")
    axes[1].set_ylabel(ylabel_b, fontsize=_FS)
    axes[1].set_ylim(0, 1.05)

    if gauge:
        mean, sd, _ = curves["iiLoss"]
        x = mean.index.values
        y, s = 100 * mean["gauge_share"].values, 100 * sd["gauge_share"].values
        axes[2].fill_between(x, y - s, y + s, color=_COL_II, alpha=0.15, lw=0)
        axes[2].plot(x, y, color=_COL_II, lw=1.8)
        axes[2].set_ylabel("gauge share of the residual / %", fontsize=_FS)
        axes[2].set_xlabel("epoch", fontsize=_FS)
        axes[2].set_xlim(0, x.max())
        axes[2].set_ylim(0, None)
        axes[2].grid(True, lw=0.4, alpha=0.5)
        axes[2].tick_params(labelsize=_FS_SM)

    handles, labels = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    fig.legend([handles[l] for l in labels], labels, loc="lower center",
               ncol=len(labels), fontsize=_FS_SM, frameon=True,
               bbox_to_anchor=(0.5, -0.12))

    for ax, letter in zip(axes, "abc"):
        ax.text(-0.17, 1.03, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=list(LAYOUT), default="iiLoss")
    parser.add_argument("--curves", default=CURVE_DIR)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or os.path.join(
        REPO_ROOT, "figures",
        "figureS1.png" if args.objective == "iiLoss" else "figureS2.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build_figure(args.objective, args.curves)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {out}")


if __name__ == "__main__":
    main()
