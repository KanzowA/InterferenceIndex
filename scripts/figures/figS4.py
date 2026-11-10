"""Figure S4: the interference diagnostic for the Matbench Discovery models.

    (a) gauge-fixed index Xi2_Q against MAE(dfH), with 95 % intervals
    (b) MAE(dH_d) / MAE(dfH), the factor by which a formation-energy error
        turns into a decomposition-energy error, against MAE(dfH)
    (c) the same factor against Xi2_Q, for the potentials

On the closed WBM hull of data/wbm/hullout_wbm.json, from
results/wbm/mlip_closed.csv (scripts/mlip_score.py). Potentials trained on MP
data alone cancel their errors (Xi2_Q < 1), potentials trained on more data do
not, at the same formation-energy accuracy. The correlation in (c) is a shift
between the two groups rather than a model-by-model relation, and is reported
in the caption, not on the panel.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

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

_COL_MP  = "#1E46E4"
_COL_EXT = "#7A1FA2"
_COL_OLD = "#6B7280"
_COL_REF = "#000000"

# -- Series ---------------------------------------------------------------
GROUPS = {
    "MP-only MLIP": dict(color=_COL_MP, marker="o", fill=True,
                         label="potential, MP data only"),
    "extended-data MLIP": dict(color=_COL_EXT, marker="s", fill=True,
                               label="potential, extended training data"),
    "pre-MLIP": dict(color=_COL_OLD, marker="^", fill=False,
                     label="earlier energy model"),
}

# A few models named in panel (b): label, offset in points, alignment.
NAMED = {"mace-mp-0": ("MACE-MP-0", (-6, -22), "right"),
         "chgnet-0.3.0": ("CHGNet", (8, 6), "left"),
         "mattersim-v1-5m": ("MatterSim", (16, -10), "left"),
         "eqv2-m-omat-salex-mp": ("eqV2-M", (6, 22), "left"),
         "dpa-3.1-mptrj": ("DPA-3.1", (-8, -20), "right")}

MAE_TICKS = [0.02, 0.05, 0.1]
MAE_LIM = (0.014, 0.16)


# -- Helpers --------------------------------------------------------------
def scatter(ax, df, x, y, interval=None):
    for group, cfg in GROUPS.items():
        sub = df[df.group == group]
        if interval is not None:
            ax.errorbar(sub[x], sub[y], yerr=[sub[y] - sub[interval[0]], sub[interval[1]] - sub[y]],
                        fmt="none", ecolor=cfg["color"], elinewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter(sub[x], sub[y], s=40, marker=cfg["marker"],
                   facecolor=cfg["color"] if cfg["fill"] else "white",
                   edgecolor="white" if cfg["fill"] else cfg["color"],
                   linewidth=0.6 if cfg["fill"] else 1.0, zorder=3, label=cfg["label"])


def mae_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(MAE_TICKS)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(*MAE_LIM)
    ax.set_xlabel(r"$\mathrm{MAE}\left(\Delta_\mathrm{f}H\right)$ / eV atom$^{-1}$", fontsize=_FS)


# -- Figure ---------------------------------------------------------------
def build_figure(df):
    df = df.assign(ratio=df.MAE_d_closed / df.MAE_f)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw=dict(wspace=0.36))

    ax = axes[0]
    scatter(ax, df, "MAE_f", "Xi2_Q", interval=("Xi2_Q_lo", "Xi2_Q_hi"))
    ax.axhline(1.0, color=_COL_REF, ls=":", lw=0.9, alpha=0.6, zorder=1)
    ax.text(0.98, 0.04, "errors cancel", transform=ax.transAxes, fontsize=_FS_SM,
            style="italic", ha="right", va="bottom")
    ax.text(0.02, 0.96, "errors accumulate", transform=ax.transAxes, fontsize=_FS_SM,
            style="italic", ha="left", va="top")
    mae_axis(ax)
    ax.set_ylabel(r"gauge-fixed index $\Xi^2_Q$", fontsize=_FS)

    ax = axes[1]
    scatter(ax, df, "MAE_f", "ratio")
    mae_axis(ax)
    ax.set_ylabel(r"$\mathrm{MAE}(\Delta_\mathrm{d}H)\,/\,\mathrm{MAE}(\Delta_\mathrm{f}H)$", fontsize=_FS)
    for key, (text, offset, ha) in NAMED.items():
        row = df[df.model == key]
        if len(row):
            ax.annotate(text, (row.MAE_f.iloc[0], row.ratio.iloc[0]), xytext=offset,
                        textcoords="offset points", fontsize=_FS_SM, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=1, shrinkB=3))

    ax = axes[2]
    scatter(ax, df[df.group != "pre-MLIP"], "Xi2_Q", "ratio")
    ax.axvline(1.0, color=_COL_REF, ls=":", lw=0.9, alpha=0.6, zorder=1)
    ax.set_xlabel(r"gauge-fixed index $\Xi^2_Q$", fontsize=_FS)
    ax.set_ylabel(r"$\mathrm{MAE}(\Delta_\mathrm{d}H)\,/\,\mathrm{MAE}(\Delta_\mathrm{f}H)$", fontsize=_FS)

    for i, ax in enumerate(axes):
        ax.grid(True, which="major", lw=0.4, alpha=0.5)
        ax.tick_params(labelsize=_FS_SM)
        ax.text(-0.24, 1.04, "abc"[i], transform=ax.transAxes, fontsize=_FS_PANEL,
                fontweight="bold", va="bottom", ha="left", clip_on=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=_FS,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.12))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(
        REPO_ROOT, "results", "wbm", "mlip_closed.csv"))
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figureS4.png"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(pd.read_csv(args.csv))
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
