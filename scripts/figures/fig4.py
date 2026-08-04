"""Figure 4: metrics against regularisation strength lam, one line per model.

    (a) Hf MAE   (b) Hd MAE   (c) F1   (d) xi_rms

Runs are grouped by name with the _s<seed> suffix removed; points are the mean
over seeds and error bars the sample standard deviation. Each objective is
shown with and without gradient surgery, the surgery variant in the same colour
at lower weight and opacity.
"""

import argparse
import os
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# -- Journal style (npj Computational Materials) --------------------------
# Fonts are scaled so they print at 7 pt (body) and 6 pt (minor) once the
# figure is reduced to a 170 mm double column.
_COL_W    = 6.69          # 170 mm in inches
_FIG_W    = 12.0          # rendering width for font scaling
_FS       = round(7.0 * _FIG_W / _COL_W)   # body / axis labels
_FS_SM    = round(6.0 * _FIG_W / _COL_W)   # minor annotations
_FS_LEG   = _FS
_FS_PANEL = 20            # panel labels a/b/c/d
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   _FS,
    "axes.linewidth": 0.8,
})

_COL_II  = "#E61D60"
_COL_HD  = "#1E46E4"
_COL_REF = "#000000"

# -- Series and panels ----------------------------------------------------
# The matched lam=0 control shares the two-stage schedule of every regularised
# run, so it anchors each series at lam=0 and sets the baseline line.
BASELINE = "iiLoss_finetune_0.0"

# Surgery variants share the colour of their objective and are distinguished by
# weight and transparency alone, so each pair reads as one comparison.
_ALPHA_PCGRAD = 0.45

SERIES = {
    "iiLoss": dict(prefix="iiLoss_finetune_", color=_COL_II,
                   marker="s", lw=1.8, ms=5, alpha=1.0, fill=True),
    "iiLoss + PCGrad": dict(prefix="iiLoss_pcgrad_", color=_COL_II,
                            marker="s", lw=1.0, ms=4, alpha=_ALPHA_PCGRAD,
                            fill=False),
    "HdLoss": dict(prefix="HdLoss_finetune_", color=_COL_HD,
                   marker="o", lw=1.8, ms=5, alpha=1.0, fill=True),
    "HdLoss + PCGrad": dict(prefix="HdLoss_pcgrad_", color=_COL_HD,
                            marker="o", lw=1.0, ms=4, alpha=_ALPHA_PCGRAD,
                            fill=False),
}

METRICS = [
    ("Hf_MAE", r"$\mathrm{MAE}\left(\Delta_\mathrm{f}H\right)$ / eV atom$^{-1}$"),
    ("Hd_MAE", r"$\mathrm{MAE}\left(\Delta_\mathrm{d}H\right)$ / eV atom$^{-1}$"),
    ("F1",     r"$F_1$"),
    ("rms_xi", r"$\xi_\mathrm{rms}$"),
]

LAM_MAX  = 0.9
LAM_MARK = 0.2


# -- Helpers --------------------------------------------------------------
def _group_by_stem(df):
    """Map each run name without its _s<seed> suffix to the matching rows."""
    groups = {}
    for _, row in df.iterrows():
        stem, _, tail = str(row["model"]).rpartition("_s")
        if stem and tail.isdigit():
            groups.setdefault(stem, []).append(row)
    return groups


def _stat(rows, col):
    vals = [float(r[col]) for r in rows]
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def _series_points(groups, prefix, col):
    """Return sorted (lam, mean, sd) for runs matching the prefix.

    The remainder after the prefix must parse as a float, which excludes the
    _w<width>, eps<value> and pcgradc variants sharing the same stem.
    """
    points = []
    for stem, rows in groups.items():
        if not stem.startswith(prefix):
            continue
        try:
            lam = float(stem[len(prefix):])
        except ValueError:
            continue
        if 0.0 < lam <= LAM_MAX:
            m, s = _stat(rows, col)
            points.append((lam, m, s))
    return sorted(points)


# -- Figure ---------------------------------------------------------------
def build_figure(df):
    groups = _group_by_stem(df)
    if BASELINE not in groups:
        raise SystemExit(f"{BASELINE} not found in the summary file")

    series = SERIES

    fig, axes_2d = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.42)
    axes = [axes_2d[0, 0], axes_2d[0, 1], axes_2d[1, 0], axes_2d[1, 1]]

    for ax_idx, (ax, (col, ylabel)) in enumerate(zip(axes, METRICS)):
        base_m, base_s = _stat(groups[BASELINE], col)

        for label, cfg in series.items():
            points = _series_points(groups, cfg["prefix"], col)
            if not points:
                continue
            lam = [0.0]    + [p[0] for p in points]
            val = [base_m] + [p[1] for p in points]
            err = [base_s] + [p[2] for p in points]
            # Filled markers without surgery, open with, so the pairing also
            # survives greyscale reproduction.
            filled = cfg["fill"]
            ax.errorbar(lam, val, yerr=err,
                        color=cfg["color"], ls="-", lw=cfg["lw"],
                        alpha=cfg["alpha"], marker=cfg["marker"],
                        markersize=cfg["ms"],
                        markerfacecolor=(cfg["color"] if filled else "white"),
                        markeredgecolor=("white" if filled else cfg["color"]),
                        markeredgewidth=(0.5 if filled else 1.0),
                        capsize=2, elinewidth=0.8,
                        zorder=(3 if filled else 2), label=label)

        ax.axvline(LAM_MARK, color=_COL_REF, ls="--", lw=0.8, alpha=0.4,
                   zorder=1, label=r"$\lambda=%.1f$" % LAM_MARK)
        ax.axhline(base_m, color=_COL_REF, ls="-", lw=1.0, alpha=0.8,
                   zorder=1, label=r"baseline ($\lambda\!=\!0$)")
        if col == "rms_xi":
            ax.axhline(1.0, color=_COL_REF, ls=":", lw=0.9, alpha=0.6,
                       zorder=1, label=r"$\xi_\mathrm{rms}=1$")

        if ax_idx // 2 == 1:
            ax.set_xlabel(r"$\lambda$", fontsize=_FS)
        else:
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(labelsize=_FS)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(True, which="major", lw=0.4, alpha=0.5)
        ax.grid(True, which="minor", lw=0.2, alpha=0.3)

    # -- Shared legend below the panels ------------------------------------
    handles = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)

    # Column-major fill, so each objective sits above its PCGrad variant and
    # the reference lines occupy the last two columns.
    order = ["iiLoss", "iiLoss + PCGrad",
             "HdLoss", "HdLoss + PCGrad",
             r"baseline ($\lambda\!=\!0$)", r"$\lambda=%.1f$" % LAM_MARK,
             r"$\xi_\mathrm{rms}=1$"]
    ordered = [(l, handles[l]) for l in order if l in handles]
    fig.legend([h for _, h in ordered], [l for l, _ in ordered],
               loc="lower center", ncol=4, fontsize=_FS_LEG,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.14))

    for ax, letter in zip(axes, ["a", "b", "c", "d"]):
        ax.text(-0.22, 1.05, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(
        REPO_ROOT, "results", "2026", "interference_summary_2026.csv"))
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figure4.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(df)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
