"""Figure S3: dependence of interference-aware regularisation on capacity.

    (a) F1 against lam, one line per base width
    (b) xi_rms against lam, one line per base width

The diagnostic and the regularisation respond to capacity in opposite ways:
xi_rms at lam = 0 is flat across the ladder, while the gain from regularisation
grows with width.

Usage
-----
    python scripts/figures/figS3.py
    python scripts/figures/figS3.py --widths 32 64 128 256 1024
    python scripts/figures/figS3.py --prefix HdLoss_finetune_
"""

import argparse
import os
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# -- Figure style ---------------------------------------------------------
# Fonts are scaled so they print at 7 pt (body) and 6 pt (minor) once the
# figure is reduced to the final column width.
_COL_W    = 6.69          # target column width in inches
_FIG_W    = 12.0          # rendering width for font scaling
_FS       = round(7.0 * _FIG_W / _COL_W)   # body / axis labels
_FS_SM    = round(6.0 * _FIG_W / _COL_W)   # minor annotations
_FS_CB    = _FS
_FS_PANEL = 20            # panel labels a/b
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   _FS,
    "axes.linewidth": 0.8,
})

_COL_II  = "#E61D60"
_COL_HD  = "#1E46E4"
_COL_MID = "#7A1FA2"
_COL_REF = "#000000"

# -- Series and panels ----------------------------------------------------
# Width is mapped onto the two objective colours of fig4 through purple, as in
# figS1, so the supporting figures share one scale.
_CMAP = LinearSegmentedColormap.from_list(
    "width", [_COL_HD, _COL_MID, _COL_II])

DEFAULT_WIDTHS = [32, 64, 128, 256, 1024]
DEFAULT_BASE_WIDTH = 1024
LAM_MAX = 0.5

METRICS = [
    ("F1",     r"$F_1$"),
    ("rms_xi", r"$\xi_\mathrm{rms}$"),
]


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


def series_points(groups, prefix, width, col):
    """Sorted (lam, mean, sd) for one base width, lam = 0 included."""
    suffix = "" if width == DEFAULT_BASE_WIDTH else f"_w{width}"
    points = []
    for stem, rows in groups.items():
        if not stem.startswith(prefix) or not stem.endswith(suffix):
            continue
        tail = stem[len(prefix):len(stem) - len(suffix)] if suffix else \
            stem[len(prefix):]
        try:
            lam = float(tail)
        except ValueError:
            continue
        if lam <= LAM_MAX:
            m, s = _stat(rows, col)
            points.append((lam, m, s))
    return sorted(points)


# -- Figure ---------------------------------------------------------------
def build_figure(df, prefix, widths):
    groups = _group_by_stem(df)
    norm = LogNorm(vmin=min(widths), vmax=max(widths))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2),
                             gridspec_kw=dict(wspace=0.28))

    for ax_idx, (ax, (col, ylabel)) in enumerate(zip(axes, METRICS)):
        for width in widths:
            points = series_points(groups, prefix, width, col)
            if not points:
                continue
            ax.errorbar([p[0] for p in points], [p[1] for p in points],
                        yerr=[p[2] for p in points],
                        color=_CMAP(norm(width)), lw=1.6, marker="o",
                        markersize=4, markeredgecolor="white",
                        markeredgewidth=0.5, capsize=2, elinewidth=0.8)

        if col == "rms_xi":
            ax.axhline(1.0, color=_COL_REF, ls=":", lw=0.9, alpha=0.6,
                       zorder=1)

        ax.set_xlabel(r"$\lambda$", fontsize=_FS)
        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(labelsize=_FS)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, which="major", lw=0.4, alpha=0.5)

        ax.text(-0.18, 1.04, "ab"[ax_idx], transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)

    # -- Shared colourbar, in place of one legend entry per width ----------
    scalar_map = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
    scalar_map.set_array([])
    cbar = fig.colorbar(scalar_map, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("base width", fontsize=_FS_CB)
    cbar.set_ticks(widths)
    cbar.set_ticklabels([str(w) for w in widths])
    cbar.ax.tick_params(labelsize=_FS_SM)
    cbar.ax.minorticks_off()

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(
        REPO_ROOT, "results", "2026", "interference_summary_2026.csv"))
    parser.add_argument("--prefix", default="iiLoss_finetune_")
    parser.add_argument("--widths", nargs="+", type=int,
                        default=DEFAULT_WIDTHS)
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figureS3.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(df, args.prefix, args.widths)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
