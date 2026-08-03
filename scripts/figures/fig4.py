"""Figure 4: metrics against regularisation strength lam, one line per model.

    (a) Hf MAE   (b) Hd MAE   (c) F1   (d) xi_rms

Runs are grouped by name with the _s<seed> suffix removed; points are the mean
over seeds and error bars the sample standard deviation. Width-tagged and
eps-tagged runs are excluded, since the remainder after the series prefix must
parse as a single float.

Usage
-----
    python scripts/figures/fig4.py
    python scripts/figures/fig4.py --csv results/2026/interference_summary_2026.csv
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

_COL_W = 6.69
_FIG_W = 12.0
_FS       = round(7.0 * _FIG_W / _COL_W)
_FS_LEG   = _FS
_FS_PANEL = 20
plt.rcParams.update({"font.family": "sans-serif", "font.size": _FS,
                     "axes.linewidth": 0.8})

# The matched lam=0 control: same two-stage schedule as every regularised run,
# so it anchors each series at lam=0 and sets the baseline line.
BASELINE = "iiLoss_finetune_0.0"

# Solid without gradient surgery, dashed with it.
SERIES = {
    "iiLoss":          dict(prefix="iiLoss_finetune_", color="#E61D60",
                            marker="s", ls="-",  lw=1.8),
    "HdLoss":          dict(prefix="HdLoss_finetune_", color="#1EACE4",
                            marker="o", ls="-",  lw=1.8),
    "iiLoss + PCGrad": dict(prefix="iiLoss_pcgrad_",   color="#E61D60",
                            marker="s", ls="--", lw=1.8),
    "HdLoss + PCGrad": dict(prefix="HdLoss_pcgrad_",   color="#1EACE4",
                            marker="o", ls="--", lw=1.8),
}

LAM_MAX  = 0.8
LAM_MARK = 0.2

# Upper clip per panel. iiLoss without surgery leaves the useful range in the
# Hf panel; points above the clip are drawn as triangles on the top spine so
# the excursion stays visible without flattening every other series.
YMAX = {"Hf_MAE": 0.22, "Hd_MAE": 0.145}

METRICS = [
    ("Hf_MAE", r"$\mathrm{MAE}\left(\Delta_\mathrm{f}H\right)$ / eV atom$^{-1}$"),
    ("Hd_MAE", r"$\mathrm{MAE}\left(\Delta_\mathrm{d}H\right)$ / eV atom$^{-1}$"),
    ("F1",     r"$F_1$"),
    ("rms_xi", r"$\xi_\mathrm{rms}$"),
]


def group_by_stem(df):
    """Map run name without the _s<seed> suffix to its list of rows."""
    groups = {}
    for _, row in df.iterrows():
        stem, _, tail = str(row["model"]).rpartition("_s")
        if stem and tail.isdigit():
            groups.setdefault(stem, []).append(row)
    return groups


def stat(rows, col):
    vals = [float(r[col]) for r in rows]
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def series_points(groups, prefix, col):
    """Return (lam, mean, sd) for every run matching the prefix exactly.

    The remainder after the prefix must parse as a float, which drops the
    _w<width> and eps<value> variants sharing the same stem.
    """
    points = []
    for stem, rows in groups.items():
        if not stem.startswith(prefix):
            continue
        try:
            lam = float(stem[len(prefix):])
        except ValueError:
            continue
        if lam <= 0.0 or lam > LAM_MAX:
            continue
        m, s = stat(rows, col)
        points.append((lam, m, s))
    return sorted(points)


def build_figure(df):
    groups = group_by_stem(df)
    if BASELINE not in groups:
        raise SystemExit(f"{BASELINE} not found in the summary file")

    fig, axes_2d = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.42)
    axes = [axes_2d[0, 0], axes_2d[0, 1], axes_2d[1, 0], axes_2d[1, 1]]

    for ax_idx, (ax, (col, ylabel)) in enumerate(zip(axes, METRICS)):
        base_m, base_s = stat(groups[BASELINE], col)

        for label, cfg in SERIES.items():
            pts = series_points(groups, cfg["prefix"], col)
            if not pts:
                continue
            lam = [0.0] + [p[0] for p in pts]
            val = [base_m] + [p[1] for p in pts]
            err = [base_s] + [p[2] for p in pts]

            ymax = YMAX.get(col)
            if ymax is not None:
                inside = [v <= ymax for v in val]
                ax.scatter([x for x, ok in zip(lam, inside) if not ok],
                           [ymax * 0.995] * inside.count(False),
                           marker="^", s=55, color=cfg["color"],
                           zorder=6, clip_on=False)
                val = [v if ok else float("nan") for v, ok in zip(val, inside)]
                err = [e if ok else 0.0 for e, ok in zip(err, inside)]

            ax.errorbar(lam, val, yerr=err,
                        color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"],
                        marker=cfg["marker"], markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        capsize=2, elinewidth=0.8, zorder=3, label=label)

        ax.axvline(LAM_MARK, color="gray", ls="--", lw=0.8, alpha=0.4,
                   zorder=1, label=r"$\lambda=%.1f$" % LAM_MARK)
        ax.axhline(base_m, color="grey", lw=1.0, ls="-", alpha=0.8, zorder=1,
                   label=r"baseline ($\lambda\!=\!0$)")
        if col == "rms_xi":
            ax.axhline(1.0, color="#6B7280", lw=0.9, ls=":", alpha=0.6,
                       zorder=1, label=r"$\xi_\mathrm{rms}=1$")

        if ax_idx // 2 == 1:
            ax.set_xlabel(r"$\lambda$", fontsize=_FS)
        else:
            ax.tick_params(labelbottom=False)

        if col in YMAX:
            ax.set_ylim(top=YMAX[col])

        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(labelsize=_FS)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(True, which="major", lw=0.4, alpha=0.5)
        ax.grid(True, which="minor", lw=0.2, alpha=0.3)

    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)

    order = list(SERIES) + [r"$\lambda=%.1f$" % LAM_MARK,
                            r"baseline ($\lambda\!=\!0$)",
                            r"$\xi_\mathrm{rms}=1$"]
    ordered = [(l, handles[l]) for l in order if l in handles]
    fig.legend([h for _, h in ordered], [l for l, _ in ordered],
               loc="lower center", ncol=3, fontsize=_FS_LEG,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.12))

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

    fig = build_figure(df)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
