"""Figure 4: the lam dependence on the validation split, and the lam it selects.

    (a) MAE(dH_d) on the validation split   (b) F1 on the validation split
    (c) how often each lam was selected

Read from the selection logs of the nested runs (results/2025/nested/), so no
number in this figure has seen a test compound. Each point is the change against
the lam = 0 control of the same seed and fold, averaged over the 25 seed-folds
at base width 1024; error bars are the sample standard deviation. The objective
of the preprint, iiLossRaw, is drawn in the colour of iiLoss with open markers.
"""

import argparse
import glob
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# -- Figure style ---------------------------------------------------------
# Fonts are scaled so they print at 7 pt (body) and 6 pt (minor) once the
# figure is reduced to the final column width.
_COL_W    = 6.69          # target column width in inches
_FIG_W    = 13.0          # rendering width for font scaling
_FS       = round(7.0 * _FIG_W / _COL_W)   # body / axis labels
_FS_SM    = round(6.0 * _FIG_W / _COL_W)   # minor annotations
_FS_LEG   = _FS
_FS_PANEL = 20            # panel labels a/b/c
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   _FS,
    "axes.linewidth": 0.8,
})

_COL_II  = "#E61D60"
_COL_HD  = "#1E46E4"
_COL_REF = "#000000"

# -- Series and panels ----------------------------------------------------
# The preprint's objective shares the colour of iiLoss and is told apart by
# open markers and a lighter line, as the PCGrad variants were in the preprint.
SERIES = {
    "iiLoss": dict(key="iiLoss", color=_COL_II, marker="s", lw=1.8, ms=5,
                   alpha=1.0, fill=True),
    "iiLossRaw (preprint)": dict(key="iiLossRaw", color=_COL_II, marker="s", lw=1.0,
                                 ms=4, alpha=0.55, fill=False),
    "HdLoss": dict(key="HdLoss", color=_COL_HD, marker="o", lw=1.8, ms=5,
                   alpha=1.0, fill=True),
}

METRICS = [
    (0, r"validation MAE$\left(\Delta_\mathrm{d}H\right)$, change / %"),
    (1, r"validation $F_1$, change / %"),
]


# -- Helpers --------------------------------------------------------------
def load_logs(log_dir, width):
    logs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "selection*.json"))):
        with open(path) as f:
            log = json.load(f)
        if log["width"] == width:
            logs.append(log)
    if not logs:
        raise SystemExit(f"no selection logs for width {width} in {log_dir}")
    return logs


def series_points(logs, key, index):
    """Sorted (lam, mean, sd) of the change against each log's own control."""
    grid = logs[0]["grid"]
    points = [(0.0, 0.0, 0.0)]
    for lam in grid:
        rel = [100 * (log["val"][f"{key}_{lam:g}"][index] / log["val"]["control"][index] - 1)
               for log in logs]
        points.append((lam, float(np.mean(rel)), float(np.std(rel, ddof=1))))
    return points


# -- Figure ---------------------------------------------------------------
def build_figure(logs):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                             gridspec_kw=dict(wspace=0.38))
    grid = logs[0]["grid"]

    for ax, (index, ylabel) in zip(axes[:2], METRICS):
        for label, cfg in SERIES.items():
            if cfg["key"] not in logs[0]["objectives"]:
                continue
            points = series_points(logs, cfg["key"], index)
            filled = cfg["fill"]
            ax.errorbar([p[0] for p in points], [p[1] for p in points],
                        yerr=[p[2] for p in points],
                        color=cfg["color"], ls="-", lw=cfg["lw"],
                        alpha=cfg["alpha"], marker=cfg["marker"],
                        markersize=cfg["ms"],
                        markerfacecolor=(cfg["color"] if filled else "white"),
                        markeredgecolor=("white" if filled else cfg["color"]),
                        markeredgewidth=(0.5 if filled else 1.0),
                        capsize=2, elinewidth=0.8,
                        zorder=(3 if filled else 2), label=label)
        ax.axhline(0.0, color=_COL_REF, ls="-", lw=1.0, alpha=0.8, zorder=1,
                   label=r"control ($\lambda\!=\!0$)")
        ax.set_xlabel(r"$\lambda$", fontsize=_FS)
        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, which="major", lw=0.4, alpha=0.5)

    # (c) the lam the criterion chose, per objective
    ax = axes[2]
    present = [(label, cfg) for label, cfg in SERIES.items()
               if cfg["key"] in logs[0]["objectives"]]
    width = 0.8 / len(present)
    for i, (label, cfg) in enumerate(present):
        count = Counter(log["selected"][cfg["key"]] for log in logs)
        x = np.arange(len(grid)) + (i - (len(present) - 1) / 2) * width
        ax.bar(x, [count.get(lam, 0) for lam in grid], width=width * 0.92,
               color=cfg["color"] if cfg["fill"] else "white",
               edgecolor=cfg["color"], alpha=cfg["alpha"] if cfg["fill"] else 1.0,
               linewidth=1.0)
    ax.set_xticks(np.arange(len(grid)))
    ax.set_xticklabels([f"{lam:g}" for lam in grid])
    ax.set_xlabel(r"selected $\lambda$", fontsize=_FS)
    ax.set_ylabel(f"seed-folds (of {len(logs)})", fontsize=_FS)
    ax.grid(True, axis="y", which="major", lw=0.4, alpha=0.5)

    handles = {}
    for a in axes[:2]:
        for handle, label in zip(*a.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    fig.legend(list(handles.values()), list(handles.keys()),
               loc="lower center", ncol=4, fontsize=_FS_LEG,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.12))

    for ax, letter in zip(axes, "abc"):
        ax.text(-0.22, 1.04, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default=os.path.join(
        REPO_ROOT, "results", "2025", "nested"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figure4.png"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(load_logs(args.logs, args.width))
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
