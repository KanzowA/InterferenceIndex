"""Figure 5: xi distribution before and after interference-aware training.

    (a) matched baseline (lam = 0)
    (b) iiLoss at the operating point (lam = 0.2)

Layout follows fig1.py; only the data source differs, so the trained model can
be read directly against the reference distributions shown there. Rows are
pooled over seeds: every run whose name is the given stem, with or without an
_s<seed> suffix, contributes to the same panel.

Arcs and scatter are drawn for N <= 10 to keep the fig1 frame, while
xi_rms and the reported fraction are computed over all reactions, so they match
the summary table.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

COL_REF = "#6B7280"
N_VALS  = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CMAP    = "RdBu_r"
NORM    = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=np.sqrt(10))

_COL_W           = 6.69
_FIG_W           = 11.0
_PT_BODY, _PT_SM = 9.0, 7.5
_FS    = round(_PT_BODY * _FIG_W / _COL_W)
_FS_SM = round(_PT_SM   * _FIG_W / _COL_W)
_FS_LEG, _FS_CB = _FS, _FS
_FS_PANEL = 30
plt.rcParams.update({"font.family": "sans-serif", "font.size": _FS,
                     "axes.linewidth": 0.8})

TICK_VALS = [0, 1, 2, 3]
CIRC_TOP  = 3.3

HIST_DENS_MAX = 2.6
HIST_H        = 1.1
HIST_SCALE    = HIST_H / HIST_DENS_MAX
FULL_H        = CIRC_TOP + HIST_H

ARC_PHI = np.linspace(0, np.pi / 2, 300)

DEFAULT_STEMS  = ["iiLoss_finetune_0.0", "iiLoss_finetune_0.2"]
DEFAULT_LABELS = [r"baseline ($\lambda=0$)", r"iiLoss ($\lambda=0.2$)"]


def _stem_of(name):
    """Drop a trailing _s<seed> so all seeds of one run pool together."""
    head, _, tail = name.rpartition("_s")
    return head if head and tail.isdigit() else name


def load_distributions(csv_path, stems):
    """Return {stem: {"per_N": {N: xi array}, "all": xi array}}."""
    wanted = set(stems)
    per_stem = {s: defaultdict(list) for s in stems}
    every    = {s: [] for s in stems}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            stem = _stem_of(row["model"])
            if stem not in wanted:
                continue
            xi = float(row["xi_err"])
            every[stem].append(xi)
            N = int(row["N"])
            if N in N_VALS:
                per_stem[stem][N].append(xi)
    data = {}
    for s in stems:
        if not every[s]:
            continue
        data[s] = {"per_N": {N: np.array(v) for N, v in per_stem[s].items()},
                   "all":   np.array(every[s])}
    return data


def draw_panel(ax, entry):
    per_N    = entry["per_N"]
    all_xi   = entry["all"]
    cmap_obj = plt.get_cmap(CMAP)

    xi_rms     = np.sqrt(np.mean(all_xi ** 2))
    frac_below = float(np.mean(all_xi < 1.0))

    # Concentric arcs, N labels
    for N in N_VALS:
        R = np.sqrt(N)
        ax.plot(R * np.cos(ARC_PHI), R * np.sin(ARC_PHI),
                color="lightgray", lw=1.0, zorder=0)
        ax.text(R * np.cos(0.24) + 0.01 - R * 0.003,
                R * np.sin(0.24),
                f"N={N}", color="gray", fontsize=_FS_SM + 1,
                va="top", ha="left", rotation=-80)

    # Scatter coloured by xi
    all_x, all_y, all_c = [], [], []
    for N in N_VALS:
        xi = np.clip(per_N.get(N, np.array([])), 0, np.sqrt(N))
        if xi.size == 0:
            continue
        delta = np.sqrt(np.maximum(N - xi ** 2, 0))
        all_x.extend(xi)
        all_y.extend(delta)
        all_c.extend(xi)
    ax.scatter(all_x, all_y, c=all_c, cmap=CMAP, norm=NORM,
               s=10, alpha=0.35, zorder=3)

    # Per-N RMS-xi diamonds
    for N in N_VALS:
        xi_N = np.clip(per_N.get(N, np.array([])), 0, np.sqrt(N))
        if xi_N.size == 0:
            continue
        m  = np.sqrt(np.mean(xi_N ** 2))
        md = np.sqrt(max(N - m ** 2, 0))
        ax.scatter([m], [md], color="black", s=30, marker="D", zorder=5)

    # Histogram bars
    edges = np.arange(0, CIRC_TOP + 0.1, 0.1)
    counts, _ = np.histogram(all_xi, bins=edges, density=True)

    for left, right, h in zip(edges[:-1], edges[1:], counts):
        xi_mid = (left + right) / 2
        bar_h  = h * HIST_SCALE
        ax.bar(left, bar_h, width=right - left, bottom=CIRC_TOP,
               color=cmap_obj(NORM(xi_mid)),
               align="edge", edgecolor="none", alpha=0.85,
               zorder=3, clip_on=False)

    # Reference lines
    def _vline_top(x_val):
        for i in range(len(counts)):
            if edges[i] <= x_val < edges[i + 1]:
                return float(CIRC_TOP + counts[i] * HIST_SCALE)
        return float(CIRC_TOP)

    xi1_top = _vline_top(1.0)
    rms_top = _vline_top(xi_rms)
    ax.plot([1.0, 1.0], [0, xi1_top], color=COL_REF, lw=0.9, ls=":",
            alpha=0.7, zorder=4, clip_on=False)
    ax.plot([xi_rms, xi_rms], [0, rms_top], color="black", lw=1.2, ls="--",
            zorder=5, clip_on=False)
    ax.text(0.98, 0.98,
            r"$\xi_\mathrm{rms} =" + rf"{xi_rms:.2f}$" + "\n"
            + rf"${100 * frac_below:.0f}\%$ at $\xi < 1$",
            transform=ax.transAxes, fontsize=_FS_SM, ha="right", va="top")

    ax.set_xlim(0, CIRC_TOP)
    ax.set_ylim(0, CIRC_TOP)
    ax.set_xticks(TICK_VALS)
    ax.set_yticks(TICK_VALS)
    ax.set_xlabel(r"$\xi = \sqrt{N}\cos\varphi$", fontsize=_FS)
    ax.set_ylabel(r"$\eta = \sqrt{N}\sin\varphi$", fontsize=_FS)
    ax.set_aspect("equal")


def build_figure(data, stems, labels):
    """Assemble the two-panel figure and return it."""
    present = [s for s in stems if s in data]
    if len(present) < 2:
        raise SystemExit("Need two runs present in the scores file")

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.5, 13))
    fig.subplots_adjust(hspace=0.35)

    draw_panel(ax_a, data[present[0]])
    draw_panel(ax_b, data[present[1]])

    ax_a.set_xlabel("")
    ax_a.tick_params(labelbottom=False)

    leg_handles = [
        mlines.Line2D([], [], color="black", marker="D", markersize=5,
                      linestyle="None", label=r"$\xi_\mathrm{rms}(N)$"),
        mlines.Line2D([], [], color=COL_REF, lw=0.9, ls=":", alpha=0.7,
                      label=r"$\xi = 1$"),
        mlines.Line2D([], [], color="black", lw=1.2, ls="--",
                      label=r"$\xi_\mathrm{rms}$"),
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=3,
               fontsize=_FS_LEG, framealpha=0.9, bbox_to_anchor=(0.5, 0.025))

    cbar_ax = fig.add_axes([0.15, -0.015, 0.70, 0.012])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$\xi$", fontsize=_FS_CB)
    cbar.set_ticks([0, 1, np.sqrt(10)])
    cbar.set_ticklabels(["0", "1", r"$\sqrt{10}$"])

    for ax, lbl, title in [(ax_a, "a", labels[0]), (ax_b, "b", labels[1])]:
        ax.text(-0.18, 1.15, lbl, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom",
                ha="left", clip_on=False)
        ax.text(1.0, 1.15, title, transform=ax.transAxes, fontsize=_FS,
                fontweight="bold", ha="right", va="bottom", clip_on=False)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(
        REPO_ROOT, "results", "2026", "interference_scores_2026.csv"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_STEMS,
                        help="two run names without the _s<seed> suffix")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                        help="panel titles, one per run")
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figure5.png"))
    args = parser.parse_args()

    labels = args.labels
    if len(labels) != len(args.models):
        labels = list(args.models)

    data = load_distributions(args.csv, args.models)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(data, args.models, labels)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
