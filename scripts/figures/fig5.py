"""Figure 5: xi distribution before and after interference-aware training.

    (a) matched baseline (lam = 0)   (b) iiLoss at the operating point

Same polar representation as fig2.py, so the trained model can be read against
the published models shown there. Rows are pooled over seeds: every run whose
name is the given stem, with or without an _s<seed> suffix, contributes to the
same panel.
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# -- Journal style (npj Computational Materials) --------------------------
# Fonts are scaled so they print at 7 pt (body) and 6 pt (minor) once the
# figure is reduced to a 170 mm double column.
_COL_W   = 6.69          # 170 mm in inches
_FIG_W   = 5.5 * 2 + 0.8
_FS      = round(7.0 * _FIG_W / _COL_W)
_FS_SM   = round(6.0 * _FIG_W / _COL_W)
_FS_LEG, _FS_CB = _FS, _FS
_FS_PANEL = 20
plt.rcParams.update({"font.family": "sans-serif", "font.size": _FS,
                     "axes.linewidth": 0.8})

_CMAP    = "RdBu_r"
_COL_REF = "#6B7280"

DEFAULT_STEMS  = ["iiLoss_finetune_0.0", "iiLoss_finetune_0.2"]
DEFAULT_LABELS = [r"baseline ($\lambda=0$)", r"iiLoss ($\lambda=0.2$)"]

BIN_W = 0.1


# -- Helpers --------------------------------------------------------------
def _stem_of(name):
    """Drop a trailing _s<seed> so all seeds of one run pool together."""
    head, _, tail = name.rpartition("_s")
    return head if head and tail.isdigit() else name


def load_scores(csv_path: str, stems: list) -> dict:
    """Load per-reaction rows from interference_scores_<year>.csv by stem."""
    wanted = set(stems)
    by_stem = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            stem = _stem_of(row["model"])
            if stem in wanted:
                by_stem[stem].append({
                    "xi_err":  float(row["xi_err"]),
                    "eta_err": float(row["eta_err"]),
                    "N":       int(row["N"]),
                })
    return by_stem


def _draw_panel(ax, rows, norm, cmap_obj, lim, hist_scale, arc_phi, tick_vals):
    Ns     = np.array([r["N"]       for r in rows])
    xis    = np.array([r["xi_err"]  for r in rows])
    etas   = np.array([r["eta_err"] for r in rows])

    for N_val in sorted(set(Ns)):
        R = math.sqrt(N_val)
        ax.plot(R * np.cos(arc_phi), R * np.sin(arc_phi),
                color="lightgray", lw=1.0, zorder=0)
        ax.text(R * np.cos(0.24) + 0.01 - R * 0.01, R * np.sin(0.24),
                f"{N_val}", color="gray", fontsize=_FS_SM * 0.6,
                va="top", ha="left", rotation=-80)

    ax.scatter(xis, etas, c=xis, cmap=_CMAP, norm=norm,
               s=10, alpha=0.35, zorder=3)

    for N_val in sorted(set(Ns)):
        xis_N = xis[Ns == N_val]
        rms_N = math.sqrt(float(np.mean(xis_N ** 2)))
        eta_N = math.sqrt(max(N_val - rms_N ** 2, 0))
        ax.scatter([rms_N], [eta_N], color="black", s=30, marker="D", zorder=5)

    rms_xi = math.sqrt(float(np.mean(xis ** 2)))
    # Share of reactions cancelling more strongly than the isotropic
    # reference. Unlike a mean or median of xi, this is stated against the
    # size-invariant value xi = 1 and so remains comparable across N.
    frac_below = float(np.mean(xis < 1.0))

    counts, edges = np.histogram(xis, bins=np.arange(0, lim + BIN_W, BIN_W),
                                 density=True)
    for left, right, h in zip(edges[:-1], edges[1:], counts):
        ax.bar(left, h * hist_scale, width=right - left, bottom=lim,
               color=cmap_obj(norm((left + right) / 2)),
               align="edge", edgecolor="none", alpha=0.85,
               zorder=3, clip_on=False)

    def _vline_top(x_val):
        for i in range(len(counts)):
            if edges[i] <= x_val < edges[i + 1]:
                return float(lim + counts[i] * hist_scale)
        return float(lim)

    ax.plot([1.0, 1.0], [0, _vline_top(1.0)], color=_COL_REF,
            lw=0.9, ls=":", alpha=0.7, zorder=4, clip_on=False)
    ax.plot([rms_xi, rms_xi], [0, _vline_top(rms_xi)], color="black",
            lw=1.2, ls="--", zorder=5, clip_on=False)

    ax.text(0.98, 0.98,
            r"$\xi_\mathrm{rms} =" + rf"{rms_xi:.2f}$" + "\n"
            + rf"${100 * frac_below:.0f}\%$ at $\xi < 1$",
            transform=ax.transAxes, fontsize=_FS_SM, ha="right", va="top")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.set_xlabel(r"$\xi = \sqrt{N}\cos\varphi$", fontsize=_FS)
    ax.set_ylabel(r"$\eta = \sqrt{N}\sin\varphi$", fontsize=_FS)
    ax.set_aspect("equal")
    ax.tick_params(top=False, right=False)


# -- Figure ---------------------------------------------------------------
def build_figure(by_stem: dict, stems: list, labels: list):
    present = [s for s in stems if s in by_stem]
    if not present:
        raise SystemExit("None of the requested runs are in the scores file")

    all_N = [r["N"] for s in present for r in by_stem[s]]
    lim   = math.sqrt(max(all_N)) + 0.2

    # One histogram scale across panels, so bar heights are comparable.
    hist_max = 0.0
    for s in present:
        xis = np.array([r["xi_err"] for r in by_stem[s]])
        counts, _ = np.histogram(xis, bins=np.arange(0, lim + BIN_W, BIN_W),
                                 density=True)
        if counts.size:
            hist_max = max(hist_max, float(counts.max()))
    hist_scale = (lim * 0.40) / hist_max if hist_max > 0 else 1.0

    norm      = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=math.sqrt(10))
    cmap_obj  = plt.get_cmap(_CMAP)
    arc_phi   = np.linspace(0, np.pi / 2, 300)
    tick_vals = [t for t in [0, 1, 2, 3] if t <= lim]

    fig, axes = plt.subplots(1, len(present),
                             figsize=(5.5 * len(present) + 0.8, 5.8),
                             squeeze=False)
    axes = axes[0]

    for ax, stem, label, letter in zip(axes, present, labels, "abcdef"):
        _draw_panel(ax, by_stem[stem], norm, cmap_obj, lim, hist_scale,
                    arc_phi, tick_vals)
        ax.text(1.0, 1.1, label, transform=ax.transAxes, fontsize=_FS,
                fontweight="bold", ha="right", va="bottom", clip_on=False)
        ax.text(-0.18, 1.1, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom",
                ha="left", clip_on=False)

    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
        ax.set_ylabel("")

    # -- Shared legend below the panels ------------------------------------
    handles = [
        mlines.Line2D([], [], color="black", marker="D", markersize=5,
                      linestyle="None", label=r"$\xi_\mathrm{rms}(N)$"),
        mlines.Line2D([], [], color=_COL_REF, lw=0.9, ls=":", alpha=0.7,
                      label=r"$\xi = 1$"),
        mlines.Line2D([], [], color="black", lw=1.2, ls="--",
                      label=r"$\xi_\mathrm{rms}$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=_FS_LEG, framealpha=0.9, handlelength=2.5,
               bbox_to_anchor=(0.5, -0.10))

    plt.tight_layout(w_pad=4.0)
    fig.subplots_adjust(right=0.88)

    # -- Shared colorbar ---------------------------------------------------
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.70])
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\xi$", fontsize=_FS_CB)
    cbar.set_ticks([0, 1, math.sqrt(10)])
    cbar.set_ticklabels(["0", "1", r"$\sqrt{10}$"])

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(
        REPO_ROOT, "results", "2026", "interference_scores_2026.csv"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_STEMS,
                        help="run names without the _s<seed> suffix")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                        help="panel titles, one per run")
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figure5.png"))
    args = parser.parse_args()

    labels = args.labels
    if len(labels) != len(args.models):
        labels = list(args.models)

    by_stem = load_scores(args.csv, args.models)
    fig = build_figure(by_stem, args.models, labels)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
