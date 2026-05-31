"""
fig3_model_comparison.py
------------------------
Fig. 3 — Interference circles for the seven baseline models.

Reads pre-computed per-compound scores from results/2020/interference_scores_2020.csv
(generated once by: python interference_score.py allMP).

Usage:
    python scripts/fig3_model_comparison.py
    python scripts/fig3_model_comparison.py --csv results/2020/interference_scores_2020.csv
    python scripts/fig3_model_comparison.py --models ElFrac Meredig Magpie
    python scripts/fig3_model_comparison.py --out figures/figure3_model_comparison.png
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.colors import TwoSlopeNorm

# ── Default model order (matches Bartel et al. 2020) ──────────────────────────
DEFAULT_MODELS = ["ElFrac", "Meredig", "Magpie", "AutoMat", "ElemNet", "Roost", "CGCNN"]

_CMAP    = "RdBu_r"
_COL_REF = "#6B7280"

# ── Journal style (npj Computational Materials) ────────────────────────────────────────────
# Figsize is dynamic; font sizes are computed inside plot() after n_cols is known.
_JOURNAL_COL_W = 6.69   # 170 mm in inches


def load_scores(csv_path: str, models: list) -> dict:
    """Load per-compound rows from interference_scores_2020.csv or interference_scores_2026.csv, keyed by model."""
    by_model = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] in models:
                by_model[row["model"]].append({
                    "xi_err":    float(row["xi_err"]),
                    "eta_err": float(row["eta_err"]),
                    "N":         int(row["N"]),
                })
    return by_model


def plot(by_model: dict, models: list, out_path: str):
    models_present = [m for m in models if m in by_model]
    if not models_present:
        raise ValueError("No data found for any of the requested models.")

    # Layout: 4 cols for 7 models (2 rows of 4+3) instead of 3 cols (3×3 with lone last panel)
    n = len(models_present)
    n_cols = 4 if n > 6 else min(n, 3)
    n_rows = math.ceil(n / n_cols)

    # Font sizes: target 7 pt (body) / 6 pt (minor) at 170 mm double-column
    _FIG_W  = 5.5 * n_cols + 0.8
    _FS     = round(7.0 * _FIG_W / _JOURNAL_COL_W)
    _FS_SM  = round(6.0 * _FIG_W / _JOURNAL_COL_W)
    _FS_LEG, _FS_CB = _FS, _FS
    plt.rcParams.update({"font.family": "sans-serif", "font.size": _FS,
                         "axes.linewidth": 0.8})

    all_N = [r["N"] for m in models_present for r in by_model[m]]
    lim   = math.sqrt(max(all_N)) + 0.2
    HIST_H = lim * 0.40

    # Global histogram scale so all panels are comparable
    hist_max = 0.0
    for m in models_present:
        xis = np.array([r["xi_err"] for r in by_model[m]])
        counts, _ = np.histogram(xis, bins=35, range=(0, lim), density=True)
        if counts.size:
            hist_max = max(hist_max, float(counts.max()))
    HIST_SCALE = HIST_H / hist_max if hist_max > 0 else 1.0

    norm      = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=math.sqrt(10))
    cmap_obj  = plt.get_cmap(_CMAP)
    arc_theta = np.linspace(0, np.pi / 2, 300)
    tick_vals = [t for t in [0, 1, 2, 3] if t <= lim]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols + 0.8, 5.5 * n_rows),
                             squeeze=False)

    # Top row: first (n_cols-1) models + legend; bottom row: remaining models
    _LEG_SLOT = n_cols - 1  # legend occupies last slot of row 0
    for idx, model_name in enumerate(models_present):
        slot = idx if idx < _LEG_SLOT else idx + 1
        ax   = axes[slot // n_cols, slot % n_cols]
        rows = by_model[model_name]
        Ns      = np.array([r["N"]         for r in rows])
        xis     = np.array([r["xi_err"]    for r in rows])
        deltas  = np.array([r["eta_err"] for r in rows])

        # Concentric arcs
        for N_val in sorted(set(Ns)):
            R = math.sqrt(N_val)
            ax.plot(R * np.cos(arc_theta), R * np.sin(arc_theta),
                    color="lightgray", lw=1.0, zorder=0)
            ax.text((R + 0.10) * np.cos(0.38), (R + 0.10) * np.sin(0.38),
                    f"{N_val}", color="gray", fontsize=_FS_SM * 0.5,
                    va="top", ha="left", rotation=-80)

        # Scatter
        ax.scatter(xis, deltas, c=xis, cmap=_CMAP, norm=norm,
                   s=7, alpha=0.45, zorder=3)

        # Per-N RMS-ξ diamond
        for N_val in sorted(set(Ns)):
            xis_N = xis[Ns == N_val]
            rms_N = math.sqrt(float(np.mean(xis_N ** 2)))
            eta_N = math.sqrt(max(N_val - rms_N ** 2, 0))
            ax.scatter([rms_N], [eta_N], color="black", s=30, marker="D", zorder=5)

        # Histogram above frame
        rms_xi = math.sqrt(float(np.mean(xis ** 2)))
        counts, edges = np.histogram(xis, bins=35, range=(0, lim), density=True)
        for left, right, h in zip(edges[:-1], edges[1:], counts):
            ax.bar(left, h * HIST_SCALE, width=right - left, bottom=lim,
                   color=cmap_obj(norm((left + right) / 2)),
                   align="edge", edgecolor="none", alpha=0.85,
                   zorder=3, clip_on=False)

        def _vline_top(x_val):
            for i in range(len(counts)):
                if edges[i] <= x_val < edges[i + 1]:
                    return float(lim + counts[i] * HIST_SCALE)
            return float(lim)

        ax.plot([1.0, 1.0], [0, _vline_top(1.0)],
                color=_COL_REF, lw=0.9, ls=":", alpha=0.7,
                zorder=4, clip_on=False)
        ax.plot([rms_xi, rms_xi], [0, _vline_top(rms_xi)],
                color="black", lw=1.2, ls="--", zorder=5, clip_on=False)
        ax.text(0.98, 0.98, r"$\sqrt{\langle\xi^2\rangle}= " + rf"{rms_xi:.2f}$",
                transform=ax.transAxes, fontsize=_FS_SM, ha="right", va="top")

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.set_xlabel(r"$\xi = \sqrt{N}\cos\theta$", fontsize=_FS)
        ax.set_ylabel(r"$\eta = \sqrt{N}\sin\theta$", fontsize=_FS)
        ax.set_aspect("equal")
        ax.tick_params(top=False, right=False)
        ax.text(1.0, 1.1, model_name, transform=ax.transAxes,
                fontsize=_FS, fontweight="bold", ha="right", va="bottom",
                clip_on=False)

    # Shared legend in first empty panel slot (or figure bottom)
    _leg_handles = [
        mlines.Line2D([], [], color="black", marker="D", markersize=5,
                      linestyle="None", label=r"$\sqrt{\langle\xi^2\rangle}_N$"),
        mlines.Line2D([], [], color=_COL_REF, lw=0.9, ls=":", alpha=0.7,
                      label=r"$\xi = 1$"),
        mlines.Line2D([], [], color="black", lw=1.2, ls="--",
                      label=r"$\sqrt{\langle\xi^2\rangle}$"),
    ]
    # Hide any empty slots after the last model (skip legend slot)
    for idx in range(len(models_present), n_rows * n_cols - 1):
        slot = idx if idx < _LEG_SLOT else idx + 1
        axes[slot // n_cols, slot % n_cols].set_visible(False)
    if True:  # always use the dedicated legend slot
        leg_ax  = axes[0, _LEG_SLOT]
        leg_ax.set_visible(True)
        leg_ax.axis("off")
        leg_ax.legend(handles=_leg_handles, loc="center",
                      fontsize=_FS_LEG, framealpha=0.9, handlelength=2.5)

    # Suppress redundant tick labels
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if c != 0:
                ax.tick_params(labelleft=False)
                ax.set_ylabel("")
            if r != n_rows - 1:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")

    plt.tight_layout(w_pad=4.0)
    fig.subplots_adjust(right=0.88)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.70])
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\xi$", fontsize=_FS_CB)
    cbar.set_ticks([0, 1, math.sqrt(10)])
    cbar.set_ticklabels(["0", "1", r"$\sqrt{10}$"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default=os.path.join(_REPO, "results", "2020", "interference_scores_2020.csv"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out",    default=os.path.join(_REPO, "figures", "figure3_model_comparison.png"))
    args = parser.parse_args()

    print(f"Loading scores from {args.csv} ...")
    by_model = load_scores(args.csv, args.models)
    print(f"Models found: {[m for m in args.models if m in by_model]}")
    plot(by_model, args.models, args.out)

if __name__ == "__main__":
    main()