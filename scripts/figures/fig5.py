"""Figure 5: what interference-aware training changes, reaction by reaction.

    (a) xi, matched control (lam = 0)     (b) xi, iiLoss (lam from validation)
    (c) distribution of dH_d,ML - dH_d,DFT, control and iiLoss
    (d) change of MAE(dH_d) against N
    (e) the same change split into its xi part and its |r| part
    (f) change of MAE(dH_f) against the DFT hull distance of the compound

Figure S6 is the same layout for the preprint's objective, iiLossRaw at
lam = 0.2, against the same control.

Every number is computed from the out-of-fold predictions in
data/<year>/ml/Hf/, with spectral.py. Each model is paired with the control of
the same seed; bars are means over the seeds and error bars the sample standard
deviation. Panels a and b pool the seeds, as fig1.py pools models. Panel e uses
the exact split |y| = xi |r| of spectral.error_split, so its two bars add up to
the bar of panel d at every N. Panel f shows where the accuracy goes: to the
compounds on and near the hull, at a small cost far above it. Why the hull
phases gain is in Figure S7 (figS7.py), by the number of reactions in which a
compound is a decomposition product.

Arcs and scatter of panels a and b are drawn for N <= 10 to keep the fig1
frame, while xi_rms and the reported fraction are computed over all reactions,
so they match the summary table.

Usage
-----
    python scripts/figures/fig5.py
    python scripts/figures/fig5.py --models iiLoss_finetune_0.0 iiLossRaw_finetune_0.2 \\
        --labels control iiLossRaw --out figures/figureS6.png
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from spectral import (HULLOUT_MAP, error_split, free_sizes, load_hull,
                      reaction_sums, residual)

COL_REF = "#6B7280"
COL_II  = "#E61D60"
N_VALS  = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CMAP    = "RdBu_r"
NORM    = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=np.sqrt(10))

# -- Figure style ---------------------------------------------------------
# Two rows of three panels across the full page width (180 mm); fonts are
# scaled so they print at 7 pt (body) and 6 pt (minor).
_COL_W    = 7.09
_FIG_W    = 15.0
_FIG_H    = 12.6
_FS       = round(7.0 * _FIG_W / _COL_W)
_FS_SM    = round(6.0 * _FIG_W / _COL_W)
_FS_N     = _FS_SM - 3            # the N labels on the arcs
_FS_COUNT = _FS_SM - 2            # the counts under the bars
_FS_PANEL = round(9.0 * _FIG_W / _COL_W)
plt.rcParams.update({"font.family": "sans-serif", "font.size": _FS,
                     "axes.linewidth": 0.8})

TICK_VALS = [0, 1, 2, 3]
CIRC_TOP  = 3.3

HIST_DENS_MAX = 2.6
HIST_H        = 1.1
HIST_SCALE    = HIST_H / HIST_DENS_MAX

ARC_PHI = np.linspace(0, np.pi / 2, 300)

# Panel geometry, in inches: a pair of tied panels (a|b, d|e) and a single one
# (c, f) per row, on a common grid.
_SIDE   = 4.1             # panel width; panels a and b are square
_TIE    = 0.18            # gap between tied panels
_GAP    = 1.55            # gap before the single panel
_LEFT   = 1.05
_ROW1_Y = 6.55
_ROW2_Y = 0.85
_ROW2_H = 3.7
_C_EXTRA = 0.42           # c reaches up to the xi histograms

DEFAULT_STEMS  = ["iiLoss_finetune_0.0", "iiLoss_nested"]
DEFAULT_LABELS = ["control", "iiLoss"]

N_GROUPS = [(r"$1$", 1, 1), (r"$2$", 2, 2), (r"$3$", 3, 3), (r"$4$", 4, 4),
            (r"$5$", 5, 5), (r"$\geq 6$", 6, None)]
# DFT hull distance in eV/atom; the bins are half-open on the left, (lo, hi]
HULL_GROUPS = [(r"$\leq 0$", None, 0.0), (r"$0$–$0.1$", 0.0, 0.1),
               (r"$0.1$–$0.3$", 0.1, 0.3), (r"$0.3$–$1$", 0.3, 1.0),
               (r"$> 1$", 1.0, None)]


# -- Data -----------------------------------------------------------------

def load_runs(H, year, stem):
    """Per seed: reaction sums y and norms |r|, and |delta| per compound."""
    nc = H["ncomp"]
    out = {}
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "data", year, "ml", "Hf",
                                              f"{stem}_s*", "ml_input.json"))):
        name = os.path.basename(os.path.dirname(path))
        head, _, seed = name.rpartition("_s")
        if head != stem or not seed.isdigit():
            continue
        with open(path) as f:
            d, _ = residual(H, json.load(f))
        y, q = reaction_sums(H, d)
        out[int(seed)] = dict(y=y, r=np.sqrt(q), abs_f=np.abs(d[:nc]))
    if not out:
        raise SystemExit(f"no predictions for {stem} in data/{year}/ml/Hf")
    return out


def paired(control, model):
    """The seeds both runs have, stacked into (seed, item) arrays."""
    seeds = sorted(set(control) & set(model))
    if not seeds:
        raise SystemExit("the two runs share no seed")
    def stack(runs, key):
        return np.array([runs[s][key] for s in seeds])
    return ({k: stack(control, k) for k in ("y", "r", "abs_f")},
            {k: stack(model, k) for k in ("y", "r", "abs_f")})


def xi_distribution(y, r, nfree):
    """{"per_N": {N: xi}, "all": xi} pooled over seeds, as fig1.py expects."""
    n = np.broadcast_to(nfree, y.shape)
    m = (n >= 2) & (r > 0)
    xi, n = np.abs(y[m]) / r[m], n[m]
    per_N = defaultdict(list)
    for N in N_VALS:
        per_N[N] = xi[n == N]
    return {"per_N": dict(per_N), "all": xi.ravel()}


def group_masks(values, groups):
    """Masks for integer groups [lo, hi]; hi = None is open."""
    return [(label, (values >= lo) & ((values <= hi) if hi is not None else True))
            for label, lo, hi in groups]


def hull_masks(hd):
    """Masks for the hull-distance bins (lo, hi]; None is open."""
    return [(label, ((hd > lo) if lo is not None else True)
             & ((hd <= hi) if hi is not None else True))
            for label, lo, hi in HULL_GROUPS]


def change_f(control, model, masks):
    """Relative change of MAE(dH_f) per group (group x seed), in %."""
    return np.array([100 * (model["abs_f"][:, m].mean(axis=1)
                            / control["abs_f"][:, m].mean(axis=1) - 1)
                     for _, m in masks])


# -- Panels ---------------------------------------------------------------

def draw_panel(ax, entry):
    """Draw one xi panel; returns the top of its histogram in axes units."""
    per_N    = entry["per_N"]
    all_xi   = entry["all"]
    cmap_obj = plt.get_cmap(CMAP)

    xi_rms     = np.sqrt(np.mean(all_xi ** 2))
    frac_below = float(np.mean(all_xi < 1.0))

    # Concentric arcs, N labels. Only N values actually present are drawn,
    # while the frame keeps the fig1 extent so the panels stay comparable.
    for N in [n for n in N_VALS if per_N.get(n, np.array([])).size]:
        R = np.sqrt(N)
        ax.plot(R * np.cos(ARC_PHI), R * np.sin(ARC_PHI),
                color="lightgray", lw=1.0, zorder=0)
        ax.text(R * np.cos(0.24) + 0.01 - R * 0.003,
                R * np.sin(0.24),
                f"N={N}", color="gray", fontsize=_FS_N,
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
    return 1 + counts.max() * HIST_SCALE / CIRC_TOP


def draw_error_distribution(ax, err_control, err_model, labels):
    """Step histograms of dH_d,ML - dH_d,DFT, pooled over seeds."""
    edges = np.arange(-0.6, 0.4001, 0.01)
    x = np.repeat(edges, 2)[1:-1]
    for err, label, control in ((err_control, labels[0], True), (err_model, labels[1], False)):
        h, _ = np.histogram(err.ravel(), bins=edges, density=True)
        y = np.repeat(h, 2)
        if control:
            ax.fill_between(x, y, color=COL_REF, alpha=0.30, lw=0, label=label)
            ax.plot(x, y, color=COL_REF, lw=1.0)
        else:
            ax.fill_between(x, y, color=COL_II, alpha=0.12, lw=0)
            ax.plot(x, y, color=COL_II, lw=1.8, label=label)
    ax.axvline(0, color="black", lw=0.9, ls="--", zorder=1)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$\Delta_\mathrm{d}H^\mathrm{ML}-\Delta_\mathrm{d}H^\mathrm{DFT}$"
                  r" / eV atom$^{-1}$", fontsize=_FS)
    ax.set_ylabel(r"$\rho\left(\Delta_\mathrm{d}H^\mathrm{ML}-\Delta_\mathrm{d}H^\mathrm{DFT}"
                  r"\right)$ / atom eV$^{-1}$", fontsize=_FS)
    ax.legend(loc="upper left", fontsize=_FS_SM, frameon=False)


def draw_bars(ax, x, values, masks=None, offset=0.0, width=0.7, hatch=None):
    """Mean and sd over seeds of `values` (group x seed); counts below if masks."""
    mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
    ax.bar(x + offset, mean, yerr=sd, width=width,
           color=("white" if hatch else COL_II), edgecolor=COL_II, hatch=hatch,
           lw=(1.0 if hatch else 0), alpha=(1.0 if hatch else 0.85),
           capsize=2.5, error_kw=dict(elinewidth=0.9))
    if masks is not None:
        span = np.ptp(np.r_[mean - sd, mean + sd, 0.0])
        for xi, (_, m), mu, s in zip(x, masks, mean, sd):
            ax.text(xi + offset, min(mu, 0) - s - 0.025 * span,
                    f"{int(m.sum()):,}".replace(",", " "),
                    ha="center", va="top", fontsize=_FS_COUNT, color="0.35")
    return mean - sd, mean + sd


def bar_frame(ax, x, labels, lo, hi):
    """Zero line, ticks and limits that leave room for the counts."""
    span = max(hi, 0) - min(lo, 0)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_FS)
    ax.set_ylim(min(lo, 0) - 0.16 * span, max(hi, 0) + 0.06 * span)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)


# -- Figure ---------------------------------------------------------------

def build_figure(H, control, model, labels):
    nfree = free_sizes(H)
    n_masks = group_masks(nfree, N_GROUPS)
    f_masks = hull_masks(H["hd"][:H["ncomp"]])
    xi_part, r_part = error_split(control["y"], control["r"], model["y"], model["r"])
    abs_c, abs_m = np.abs(control["y"]), np.abs(model["y"])

    def rel_change(numerator, m):
        return 100 * numerator[:, m].mean(axis=1) / abs_c[:, m].mean(axis=1)

    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    def box(x, y, w, h):
        return fig.add_axes([x / _FIG_W, y / _FIG_H, w / _FIG_W, h / _FIG_H])
    x_pair = (_LEFT, _LEFT + _SIDE + _TIE)
    x_single = _LEFT + 2 * _SIDE + _TIE + _GAP
    h_c = _SIDE + _C_EXTRA
    ax_a, ax_b = (box(x, _ROW1_Y, _SIDE, _SIDE) for x in x_pair)
    ax_c = box(x_single, _ROW1_Y, _SIDE, h_c)
    ax_d, ax_e = (box(x, _ROW2_Y, _SIDE, _ROW2_H) for x in x_pair)
    ax_f = box(x_single, _ROW2_Y, _SIDE, _ROW2_H)

    # (a, b) tied: one y axis, one colour bar and legend beneath the pair
    hist_top = max(draw_panel(ax_a, xi_distribution(control["y"], control["r"], nfree)),
                   draw_panel(ax_b, xi_distribution(model["y"], model["r"], nfree)))
    ax_b.set_ylabel("")
    ax_b.tick_params(labelleft=False)

    pair_w = 2 * _SIDE + _TIE
    cax = box(_LEFT + 0.08 * pair_w, _ROW1_Y - 1.2, 0.5 * pair_w, 0.13)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0, 1, np.sqrt(10)])
    cbar.set_ticklabels(["0", "1", r"$\sqrt{10}$"])
    cbar.ax.tick_params(labelsize=_FS_SM)
    cbar.ax.text(-0.03, 0.5, r"$\xi$", transform=cbar.ax.transAxes,
                 ha="right", va="center", fontsize=_FS)
    leg_handles = [
        mlines.Line2D([], [], color="black", marker="D", markersize=5,
                      linestyle="None", label=r"$\xi_\mathrm{rms}(N)$"),
        mlines.Line2D([], [], color=COL_REF, lw=0.9, ls=":", alpha=0.7,
                      label=r"$\xi = 1$"),
        mlines.Line2D([], [], color="black", lw=1.2, ls="--",
                      label=r"$\xi_\mathrm{rms}$"),
    ]
    fig.legend(handles=leg_handles, loc="center left", fontsize=_FS_SM,
               frameon=False, handlelength=1.6, labelspacing=0.25,
               bbox_to_anchor=((_LEFT + 0.64 * pair_w) / _FIG_W,
                               (_ROW1_Y - 1.25) / _FIG_H))

    # (c)
    draw_error_distribution(ax_c, -control["y"], -model["y"], labels)

    # (d, e) tied: e splits each bar of d into its xi part and its |r| part
    x = np.arange(len(n_masks))
    total = np.array([rel_change(abs_m - abs_c, m) for _, m in n_masks])
    lo_d, hi_d = draw_bars(ax_d, x, total, masks=n_masks)
    lo_1, hi_1 = draw_bars(ax_e, x, np.array([rel_change(xi_part, m) for _, m in n_masks]),
                           offset=-0.2, width=0.38)
    lo_2, hi_2 = draw_bars(ax_e, x, np.array([rel_change(r_part, m) for _, m in n_masks]),
                           offset=0.2, width=0.38, hatch="////")
    lo = min(lo_d.min(), lo_1.min(), lo_2.min())
    hi = max(hi_d.max(), hi_1.max(), hi_2.max())
    for ax in (ax_d, ax_e):
        bar_frame(ax, x, [g[0] for g in n_masks], lo, hi)
        ax.set_xlabel(r"$N$", fontsize=_FS)
    ax_d.set_ylabel(r"$\Delta\mathrm{MAE}(\Delta_\mathrm{d}H)$ / %", fontsize=_FS)
    ax_e.tick_params(labelleft=False)
    ax_e.legend(handles=[
        mpatches.Patch(facecolor=COL_II, alpha=0.85, label=r"via $\xi$"),
        mpatches.Patch(facecolor="white", edgecolor=COL_II, hatch="////",
                       label=r"via $\Vert\mathbf{r}\Vert$")],
        loc="lower left", fontsize=_FS_SM, frameon=False)

    # (f)
    x = np.arange(len(f_masks))
    lo_f, hi_f = draw_bars(ax_f, x, change_f(control, model, f_masks), masks=f_masks)
    bar_frame(ax_f, x, [g[0] for g in f_masks], lo_f.min(), hi_f.max())
    ax_f.set_xlabel(r"$\Delta_\mathrm{d}H^\mathrm{DFT}$ / eV atom$^{-1}$", fontsize=_FS)
    ax_f.set_ylabel(r"$\Delta\mathrm{MAE}(\Delta_\mathrm{f}H)$ / %", fontsize=_FS)

    # Panel letters on one line per row, above the tallest xi histogram
    top_row1 = max(h_c / _SIDE, hist_top)
    for ax, letter, top in ((ax_a, "a", top_row1), (ax_b, "b", top_row1),
                            (ax_c, "c", top_row1 * _SIDE / h_c), (ax_d, "d", 1.0),
                            (ax_e, "e", 1.0), (ax_f, "f", 1.0)):
        dx = -0.06 if letter in "be" else -0.24
        ax.text(dx, top + 0.03, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)
        ax.tick_params(labelsize=_FS_SM)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2025", choices=["2020", "2025", "2026"])
    parser.add_argument("--models", nargs=2, default=DEFAULT_STEMS,
                        help="control and model, run names without the _s<seed> suffix")
    parser.add_argument("--labels", nargs=2, default=DEFAULT_LABELS,
                        help="legend entries of panel c")
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figure5.png"))
    args = parser.parse_args()

    H = load_hull(HULLOUT_MAP[args.year])
    control = load_runs(H, args.year, args.models[0])
    model = load_runs(H, args.year, args.models[1])
    control, model = paired(control, model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(H, control, model, args.labels)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
