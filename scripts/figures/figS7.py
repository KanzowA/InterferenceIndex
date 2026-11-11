"""Figure S7: where interference-aware training puts its accuracy in dfH.

    (a) change of MAE(dfH) against the number of reactions in which a compound
        is a decomposition product
    (b) the same against the DFT hull distance of the compound (Figure 5f)

for iiLoss and HdLoss with lam chosen on the validation split, each against
the lam = 0 control of the same seed. Bars are means over the seeds, error
bars the sample standard deviation; the number of compounds is printed below.

Both objectives gain on the compounds that enter many reactions; only HdLoss
pays for it on the compounds that enter none.

Usage
-----
    python scripts/figures/figS7.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

import fig5
from spectral import HULLOUT_MAP, load_hull, product_counts

# -- Figure style ---------------------------------------------------------
# Fonts are scaled so they print at 7 pt (body) and 6 pt (minor) once the
# figure is reduced to the final column width.
_COL_W    = 6.69          # target column width in inches
_FIG_W    = 13.0          # rendering width for font scaling
_FS       = round(7.0 * _FIG_W / _COL_W)   # body / axis labels
_FS_SM    = round(6.0 * _FIG_W / _COL_W)   # minor annotations
_FS_PANEL = 20            # panel labels a/b
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   _FS,
    "axes.linewidth": 0.8,
})

_COL_II = "#E61D60"
_COL_HD = "#1E46E4"

CONTROL = "iiLoss_finetune_0.0"
SERIES = [("iiLoss", "iiLoss_nested", _COL_II), ("HdLoss", "HdLoss_nested", _COL_HD)]

ROLE_GROUPS = [(r"$0$", 0, 0), (r"$1$–$2$", 1, 2), (r"$3$–$9$", 3, 9),
               (r"$10$–$99$", 10, 99), (r"$\geq 100$", 100, None)]


# -- Figure ---------------------------------------------------------------
def draw(ax, changes, masks, xlabel):
    """Grouped bars, one series per objective; counts below each group."""
    x = np.arange(len(masks))
    width = 0.8 / len(changes)
    lows, highs = [], []
    for i, (label, color, values) in enumerate(changes):
        mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
        ax.bar(x + (i - (len(changes) - 1) / 2) * width, mean, yerr=sd,
               width=width * 0.92, color=color, alpha=0.85, capsize=2.5,
               error_kw=dict(elinewidth=0.9), label=label)
        lows.append(min(0, (mean - sd).min()))
        highs.append(max(0, (mean + sd).max()))
    lo, hi = min(lows), max(highs)
    span = hi - lo
    for xi, (_, m) in zip(x, masks):
        ax.text(xi, lo - 0.03 * span, f"{int(m.sum()):,}".replace(",", " "),
                ha="center", va="top", fontsize=_FS_SM - 2, color="0.35")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in masks])
    ax.set_ylim(lo - 0.16 * span, hi + 0.06 * span)
    ax.set_xlabel(xlabel, fontsize=_FS)
    ax.set_ylabel(r"$\Delta\mathrm{MAE}(\Delta_\mathrm{f}H)$ / %", fontsize=_FS)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)


def build_figure(H):
    groups = {
        "roles": fig5.group_masks(product_counts(H), ROLE_GROUPS),
        "hull": fig5.hull_masks(H["hd"][:H["ncomp"]]),
    }
    control_runs = fig5.load_runs(H, "2025", CONTROL)
    changes = {key: [] for key in groups}
    for label, stem, color in SERIES:
        control, model = fig5.paired(control_runs, fig5.load_runs(H, "2025", stem))
        for key, masks in groups.items():
            changes[key].append((label, color, fig5.change_f(control, model, masks)))

    fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, 4.6),
                             gridspec_kw=dict(wspace=0.3))
    draw(axes[0], changes["roles"], groups["roles"], "reactions as decomposition product")
    draw(axes[1], changes["hull"], groups["hull"],
         r"$\Delta_\mathrm{d}H^\mathrm{DFT}$ / eV atom$^{-1}$")
    axes[0].legend(loc="upper right", fontsize=_FS_SM, frameon=False)

    for ax, letter in zip(axes, "ab"):
        ax.text(-0.15, 1.03, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)
        ax.tick_params(labelsize=_FS_SM)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figureS7.png"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(load_hull(HULLOUT_MAP["2025"]))
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
