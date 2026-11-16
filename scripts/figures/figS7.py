"""Figure S7: accuracy, error cancellation and bias of the compositional models.

    (a) xi_rms against MAE(dfH)
    (b) mean signed error of the decomposition enthalpy against MAE(dfH)

for the models of Bartel et al. (2020 snapshot) and for the models of this
work (2025 snapshot): the control, iiLoss and HdLoss over the capacity ladder
(base widths 32 to 1024, lambda selected on the validation split; marker size
grows with width) and the naive <xi^2> penalty.

xi_rms is size-invariant and comparable across both snapshots, MAE(dfH) and
the bias only within a snapshot. Points of this work are means over the three
seeds of the capacity ladder (five for the naive penalty); error bars give
the sample standard deviation.

The bias, mean(dHd_ML - dHd_DFT) over all decomposition reactions, is cached
in results/compositional_bias.csv and recomputed from the predictions with
--recompute (about half a minute).

Usage
-----
    python scripts/figures/figS7.py
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

BIAS_CSV = os.path.join(REPO_ROOT, "results", "compositional_bias.csv")

# -- Figure style ---------------------------------------------------------
_COL_W    = 6.69          # target column width in inches
_FIG_W    = 13.0          # rendering width for font scaling
_FS       = round(7.0 * _FIG_W / _COL_W)
_FS_SM    = round(6.0 * _FIG_W / _COL_W)
_FS_PANEL = 20
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   _FS,
    "axes.linewidth": 0.8,
})

_COL_II  = "#E61D60"
_COL_HD  = "#1E46E4"
_COL_REF = "#000000"
_COL_OLD = "#6B7280"

BARTEL = ["ElFrac", "Meredig", "Magpie", "AutoMat", "ElemNet", "Roost", "CGCNN"]
WIDTHS = [32, 64, 128, 256, 512, 1024]
LADDER = {
    "control": dict(stem="iiLoss_finetune_0.0", color=_COL_REF),
    "iiLoss":  dict(stem="iiLoss_nested",       color=_COL_II),
    "HdLoss":  dict(stem="HdLoss_nested",       color=_COL_HD),
}
NAIVE = "iiLossRaw_nested"
SEEDS = [0, 1, 2]


def _stem(base, width):
    return base if width == 1024 else f"{base}_w{width}"


# -- Data -----------------------------------------------------------------
def compute_bias():
    """Mean signed dHd error per model and seed, written to BIAS_CSV."""
    from spectral import HULLOUT_MAP, load_hull, residual, reaction_sums

    def bias(H, path):
        d, _ = residual(H, json.load(open(path)))
        y, _ = reaction_sums(H, d)              # y = S delta = -(dHd error)
        return -1000 * float(np.mean(y))

    rows = []
    H = load_hull(HULLOUT_MAP["2020"])
    for m in BARTEL:
        rows.append(("2020", m, 0, bias(H, os.path.join(
            REPO_ROOT, "data", "2020", "ml", "Hf", m, "ml_input.json"))))
    H = load_hull(HULLOUT_MAP["2025"])
    stems = [_stem(c["stem"], w) for w in WIDTHS for c in LADDER.values()] + [NAIVE]
    for st in stems:
        pattern = os.path.join(REPO_ROOT, "data", "2025", "ml", "Hf", f"{st}_s*", "ml_input.json")
        for p in sorted(glob.glob(pattern)):
            run = os.path.basename(os.path.dirname(p))
            if run.rpartition("_s")[0] != st:
                continue
            rows.append(("2025", st, int(run.rpartition("_s")[2]), bias(H, p)))
    df = pd.DataFrame(rows, columns=["snapshot", "model", "seed", "bias_dHd_meV"])
    df.to_csv(BIAS_CSV, index=False)
    return df


def load(recompute=False):
    old = pd.read_csv(os.path.join(REPO_ROOT, "results", "2020",
                                   "spectral_summary_2020.csv")).set_index("model")
    new = pd.read_csv(os.path.join(REPO_ROOT, "results", "2025",
                                   "spectral_summary_2025.csv"))
    new["stem"] = new.model.str.replace(r"_s\d+$", "", regex=True)
    new["seed"] = new.model.str.extract(r"_s(\d+)$")[0].astype(int)
    bias = compute_bias() if (recompute or not os.path.exists(BIAS_CSV)) \
        else pd.read_csv(BIAS_CSV)
    bias["snapshot"] = bias["snapshot"].astype(str)

    old["bias"] = bias[bias.snapshot == "2020"].set_index("model")["bias_dHd_meV"]

    def stats(stem, seeds=SEEDS):
        x = new[(new.stem == stem) & new.seed.isin(seeds)]
        b = bias[(bias.model == stem) & bias.seed.isin(seeds)]["bias_dHd_meV"]
        if x.empty or b.empty:
            raise KeyError(stem)
        out = {c: (x[c].mean(), x[c].std(ddof=1)) for c in ("MAE_f", "xi_rms")}
        out["bias"] = (b.mean(), b.std(ddof=1))
        return out

    ladder = {k: [stats(_stem(c["stem"], w)) for w in WIDTHS] for k, c in LADDER.items()}
    naive = stats(NAIVE, seeds=range(5))
    return old, ladder, naive


# -- Drawing --------------------------------------------------------------
def _ladder(ax, ladder, ykey, ends):
    """Capacity series; `ends` maps a series to the label offsets of 32/1024."""
    for key, cfg in LADDER.items():
        pts = ladder[key]
        x = [p["MAE_f"][0] for p in pts]
        y = [p[ykey][0] for p in pts]
        ys = [p[ykey][1] for p in pts]
        ax.plot(x, y, color=cfg["color"], lw=1.0, alpha=0.6, zorder=2)
        ax.errorbar(x, y, yerr=ys, fmt="none", ecolor=cfg["color"],
                    elinewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter(x, y, s=np.linspace(22, 70, len(pts)), color=cfg["color"],
                   edgecolor="white", lw=0.6, zorder=3)
        if key in ends:
            dy = ends[key]
            for i in (0, -1):
                ax.annotate(str(WIDTHS[i]), (x[i], y[i]), xytext=(0, dy),
                            textcoords="offset points", ha="center",
                            va="bottom" if dy > 0 else "top",
                            fontsize=_FS_SM - 4, color=cfg["color"])


def _arrows(ax, ladder, ykey):
    """Control -> objective at the full width (1024), the operating point of
    the main text: iiLoss moves left (more accurate), HdLoss to the right."""
    c = ladder["control"][-1]
    for key in ("iiLoss", "HdLoss"):
        p = ladder[key][-1]
        ax.annotate("", xy=(p["MAE_f"][0], p[ykey][0]),
                    xytext=(c["MAE_f"][0], c[ykey][0]),
                    arrowprops=dict(arrowstyle="-|>", color=LADDER[key]["color"],
                                    lw=1.6, shrinkA=6, shrinkB=7,
                                    mutation_scale=14), zorder=4)


def _bartel(ax, old, ycol, offsets):
    for m in BARTEL:
        r = old.loc[m]
        ax.scatter(r["MAE_f"], r[ycol], s=55, marker="D",
                   facecolor="white" if m == "CGCNN" else _COL_OLD,
                   edgecolor=_COL_OLD, lw=1.0, zorder=3)
        dx, dy, ha = offsets.get(m, (7, 0, "left"))
        ax.annotate(m, (r["MAE_f"], r[ycol]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=_FS_SM - 3, color=_COL_OLD)


def _naive(ax, naive, ykey, dx=-9, ha="right"):
    ax.errorbar(naive["MAE_f"][0], naive[ykey][0], yerr=naive[ykey][1],
                fmt="o", ms=8, mfc="white", mec=_COL_II, mew=1.2,
                ecolor=_COL_II, elinewidth=0.8, alpha=0.7, zorder=3)
    ax.annotate(r"naive $\langle\xi^2\rangle$", (naive["MAE_f"][0], naive[ykey][0]),
                xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                fontsize=_FS_SM - 3, color=_COL_II, alpha=0.8)


def _xaxis(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.028, 0.29)
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.03, 0.05, 0.1, 0.2]))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(r"MAE($\Delta_\mathrm{f}H$) / eV atom$^{-1}$", fontsize=_FS)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.tick_params(labelsize=_FS_SM)


def panel_xi(ax, old, ladder, naive):
    ax.axhline(1.0, color=_COL_REF, ls=":", lw=0.9, alpha=0.6, zorder=1)
    _bartel(ax, old, "xi_rms", {"ElFrac": (-7, 0, "right"),
                                "Roost": (0, 10, "center"),
                                "ElemNet": (-7, 7, "right"),
                                "AutoMat": (7, 7, "left"),
                                "Magpie": (7, -7, "left")})
    _ladder(ax, ladder, "xi_rms", ends={"control": 7, "HdLoss": -9})
    _arrows(ax, ladder, "xi_rms")
    _naive(ax, naive, "xi_rms")
    ax.set_ylabel(r"$\xi_\mathrm{rms}$", fontsize=_FS)
    ax.set_ylim(0.55, 1.17)
    ax.text(0.03, 0.97, "errors accumulate", transform=ax.transAxes,
            ha="left", va="top", fontsize=_FS_SM - 2, style="italic")
    ax.text(0.03, 0.03, "errors cancel", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=_FS_SM - 2, style="italic")
    _xaxis(ax)


def panel_bias(ax, old, ladder, naive):
    ax.axhline(0.0, color=_COL_REF, ls=":", lw=0.9, alpha=0.6, zorder=1)
    _bartel(ax, old, "bias", {"ElFrac": (-7, 0, "right"),
                              "Meredig": (-7, 0, "right"),
                              "Magpie": (7, 6, "left"),
                              "AutoMat": (7, 6, "left"),
                              "ElemNet": (-7, -7, "right"),
                              "Roost": (0, 10, "center"),
                              "CGCNN": (7, 0, "left")})
    _ladder(ax, ladder, "bias", ends={"control": -9, "HdLoss": 7})
    _arrows(ax, ladder, "bias")
    _naive(ax, naive, "bias", dx=9, ha="left")
    ax.set_ylabel(r"mean error of $\Delta_\mathrm{d}H$ / meV atom$^{-1}$", fontsize=_FS)
    ax.set_ylim(-160, 15)
    ax.text(0.03, 0.03, "unstable compounds appear too stable", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=_FS_SM - 2, style="italic")
    _xaxis(ax)


# -- Figure ---------------------------------------------------------------
def build_figure(recompute=False):
    old, ladder, naive = load(recompute)
    fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, 5.0),
                             gridspec_kw=dict(wspace=0.3))
    panel_xi(axes[0], old, ladder, naive)
    panel_bias(axes[1], old, ladder, naive)

    handles = [
        plt.Line2D([], [], marker="D", ls="", color=_COL_OLD, ms=7,
                   label="Bartel et al. (2020)"),
        plt.Line2D([], [], marker="D", ls="", mfc="white", mec=_COL_OLD, ms=7,
                   label="CGCNN (structure)"),
        plt.Line2D([], [], marker="o", color=_COL_REF, lw=1, ms=7,
                   label=r"control ($\lambda\!=\!0$)"),
        plt.Line2D([], [], marker="o", color=_COL_II, lw=1, ms=7, label="iiLoss"),
        plt.Line2D([], [], marker="o", color=_COL_HD, lw=1, ms=7, label="HdLoss"),
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec=_COL_II, ms=7,
                   alpha=0.7, label=r"naive $\langle\xi^2\rangle$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6,
               fontsize=_FS_SM - 2, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.15, 1.03, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="recompute the bias cache from the predictions")
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figureS7.png"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(args.recompute)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
