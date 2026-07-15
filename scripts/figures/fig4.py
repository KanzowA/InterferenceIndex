"""
plot_lambda_sweep.py
--------------------
2×2 figure: (a) Hf MAE, (b) Hd MAE, (c) F1, (d) RMS xi
vs regularisation strength lambda.  One line per model family.

Usage:
    python scripts/figures/fig6_lambda_sweep.py
    python scripts/figures/fig4.py --csv results/2026/interference_summary_2026.csv --out figures/figure4.png
"""

import argparse
import os
import re
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Journal style (npj Computational Materials) ──────────────────────────
_JOURNAL_COL_W = 6.69          # 170 mm double-column in inches
_FIG_W         = 12.0          # scaling reference
_FS    = round(7.0 * _FIG_W / _JOURNAL_COL_W)   # → 13 pt  (body / axis labels)
_FS_SM = round(6.0 * _FIG_W / _JOURNAL_COL_W)   # → 11 pt  (minor annotations)
_FS_LEG = _FS
_FS_PANEL = 20   # panel labels a/b/c/d
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': _FS,
                     'axes.linewidth': 0.8})

# ── Series definitions ────────────────────────────────────────────────────
SERIES = {
    "iiLoss":        dict(prefix="iiLoss_",        exclude="pcgrad|save|finetune",
                          color="#7B9FFF", marker="o", ls="-",  lw=1.2, alpha=0.35),
    "HdLoss":        dict(prefix="HdLoss_",         exclude="pcgrad|save|finetune",
                          color="#FF85C8", marker="s", ls="-",  lw=1.8),
    "iiLoss+PCGrad": dict(prefix="iiLoss_pcgrad_", exclude=None,
                          color="#4361EE", marker="^", ls="--", lw=1.8,
                          anchor="iiLoss_save_0.0"),
    "HdLoss+PCGrad": dict(prefix="HdLoss_pcgrad_", exclude=None,
                          color="#E040AB", marker="D", ls="--", lw=1.8,
                          anchor="HdLoss_0.0"),
}

ALPHA_MAX = 0.8

# ── Panels ────────────────────────────────────────────────────────────────
METRICS = [
    ("Hf_MAE", r"$\Delta H_\mathrm{f}$ MAE / eV atom$^{-1}$"),
    ("Hd_MAE", r"$\Delta H_\mathrm{d}$ MAE / eV atom$^{-1}$"),
    ("F1",     r"$F_1$"),
    ("rms_xi", r"$\xi_\mathrm{rms}$"),
]

MAE_COLS = {"Hf_MAE", "Hd_MAE"}   # fixed ylim + overflow markers

# lambda* markers: col -> "min" (lower-is-better) or "max" (higher-is-better)
OPTIMA_SERIES = ["iiLoss+PCGrad", "HdLoss+PCGrad"]
OPTIMA_PANELS = {"Hd_MAE": "min", "F1": "max"}


def extract_alpha(name, prefix):
    tail = name[len(prefix):]
    m = re.search(r"[\d.]+$", tail)
    return float(m.group()) if m else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(_REPO, "results", "2026",
                                                       "interference_summary_2026.csv"))
    parser.add_argument("--out", default=os.path.join(_REPO, "figures",
                                                       "figure4.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    fig, axes_2d = plt.subplots(2, 2, figsize=(10, 7), sharey=False, sharex=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.42)

    # Flatten in reading order: a=top-left, b=top-right, c=bottom-left, d=bottom-right
    axes = [axes_2d[0, 0], axes_2d[0, 1], axes_2d[1, 0], axes_2d[1, 1]]

    series_data = {}   # {col: {label: (sub_df, color)}}

    for ax_idx, (ax, (col, ylabel)) in enumerate(zip(axes, METRICS)):
        row = ax_idx // 2   # 0 = top, 1 = bottom

        for label, cfg in SERIES.items():
            prefix = cfg["prefix"]
            exclude = cfg["exclude"]
            color  = cfg["color"]
            mk     = cfg["marker"]
            ls     = cfg["ls"]
            lw     = cfg["lw"]
            alpha  = cfg.get("alpha", 1.0)
            anchor = cfg.get("anchor")

            mask = df["model"].str.startswith(prefix)
            if exclude:
                mask &= ~df["model"].str.contains(exclude, regex=True)
            sub = df[mask].copy()
            if sub.empty:
                continue

            sub["alpha"] = sub["model"].apply(lambda n: extract_alpha(n, prefix))
            sub = sub.sort_values("alpha").reset_index(drop=True)
            sub = sub[sub["alpha"] <= ALPHA_MAX].reset_index(drop=True)
            if sub.empty:
                continue

            if anchor is not None:
                anc = df[df["model"] == anchor].copy()
                if not anc.empty:
                    anc["alpha"] = 0.0
                    sub = pd.concat([anc, sub], ignore_index=True)

            ax.plot(sub["alpha"], sub[col],
                    color=color, ls=ls, lw=lw, marker=mk,
                    markersize=5, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3, label=label,
                    alpha=alpha, clip_on=True)

            # Overflow triangles only for MAE panels, non-faded series
            if col in MAE_COLS and alpha >= 1.0:
                ylim_max = 0.265
                over = sub[sub[col] > ylim_max]
                if not over.empty:
                    ax.scatter(over["alpha"], [ylim_max * 0.995] * len(over),
                               marker="^", s=55, color=color, zorder=6, clip_on=False)

            if col in OPTIMA_PANELS:
                series_data.setdefault(col, {})[label] = (sub, color)

        # ── Axes formatting ───────────────────────────────────────────────
        if row == 1:
            ax.set_xlabel(r"$\lambda$", fontsize=_FS)
        else:
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(labelsize=_FS)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        if col in MAE_COLS:
            ax.set_yticks([0.10, 0.15, 0.20, 0.25])
            ax.set_ylim(0.087, 0.265)
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        elif col == "F1":
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
            ax.margins(y=0.08)
        elif col == "rms_xi":
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
            ax.margins(y=0.08)
            ax.axhline(1.0, color='#6B7280', lw=0.9, ls='--', alpha=0.6,
                       zorder=1, label=r'i.i.d. ($\xi_\mathrm{rms}=1$)')

        ax.grid(True, which="major", lw=0.4, alpha=0.5)
        ax.grid(True, which="minor", lw=0.2, alpha=0.3)

        # Baseline dotted line
        baseline_row = df[df["model"] == "iiLoss_0.0"]
        if not baseline_row.empty:
            bval = baseline_row[col].values[0]
            ax.axhline(bval, color="grey", lw=1.0, ls=":", alpha=0.8, zorder=1,
                       label=r"baseline ($\lambda\!=\!0$)")

        # lambda* markers
        if col in OPTIMA_PANELS and col in series_data:
            direction = OPTIMA_PANELS[col]
            for ser_label in OPTIMA_SERIES:
                if ser_label not in series_data[col]:
                    continue
                sub, color = series_data[col][ser_label]
                cand = sub[sub["alpha"] > 0]
                if cand.empty:
                    continue
                best_idx = (cand[col].idxmax() if direction == "max"
                            else cand[col].idxmin())
                xv = cand.loc[best_idx, "alpha"]
                yv = cand.loc[best_idx, col]
                ax.scatter(xv, yv, marker="*", s=280, color=color,
                           edgecolors="white", linewidths=0.6, zorder=6)
                ax.annotate(r"$\mathbf{\lambda^*}$",
                            xy=(xv, yv), xytext=(7, -12),
                            textcoords="offset points",
                            fontsize=_FS_SM, color=color, fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h

    blank = Patch(visible=False)
    LEGEND_ORDER = [
        "iiLoss", "iiLoss+PCGrad","HdLoss", "HdLoss+PCGrad",r"baseline ($\lambda\!=\!0$)"
    ]
    ordered    = [seen[l] for l in LEGEND_ORDER if l in seen]
    labels_leg = [l       for l in LEGEND_ORDER if l in seen]
    ordered.insert(4, blank)
    labels_leg.insert(4, "")
    fig.legend(ordered, labels_leg,
               loc="lower center", ncol=3,
               fontsize=_FS_LEG, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))

    # ── Panel labels ──────────────────────────────────────────────────────
    for ax, letter in zip(axes, ["a", "b", "c", "d"]):
        ax.text(-0.22, 1.05, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
