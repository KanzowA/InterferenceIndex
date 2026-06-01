"""
plot_lambda_sweep.py
--------------------
Three-panel figure: Ef MAE, Ed MAE, and RMS xi vs regularisation strength lambda.
One line per model family, all on the same x-axis.

Usage:
    python plot_lambda_sweep.py
    python scripts/fig6_lambda_sweep.py --csv results/2026/interference_summary_2026.csv --out figures/figure6_lambda_sweep.png
"""

import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ── Journal style (npj Computational Materials) ──────────────────────────
# source_pt = target_print_pt × (fig_width / journal_col_width)
# Targets 7 pt (body) and 6 pt (minor) at 170 mm double-column.
_JOURNAL_COL_W = 6.69          # 170 mm in inches
_FIG_W         = 12.0          # this figure's width in inches
_FS    = round(7.0 * _FIG_W / _JOURNAL_COL_W)   # → 13 pt  (body / axis labels)
_FS_SM = round(6.0 * _FIG_W / _JOURNAL_COL_W)   # → 11 pt  (minor annotations)
_FS_LEG, _FS_CB = _FS, _FS
_FS_PANEL = 20   # panel labels a/b/c — fixed across all figures
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': _FS,
                     'axes.linewidth': 0.8})

# Series definitions
SERIES = {
    "iiLoss":        dict(prefix="iiLoss_",        exclude="pcgrad|save",
                          color="#7B9FFF", marker="o", ls="-",  lw=1.8),
    "HdLoss":        dict(prefix="HdLoss_",         exclude="pcgrad|save",
                          color="#FF85C8", marker="s", ls="-",  lw=1.8),
    "iiLoss+PCGrad": dict(prefix="iiLoss_pcgrad_", exclude=None,
                          color="#4361EE", marker="^", ls="--", lw=1.4,
                          anchor="iiLoss_0.0"),
    "HdLoss+PCGrad": dict(prefix="HdLoss_pcgrad_", exclude=None,
                          color="#E040AB", marker="D", ls="--", lw=1.4,
                          anchor="HdLoss_0.0"),
}

EF_MAE_MAX = 0.32
ALPHA_MAX  = 0.8

METRICS = [
    ("Hf_MAE", r"$\Delta H_\mathrm{f}$ MAE / eV atom$^{-1}$"),
    ("Hd_MAE", r"$\Delta H_\mathrm{d}$ MAE / eV atom$^{-1}$"),
    ("rms_xi", r"$\xi_\mathrm{rms}$"),
]

# λ* markers: shown on panel b (Hd_MAE) only for PCGrad variants.
# Optimum is detected dynamically from the data (argmin).
OPTIMA_SERIES  = ["iiLoss+PCGrad", "HdLoss+PCGrad"]
OPTIMA_PANELS  = {"Hd_MAE"}


def extract_alpha(name, prefix):
    tail = name[len(prefix):]
    m = re.search(r"[\d.]+$", tail)
    return float(m.group()) if m else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/2026/interference_summary_2026.csv")
    parser.add_argument("--out", default="figure6_lambda_sweep.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(3, 1, figsize=(5, 10), sharey=False, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # Collect series data for optimum markers (Hd_MAE and rms_xi panels)
    series_data = {}   # {col: {label: (sub_df, color)}}

    for ax_idx, (ax, (col, ylabel)) in enumerate(zip(axes, METRICS)):
        for label, cfg in SERIES.items():
            prefix  = cfg["prefix"]
            exclude = cfg["exclude"]
            color   = cfg["color"]
            mk      = cfg["marker"]
            ls      = cfg["ls"]
            lw      = cfg["lw"]
            anchor  = cfg.get("anchor")

            mask = df["model"].str.startswith(prefix)
            if exclude:
                mask &= ~df["model"].str.contains(exclude, regex=True)
            sub = df[mask].copy()
            if sub.empty:
                continue

            sub["alpha"] = sub["model"].apply(lambda n: extract_alpha(n, prefix))
            sub = sub.sort_values("alpha").reset_index(drop=True)
            sub = sub[sub["Hf_MAE"] <= EF_MAE_MAX].reset_index(drop=True)
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
                    markeredgewidth=0.5, zorder=3, label=label)

            if col in OPTIMA_PANELS:
                series_data.setdefault(col, {})[label] = (sub, color)

        # x-axis: ticks + label only on bottom panel
        if ax_idx < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"$\lambda$", fontsize=_FS)
        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(labelsize=_FS)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        if ax_idx in (0, 1):
            ax.set_yticks([0.10, 0.15, 0.20, 0.25])
            ax.set_ylim(0.087, 0.265)
        elif ax_idx == 2:
            # No hardcoded range — let data drive it, then add 8 % padding
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.margins(y=0.08)
        ax.grid(True, which="major", lw=0.4, alpha=0.5)
        ax.grid(True, which="minor", lw=0.2, alpha=0.3)

        # Baseline dotted line
        baseline_row = df[df["model"] == "iiLoss_0.0"]
        if not baseline_row.empty:
            bval = baseline_row[col].values[0]
            ax.axhline(bval, color="grey", lw=1.0, ls=":", alpha=0.8, zorder=1,
                       label=r"baseline ($\lambda\!=\!0$)")

        # λ* markers: dynamic argmin for Hd_MAE and rms_xi panels
        if col in OPTIMA_PANELS and col in series_data:
            for ser_label in OPTIMA_SERIES:
                if ser_label not in series_data[col]:
                    continue
                sub, color = series_data[col][ser_label]
                # exclude λ=0 anchor when finding minimum
                cand = sub[sub["alpha"] > 0]
                if cand.empty:
                    continue
                best_idx = cand[col].idxmin()
                xv = cand.loc[best_idx, "alpha"]
                yv = cand.loc[best_idx, col]
                ax.scatter(xv, yv, marker="*", s=280, color=color,
                           edgecolors="white", linewidths=0.6, zorder=6)
                ax.annotate(rf"$\mathbf{{\lambda^*}}$",
                            xy=(xv, yv), xytext=(7, -12),
                            textcoords="offset points",
                            fontsize=_FS_SM, color=color, fontweight="bold")

    # Legend — below the figure

    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h

    blank = Patch(visible=False)

    LEGEND_ORDER = [
    "iiLoss",
    "HdLoss",
    "iiLoss+PCGrad",
    "HdLoss+PCGrad",
    r"baseline ($\lambda\!=\!0$)",
    ]

    ordered = [seen[l] for l in LEGEND_ORDER if l in seen]
    labels = [l for l in LEGEND_ORDER if l in seen]

    ordered.insert(2, blank)
    labels.insert(2, "")
    fig.legend(ordered, labels,
               loc="lower right", ncol=2,
               fontsize=_FS_LEG, framealpha=0.9,
               bbox_to_anchor=(1.0, -0.08))

    # Panel labels
    for ax, letter in zip(axes, ["a", "b", "c"]):
        ax.text(-0.30, 1.05, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()
