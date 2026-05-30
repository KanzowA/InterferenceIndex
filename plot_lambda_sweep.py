"""
plot_lambda_sweep.py
--------------------
Three-panel figure: Ef MAE, Ed MAE, and RMS xi vs regularisation strength lambda.
One line per model family, all on the same x-axis.

Usage:
    python plot_lambda_sweep.py
    python plot_lambda_sweep.py --csv interference_summary_current.csv --out figure6.pdf
"""

import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Series definitions
SERIES = {
    "iiLoss":        dict(prefix="iiLoss_",        exclude="pcgrad|finetune|save",
                          color="#2166ac", marker="o", ls="-",  lw=1.8),
    "EdLoss":        dict(prefix="EdLoss_",         exclude="pcgrad|save",
                          color="#d73027", marker="s", ls="-",  lw=1.8),
    "iiLoss+PCGrad": dict(prefix="iiLoss_pcgrad_", exclude=None,
                          color="#1a9850", marker="^", ls="--", lw=1.4,
                          anchor="iiLoss_save_0.0"),
    "EdLoss+PCGrad": dict(prefix="EdLoss_pcgrad_", exclude=None,
                          color="#f46d43", marker="D", ls="--", lw=1.4,
                          anchor="EdLoss_0.0"),
}

EF_MAE_MAX = 0.32
ALPHA_MAX  = 0.8

METRICS = [
    ("Ef_MAE", r"$\Delta H_\mathrm{f}$ MAE / eV atom$^{-1}$"),
    ("Ed_MAE", r"$\Delta H_\mathrm{d}$ MAE / eV atom$^{-1}$"),
    ("rms_xi", r"$\xi_\mathrm{rms}$"),
]

# Optimal lambda for iiLoss and EdLoss in panel (b) — marked with a star
ED_OPTIMA = {
    "iiLoss+PCGrad": 0.3,   # lowest Ed_MAE across iiLoss+PCGrad sweep
    "EdLoss+PCGrad": 0.4,   # lowest Ed_MAE across EdLoss+PCGrad sweep
}


def extract_alpha(name, prefix):
    tail = name[len(prefix):]
    m = re.search(r"[\d.]+$", tail)
    return float(m.group()) if m else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="interference_summary_current.csv")
    parser.add_argument("--out", default="figure6_lambda_sweep.pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    # Collect series data for optimum markers in panel b
    series_data = {}

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
            sub = sub[sub["Ef_MAE"] <= EF_MAE_MAX].reset_index(drop=True)
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

            if col == "Ed_MAE":
                series_data[label] = (sub, color)

        ax.set_xlabel(r"$\lambda$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", lw=0.4, alpha=0.5)
        ax.grid(True, which="minor", lw=0.2, alpha=0.3)

        # Baseline axhline — label anchored to right end of line
        baseline_row = df[df["model"] == "iiLoss_0.0"]
        if not baseline_row.empty:
            bval = baseline_row[col].values[0]
            ax.axhline(bval, color="grey", lw=1.0, ls=":", alpha=0.8, zorder=1, label=r"baseline ($\lambda\!=\!0$)")
            #ax.text(0.97, bval, r"baseline ($\lambda\!=\!0$)",
            #        transform=ax.get_yaxis_transform(),
            #        ha="right", va="bottom", fontsize=7, color="grey",
            #        style="italic",)

        # Optimum markers in panel (b) for iiLoss and EdLoss
        if col == "Ed_MAE":
            for ser_label, opt_alpha in ED_OPTIMA.items():
                if ser_label not in series_data:
                    continue
                sub, color = series_data[ser_label]
                row = sub[sub["alpha"] == opt_alpha]
                if row.empty:
                    row = sub.iloc[(sub["alpha"] - opt_alpha).abs().argsort()[:1]]
                xv = row["alpha"].values[0]
                yv = row[col].values[0]
                ax.scatter(xv, yv, marker="*", s=180, color=color,
                           edgecolors="white", linewidths=0.5, zorder=6)
                ax.annotate(fr"$\lambda^*$",
                            xy=(xv, yv), xytext=(6, 6),
                            textcoords="offset points",
                            fontsize=7.5, color=color, fontweight="bold")

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(),
               loc="lower center", ncol=len(seen),
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.13))

    # Panel labels
    for ax, letter in zip(axes, ["a)", "b)", "c)"]):
        ax.set_title(letter, fontsize=13, fontweight="bold", loc="left", pad=6)

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
