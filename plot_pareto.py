"""
plot_pareto.py
--------------
Pareto-frontier curves for iiLoss, EdLoss, iiLoss+PCGrad, EdLoss+PCGrad.

X-axis : Ef MAE  (eV/atom)   — lower is better  →
Y-axis : mean ξ              — lower is better  ↓

Usage:
    python plot_pareto.py                                   # reads interference_summary_current.csv
    python plot_pareto.py --csv my_summary.csv --out fig6.pdf
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────────

BASELINES = ["ElFrac", "Meredig", "Magpie", "AutoMat", "ElemNet", "Roost", "CGCNN", "CHGNet"]

SERIES = {
    "iiLoss":          dict(prefix="iiLoss_",          exclude="pcgrad|finetune|save", color="#2166ac", marker="o", ls="-"),
    "iiLoss finetune": dict(prefix="iiLoss_finetune_", exclude=None,                   color="#92c5de", marker="o", ls="-."),
    "EdLoss":          dict(prefix="EdLoss_",           exclude="pcgrad|save",          color="#d6604d", marker="s", ls="-"),
    "iiLoss+PCGrad":   dict(prefix="iiLoss_pcgrad_",   exclude=None,                   color="#4dac26", marker="^", ls="--"),
    "EdLoss+PCGrad":   dict(prefix="EdLoss_pcgrad_",   exclude=None,                   color="#b8860b", marker="D", ls="--"),
}

# Clip degenerate high-loss endpoints (collapsed / diverged runs)
EF_MAE_MAX = 0.30   # exclude any model with Ef MAE above this

BASELINE_MARKERS = {
    "ElFrac":  ("x",  "#aaaaaa"),
    "Meredig": ("+",  "#aaaaaa"),
    "Magpie":  ("*",  "#888888"),
    "AutoMat": ("P",  "#888888"),
    "ElemNet": ("h",  "#555555"),
    "Roost":   ("v",  "#333333"),
    "CGCNN":   ("D",  "#111111"),
    "CHGNet":  ("H",  "#9467bd"),
}

LABEL_ALPHA_SKIP = {   # α values too close to label clearly; label every N-th point
    "iiLoss":        2,
    "EdLoss":        2,
    "iiLoss+PCGrad": 1,
    "EdLoss+PCGrad": 2,
}


def extract_alpha(name: str, prefix: str) -> float:
    """Return the numeric suffix of a model name."""
    tail = name[len(prefix):]
    m = re.search(r"[\d.]+$", tail)
    return float(m.group()) if m else float("nan")


def pareto_frontier(df_series: pd.DataFrame):
    """Return rows on the Pareto frontier (min Ef_MAE AND min mean_xi)."""
    df = df_series.sort_values("Ef_MAE").reset_index(drop=True)
    frontier = []
    best_xi = float("inf")
    for _, row in df.iterrows():
        if row["mean_xi"] <= best_xi:
            frontier.append(row)
            best_xi = row["mean_xi"]
    return pd.DataFrame(frontier)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="interference_summary_current.csv")
    parser.add_argument("--out", default="fig6_pareto.pdf")
    parser.add_argument("--no-pareto", action="store_true",
                        help="Connect all points instead of only frontier")
    parser.add_argument("--xi-col", default="mean_xi",
                        choices=["mean_xi", "median_xi", "rms_xi"],
                        help="Which ξ column to use on y-axis")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Normalise column names
    df.columns = df.columns.str.strip()
    xi_col = args.xi_col
    if xi_col not in df.columns:
        # try rms_xi as fallback
        xi_col = "rms_xi" if "rms_xi" in df.columns else df.columns[-2]
        print(f"  [warn] '{args.xi_col}' not found, using '{xi_col}'")
    df = df.rename(columns={xi_col: "mean_xi"})

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # ── Baseline models ───────────────────────────────────────────────────────
    for name in BASELINES:
        row = df[df["model"] == name]
        if row.empty:
            continue
        mk, col = BASELINE_MARKERS.get(name, ("o", "#999999"))
        ax.scatter(row["Ef_MAE"], row["mean_xi"],
                   marker=mk, color=col, s=55, zorder=5,
                   linewidths=0.8, edgecolors="white")
        ax.annotate(name, xy=(row["Ef_MAE"].values[0], row["mean_xi"].values[0]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, color=col)

    # ── Sweep series ──────────────────────────────────────────────────────────
    legend_handles = []
    for label, cfg in SERIES.items():
        prefix  = cfg["prefix"]
        exclude = cfg["exclude"]
        color   = cfg["color"]
        mk      = cfg["marker"]
        ls      = cfg["ls"]
        skip    = LABEL_ALPHA_SKIP.get(label, 1)

        mask = df["model"].str.startswith(prefix)
        if exclude:
            mask &= ~df["model"].str.contains(exclude, regex=True)
        sub = df[mask].copy()
        if sub.empty:
            continue

        sub["alpha"] = sub["model"].apply(lambda n: extract_alpha(n, prefix))
        sub = sub.sort_values("alpha").reset_index(drop=True)

        # Drop degenerate / diverged runs
        sub = sub[sub["Ef_MAE"] <= EF_MAE_MAX].reset_index(drop=True)
        if sub.empty:
            continue

        if not args.no_pareto:
            plot_df = pareto_frontier(sub)
        else:
            plot_df = sub

        ax.plot(plot_df["Ef_MAE"], plot_df["mean_xi"],
                color=color, ls=ls, lw=1.4, alpha=0.85, zorder=3)
        sc = ax.scatter(plot_df["Ef_MAE"], plot_df["mean_xi"],
                        color=color, marker=mk, s=45, zorder=4,
                        edgecolors="white", linewidths=0.6)

        # Label every skip-th point with α value
        for i, (_, row) in enumerate(plot_df.iterrows()):
            if i % skip == 0:
                ax.annotate(f"{row['alpha']:.1f}",
                            xy=(row["Ef_MAE"], row["mean_xi"]),
                            xytext=(4, 3), textcoords="offset points",
                            fontsize=6, color=color, alpha=0.85)

        legend_handles.append(
            Line2D([0], [0], color=color, marker=mk, ls=ls,
                   markersize=5, linewidth=1.4, label=label)
        )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel(r"$E_f$ MAE  (eV atom$^{-1}$)", fontsize=11)
    yi_label = {"mean_xi": r"Mean $\xi$",
                "median_xi": r"Median $\xi$",
                "rms_xi": r"RMS $\xi$"}.get(args.xi_col, r"$\xi$")
    ax.set_ylabel(yi_label, fontsize=11)
    ax.set_title("Pareto frontiers: formation-energy accuracy vs. interference index",
                 fontsize=10, pad=8)

    # Grey band: ξ < 1 = net cancellation region
    ymin, ymax = ax.get_ylim()
    ax.axhline(1.0, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.fill_between(ax.get_xlim(), ymin, 1.0,
                    color="grey", alpha=0.06, zorder=0)
    ax.text(ax.get_xlim()[1] * 0.98, 0.97,
            "net cancellation", ha="right", va="top",
            fontsize=7, color="grey", style="italic")
    ax.set_ylim(ymin, ymax)

    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9,
              loc="upper left")
    ax.tick_params(labelsize=9)

    xl = ax.get_xlim()[0]
    xr = ax.get_xlim()[1]
    yl = ax.get_ylim()[0]
    yr = ax.get_ylim()[1]
    ax.annotate("", xy=(xl, yl), xytext=(xl + (xr-xl)*0.06, yl),
                arrowprops=dict(arrowstyle="<-", color="black", lw=0.8))
    ax.annotate("", xy=(xl, yl), xytext=(xl, yl + (yr-yl)*0.06),
                arrowprops=dict(arrowstyle="<-", color="black", lw=0.8))

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
