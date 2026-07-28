"""Figure 4: metrics against regularisation strength lam, one line per model.

    (a) Hf MAE   (b) Hd MAE   (c) F1   (d) xi_rms
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
_COL_W = 6.69
_FIG_W         = 12.0
_FS    = round(7.0 * _FIG_W / _COL_W)
_FS_SM = round(6.0 * _FIG_W / _COL_W)
_FS_LEG = _FS
_FS_PANEL = 20
plt.rcParams.update({"font.family": "sans-serif", "font.size": _FS,
                     "axes.linewidth": 0.8})

# Series definitions
SERIES = {
    "iiLoss+PCGrad": dict(prefix="iiLoss_pcgrad_", exclude=None,
                          color="#E61D60", marker="s", markersize = 14, ls="--", lw=1.8,
                          anchor="iiLoss_save_0.0"),
    "HdLoss+PCGrad": dict(prefix="HdLoss_pcgrad_", exclude=None,
                          color="#1EACE4", marker="o", markersize = 14, ls="--", lw=1.8,
                          anchor="HdLoss_0.0"),
}

ALPHA_MAX = 0.8

# Panels
METRICS = [
    ("Hf_MAE", r"$\mathrm{MAE}\left(\Delta_\mathrm{f}H\right)$ / eV atom$^{-1}$"),
    ("Hd_MAE", r"$\mathrm{MAE}\left(\Delta_\mathrm{d}H\right)$ / eV atom$^{-1}$"),
    ("F1",     r"$F_1$"),
    ("rms_xi", r"$\xi_\mathrm{rms}$"),
]


def extract_alpha(name, prefix):
    tail = name[len(prefix):]
    m = re.search(r"[\d.]+$", tail)
    return float(m.group()) if m else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(REPO_ROOT, "results", "2026",
                                                       "interference_summary_2026.csv"))
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "figures",
                                                       "figure4.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    fig, axes_2d = plt.subplots(2, 2, figsize=(10, 7), sharey=False, sharex=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.42)

    axes = [axes_2d[0, 0], axes_2d[0, 1], axes_2d[1, 0], axes_2d[1, 1]]

    for ax_idx, (ax, (col, ylabel)) in enumerate(zip(axes, METRICS)):
        row = ax_idx // 2

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
            ax.axvline(0.3, color="gray", ls="--", lw=0.8, alpha=0.4, label = r"$\lambda=0.3$")

            if col in ("Hf_MAE", "Hd_MAE") and alpha >= 1.0:
                ylim_max = 0.265
                over = sub[sub[col] > ylim_max]
                if not over.empty:
                    ax.scatter(over["alpha"], [ylim_max * 0.995] * len(over),
                               marker="^", s=55, color=color, zorder=6, clip_on=False)

        # Axes formatting
        if row == 1:
            ax.set_xlabel(r"$\lambda$", fontsize=_FS)
        else:
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(labelsize=_FS)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        if col == "Hf_MAE":
            ax.set_ylim(0.097, 0.131)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.002))

        elif col == "Hd_MAE":
            ax.set_ylim(0.109, 0.137)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.002))

        elif col == "F1":
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.002))
            ax.set_ylim(0.509,0.541)

        elif col == "rms_xi":
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.set_ylim(0.895, 1.115)
            ax.axhline(1.0, color="#6B7280", lw=0.9, ls=":", alpha=0.6,
                       zorder=1, label=r"$\xi_\mathrm{rms}=1$")

        ax.grid(True, which="major", lw=0.4, alpha=0.5)
        ax.grid(True, which="minor", lw=0.2, alpha=0.3)

        # Baseline dotted line
        baseline_row = df[df["model"] == "iiLoss_0.0"]
        if not baseline_row.empty:
            bval = baseline_row[col].values[0]
            ax.axhline(bval, color="grey", lw=1.0, ls="-", alpha=0.8, zorder=1,
                       label=r"baseline ($\lambda\!=\!0$)")

    # Legend
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen[l] = h

    LEGEND_ORDER = [
        "iiLoss+PCGrad", "HdLoss+PCGrad", r"$\lambda=0.3$", r"baseline ($\lambda\!=\!0$)"
    ]
    ordered    = [seen[l] for l in LEGEND_ORDER if l in seen]
    labels_leg = [l.split("+")[0]       for l in LEGEND_ORDER if l in seen]
    fig.legend(ordered, labels_leg,
               loc="lower center", ncol=2,
               fontsize=_FS_LEG, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))

    # Panel labels
    for ax, letter in zip(axes, ["a", "b", "c", "d"]):
        ax.text(-0.22, 1.05, letter, transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
