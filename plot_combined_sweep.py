"""
plot_combined_sweep.py
----------------------
Side-by-side 3×2 figure comparing:
  LEFT  column — GammaLoss λ-sweep  (normalised γ² objective)
  RIGHT column — EdLoss    α-sweep  (raw Ed-MSE auxiliary loss)

Three rows, shared y-axes within each row:
  Row 1 — MAE(ΔHf)   formation enthalpy accuracy
  Row 2 — MAE(ΔHd)   decomposition enthalpy accuracy  ← key metric
  Row 3 — ⟨γ⟩        mean interference score

Narrative: GammaLoss offers a controlled Pareto trade-off with a clear
Ed MAE minimum at λ=0.1.  EdLoss degrades monotonically because the
raw Ed-MSE term is unscaled — α can never be a genuine fractional weight.

Usage (from TestStabilityMl dir):
    python plot_combined_sweep.py
    python plot_combined_sweep.py interference_summary.csv   # explicit path
"""

import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────────
SUMMARY_CSV = sys.argv[1] if len(sys.argv) > 1 else "interference_summary.csv"
OUT_FILE    = "combined_sweep.png"

# Bartel et al. reference lines
REFERENCES = {
    "ElFrac":  {"Ef_MAE": 0.2314, "Ed_MAE": 0.1722, "mean_gamma": 0.5874,
                "color": "#5B9BD5", "ls": "--"},
    "ElemNet": {"Ef_MAE": 0.0971, "Ed_MAE": 0.1158, "mean_gamma": 0.8841,
                "color": "#ED7D31", "ls": "--"},
    "CGCNN":   {"Ef_MAE": 0.0340, "Ed_MAE": 0.0419, "mean_gamma": 0.8935,
                "color": "#70AD47", "ls": "--"},
}

GAMMA_COLOR = "#2563EB"   # blue for GammaLoss
ED_COLOR    = "#DC2626"   # red for EdLoss
MARKER      = "o"
MS          = 7
LW          = 2.0

# ── Load CSV ──────────────────────────────────────────────────────────────────
if not os.path.exists(SUMMARY_CSV):
    print(f"ERROR: {SUMMARY_CSV} not found. Run interference_score.py first.")
    sys.exit(1)

gamma_data = {}   # lam  -> {Ef_MAE, Ed_MAE, mean_gamma}
ed_data    = {}   # alpha -> {Ef_MAE, Ed_MAE, mean_gamma}

with open(SUMMARY_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row["model"]

        # GammaLoss_<lam>
        if model.startswith("GammaLoss_"):
            try:
                lam = float(model.split("_", 1)[1])
            except ValueError:
                continue
            if lam >= 1.0:   # omit degenerate pure-γ² point
                continue
            gamma_data[lam] = {
                "Ef_MAE":     float(row["Ef_MAE"]),
                "Ed_MAE":     float(row["Ed_MAE"]),
                "mean_gamma": float(row["mean_gamma"]),
            }

        # EdLoss_<alpha>
        elif model.startswith("EdLoss_"):
            try:
                alpha = float(model.split("_", 1)[1])
            except ValueError:
                continue
            ed_data[alpha] = {
                "Ef_MAE":     float(row["Ef_MAE"]),
                "Ed_MAE":     float(row["Ed_MAE"]),
                "mean_gamma": float(row["mean_gamma"]),
            }

if not gamma_data:
    print("No GammaLoss_* rows found — run interference_score.py first.")
    sys.exit(1)
if not ed_data:
    print("No EdLoss_* rows found — run train_ed_sweep.py + interference_score.py first.")
    sys.exit(1)

# Sort by hyperparameter value
lams   = sorted(gamma_data)
alphas = sorted(ed_data)

g_ef  = [gamma_data[l]["Ef_MAE"]     for l in lams]
g_ed  = [gamma_data[l]["Ed_MAE"]     for l in lams]
g_gam = [gamma_data[l]["mean_gamma"] for l in lams]

e_ef  = [ed_data[a]["Ef_MAE"]     for a in alphas]
e_ed  = [ed_data[a]["Ed_MAE"]     for a in alphas]
e_gam = [ed_data[a]["mean_gamma"] for a in alphas]

# ── Log positions for EdLoss x-axis ──────────────────────────────────────────
# All alpha values are strictly positive, so log10 is well-defined.
# Evenly spaced on log scale (e.g. 0.01, 0.1, 1, 10, 100 → -2,-1,0,1,2).
alpha_x = [np.log10(a) for a in alphas]

def _fmt_alpha(a):
    """Format alpha tick label: 1.0 → '1', 0.01 → '0.01', 100.0 → '100'."""
    if a >= 1:
        return str(int(round(a)))
    return str(a)

alpha_tick_labels = [_fmt_alpha(a) for a in alphas]

# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    3, 2,
    figsize=(12, 11),
    sharey="row",          # shared y scale within each row
)
fig.subplots_adjust(hspace=0.08, wspace=0.12)

col_titles = [
    r"GammaLoss: $L = (1-\lambda)\,\frac{\mathrm{MSE}}{\mathrm{MSE}_0} + \lambda\,\frac{\gamma^2}{\gamma^2_0}$",
    r"EdLoss: $L = \frac{\mathrm{MSE}}{\mathrm{MSE}_0} + \alpha\cdot\overline{E_d^{\,2}}$",
]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=10, pad=8)

ylabels = [
    r'MAE$(\Delta H_\mathrm{f})\,/\,\mathrm{eV\,atom^{-1}}$',
    r'MAE$(\Delta H_\mathrm{d})\,/\,\mathrm{eV\,atom^{-1}}$',
    r'$\langle\gamma\rangle$',
]

metrics_gamma = [g_ef,  g_ed,  g_gam]
metrics_ed    = [e_ef,  e_ed,  e_gam]
metric_keys   = ["Ef_MAE", "Ed_MAE", "mean_gamma"]

for row in range(3):
    # ── Left: GammaLoss ──────────────────────────────────────────────────────
    ax = axes[row, 0]
    ax.plot(lams, metrics_gamma[row],
            color=GAMMA_COLOR, marker=MARKER, ms=MS, lw=LW, zorder=3)

    # Best-λ marker on Ed MAE row
    if row == 1:
        best_i = metrics_gamma[row].index(min(metrics_gamma[row]))
        ax.scatter([lams[best_i]], [metrics_gamma[row][best_i]],
                   color=GAMMA_COLOR, edgecolors="black", s=90, zorder=5,
                   label=rf"best $\lambda={lams[best_i]}$")
        ax.legend(fontsize=8, framealpha=0.85, loc="upper left")

    for ref_name, ref in REFERENCES.items():
        ax.axhline(ref[metric_keys[row]], color=ref["color"],
                   lw=1.2, ls=ref["ls"], alpha=0.75, zorder=2)

    ax.set_xlim(-0.04, max(lams) + 0.04)
    ax.set_xticks(lams)
    ax.set_ylabel(ylabels[row], fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25)

    # x-axis label only on bottom row
    if row == 2:
        ax.set_xlabel(r"$\lambda$", fontsize=13)
    else:
        ax.tick_params(labelbottom=False)

    # ── Right: EdLoss ─────────────────────────────────────────────────────────
    ax = axes[row, 1]
    ax.plot(alpha_x, metrics_ed[row],
            color=ED_COLOR, marker=MARKER, ms=MS, lw=LW, zorder=3)

    for ref_name, ref in REFERENCES.items():
        ax.axhline(ref[metric_keys[row]], color=ref["color"],
                   lw=1.2, ls=ref["ls"], alpha=0.75, zorder=2)

    ax.set_xlim(min(alpha_x) - 0.25, max(alpha_x) + 0.25)
    ax.set_xticks(alpha_x)
    ax.set_xticklabels(alpha_tick_labels)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.25)

    if row == 2:
        ax.set_xlabel(r"$\alpha$  (log scale)", fontsize=13)
    else:
        ax.tick_params(labelbottom=False)

# ── Shared legend for reference models ───────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=GAMMA_COLOR, lw=LW, marker=MARKER, ms=MS,
           label="GammaLoss"),
    Line2D([0], [0], color=ED_COLOR, lw=LW, marker=MARKER, ms=MS,
           label="EdLoss"),
] + [
    Line2D([0], [0], color=ref["color"], lw=1.2, ls=ref["ls"],
           alpha=0.75, label=name)
    for name, ref in REFERENCES.items()
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    fontsize=9,
    framealpha=0.9,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Normalised γ² objective vs. raw Ed-MSE auxiliary loss",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.show()
