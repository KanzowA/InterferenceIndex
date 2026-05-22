"""
plot_gamma_sweep.py
-------------------
Quick plots of GammaLoss λ-sweep results from interference_summary.csv.

Three subplots (x-axis = λ in all cases):
  1. Ef MAE  (formation enthalpy, per compound)
  2. Ed MAE  (decomposition enthalpy, per reaction)
  3. Mean γ  and  Mean γ²  (simultaneously, same axis)

Usage (from TestStabilityMl dir):
    python plot_gamma_sweep.py
    python plot_gamma_sweep.py interference_summary.csv   # explicit path
"""

import sys
import csv
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ─────────────────────────────────────────────────────────────────────
SUMMARY_CSV = sys.argv[1] if len(sys.argv) > 1 else "interference_summary.csv"
OUT_FILE    = "gamma_sweep.png"

# Reference lines from the Bartel models (drawn as horizontal dashed lines)
# Set to None to omit a reference
REFERENCES = {
    "ElFrac":   {"Ef_MAE": 0.2314, "Ed_MAE": 0.1722, "mean_gamma": 0.5874, "color": "steelblue"},
    "ElemNet":  {"Ef_MAE": 0.0971, "Ed_MAE": 0.1158, "mean_gamma": 0.8841, "color": "darkorange"},
    "CGCNN":    {"Ef_MAE": 0.0340, "Ed_MAE": 0.0419, "mean_gamma": 0.8935, "color": "forestgreen"},
}

# ── Load CSV ────────────────────────────────────────────────────────────────────
if not os.path.exists(SUMMARY_CSV):
    print(f"ERROR: {SUMMARY_CSV} not found. Run interference_score.py first.")
    sys.exit(1)

lambdas   = []
ef_maes   = []
ed_maes   = []
mean_gammas = []

with open(SUMMARY_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row["model"]
        if not model.startswith("GammaLoss_"):
            continue
        try:
            lam = float(model.split("_", 1)[1])
        except ValueError:
            continue
        lambdas.append(lam)
        ef_maes.append(float(row["Ef_MAE"]))
        ed_maes.append(float(row["Ed_MAE"]))
        mean_gammas.append(float(row["mean_gamma"]))

if not lambdas:
    print("No GammaLoss_* rows found in CSV.")
    sys.exit(1)

# Sort by lambda
order = sorted(range(len(lambdas)), key=lambda i: lambdas[i])
lambdas     = [lambdas[i]     for i in order]
ef_maes     = [ef_maes[i]     for i in order]
ed_maes     = [ed_maes[i]     for i in order]
mean_gammas = [mean_gammas[i] for i in order]

# Derived: γ² = (mean γ)²   [square of the mean — proxy for mean γ²]
mean_gamma2 = [g**2 for g in mean_gammas]

# ── Plot ────────────────────────────────────────────────────────────────────────
ACCENT = "#2563EB"     # blue for primary line
ACCENT2 = "#DC2626"    # red for γ²
MARKER = "o"
MS     = 7
LW     = 2.0

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("GammaLoss λ-sweep: Ef accuracy vs. error cancellation",
             fontsize=13, fontweight="bold", y=1.02)

x_ticks = lambdas
x_pad   = 0.04

# ── Subplot 1: Ef MAE ──────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(lambdas, ef_maes, color=ACCENT, marker=MARKER, ms=MS, lw=LW,
        label="GammaLoss")
for ref_name, ref in REFERENCES.items():
    ax.axhline(ref["Ef_MAE"], color=ref["color"], lw=1.2, ls="--",
               alpha=0.75, label=ref_name)
ax.set_xlabel("λ  (γ² loss weight)", fontsize=11)
ax.set_ylabel("Ef  MAE  (eV/atom)", fontsize=11)
ax.set_title("Formation enthalpy accuracy", fontsize=11)
ax.set_xticks(x_ticks)
ax.set_xlim(-x_pad, max(lambdas) + x_pad)
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, alpha=0.25)

# ── Subplot 2: Ed MAE ──────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(lambdas, ed_maes, color=ACCENT, marker=MARKER, ms=MS, lw=LW,
        label="GammaLoss")
# Mark the minimum
min_i = ed_maes.index(min(ed_maes))
ax.scatter([lambdas[min_i]], [ed_maes[min_i]],
           color=ACCENT, edgecolors="black", s=80, zorder=5,
           label=f"best  λ={lambdas[min_i]}")
for ref_name, ref in REFERENCES.items():
    ax.axhline(ref["Ed_MAE"], color=ref["color"], lw=1.2, ls="--",
               alpha=0.75, label=ref_name)
ax.set_xlabel("λ  (γ² loss weight)", fontsize=11)
ax.set_ylabel("Ed  MAE  (eV/atom)", fontsize=11)
ax.set_title("Decomposition enthalpy accuracy", fontsize=11)
ax.set_xticks(x_ticks)
ax.set_xlim(-x_pad, max(lambdas) + x_pad)
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, alpha=0.25)

# ── Subplot 3: Mean γ and Mean γ² ─────────────────────────────────────────────
ax = axes[2]
ax.plot(lambdas, mean_gammas,  color=ACCENT,  marker=MARKER,  ms=MS, lw=LW,
        label=r"mean $\gamma$")
ax.plot(lambdas, mean_gamma2,  color=ACCENT2, marker="s",     ms=MS, lw=LW,
        ls="--", label=r"mean $\gamma^2$  $\approx$ (mean $\gamma)^2$")
# Reference lines for γ only (γ² not in CSV for Bartel models)
for ref_name, ref in REFERENCES.items():
    ax.axhline(ref["mean_gamma"], color=ref["color"], lw=1.2, ls=":",
               alpha=0.75, label=f"{ref_name} γ")
ax.set_xlabel("λ  (γ² loss weight)", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title(r"Error cancellation  ($\gamma \to 0$ = better)", fontsize=11)
ax.set_xticks(x_ticks)
ax.set_xlim(-x_pad, max(lambdas) + x_pad)
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, alpha=0.25)

# ── Annotate λ values on all plots ────────────────────────────────────────────
for ax_i, (ax, ys) in enumerate(zip(axes, [ef_maes, ed_maes, mean_gammas])):
    for lam, y in zip(lambdas, ys):
        offset = 0.003 * (max(ys) - min(ys)) * 3
        ax.annotate(f"λ={lam}", xy=(lam, y),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7, color="gray")

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_FILE}")
plt.show()
