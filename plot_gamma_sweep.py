"""
plot_gamma_sweep.py
-------------------
Quick plots of InterferenceLoss λ-sweep results from interference_summary.csv.

Three subplots stacked vertically (3×1), shared x-axis (λ):
  1. MAE(ΔHf)   — formation enthalpy accuracy
  2. MAE(ΔHd)   — decomposition enthalpy accuracy
  3. ⟨γ⟩        — mean interference score

Usage (from TestStabilityMl dir):\
    python plot_gamma_sweep.py
    python plot_gamma_sweep.py interference_summary.csv   # explicit path
"""

import sys
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ─────────────────────────────────────────────────────────────────────
SUMMARY_CSV = sys.argv[1] if len(sys.argv) > 1 else "interference_summary.csv"
OUT_FILE    = "xi_sweep.png"

# Reference lines from the Bartel models (drawn as horizontal dashed lines)
REFERENCES = {
    "ElFrac":  {"Ef_MAE": 0.2314, "Ed_MAE": 0.1722, "rms_xi": 0.5874, "color": "steelblue"},
    "ElemNet": {"Ef_MAE": 0.0971, "Ed_MAE": 0.1158, "rms_xi": 0.8841, "color": "darkorange"},
    "CGCNN":   {"Ef_MAE": 0.0340, "Ed_MAE": 0.0419, "rms_xi": 0.8935, "color": "forestgreen"},
}

# ── Load CSV ────────────────────────────────────────────────────────────────────
if not os.path.exists(SUMMARY_CSV):
    print(f"ERROR: {SUMMARY_CSV} not found. Run interference_score.py first.")
    sys.exit(1)

lambdas  = []
ef_maes  = []
ed_maes  = []
rms_xis  = []

with open(SUMMARY_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        model = row["model"]
        if not model.startswith("iiLoss_"):
            continue
        try:
            lam = float(model.split("_", 1)[1])
        except ValueError:
            continue
        if lam >= 1.0:      # omit λ=1 (degenerate pure-γ² objective)
            continue
        lambdas.append(lam)
        ef_maes.append(float(row["Ef_MAE"]))
        ed_maes.append(float(row["Ed_MAE"]))
        rms_xis.append(float(row["rms_xi"]))

if not lambdas:
    print("No iiLoss_* rows found in CSV.")
    sys.exit(1)

# Sort by lambda
order   = sorted(range(len(lambdas)), key=lambda i: lambdas[i])
lambdas = [lambdas[i]  for i in order]
ef_maes = [ef_maes[i]  for i in order]
ed_maes = [ed_maes[i]  for i in order]
rms_xis = [rms_xis[i]  for i in order]

# ── Plot ────────────────────────────────────────────────────────────────────────
ACCENT = "#2563EB"
MARKER = "o"
MS     = 7
LW     = 2.0
x_pad  = 0.04

fig, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
#fig.suptitle(r"InterferenceLoss $\lambda$-sweep", fontsize=13, fontweight="bold")

# ── Subplot 1: Ef MAE ──────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(lambdas, ef_maes, color=ACCENT, marker=MARKER, ms=MS, lw=LW,
        label="iiLoss")
for ref_name, ref in REFERENCES.items():
    ax.axhline(ref["Ef_MAE"], color=ref["color"], lw=1.2, ls="--",
               alpha=0.75, label=ref_name)
ax.set_ylabel(r'MAE$(\Delta H_\mathrm{f})\,/\,\mathrm{eV\,atom^{-1}}$', fontsize=11)
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, alpha=0.25)

# ── Subplot 2: Ed MAE ──────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(lambdas, ed_maes, color=ACCENT, marker=MARKER, ms=MS, lw=LW,
        label="iiLoss")
min_i = ed_maes.index(min(ed_maes))
ax.scatter([lambdas[min_i]], [ed_maes[min_i]],
           color=ACCENT, edgecolors="black", s=80, zorder=5,
           label=rf"best $\lambda={lambdas[min_i]}$")
for ref_name, ref in REFERENCES.items():
    ax.axhline(ref["Ed_MAE"], color=ref["color"], lw=1.2, ls="--",
               alpha=0.75, label=ref_name)
ax.set_ylabel(r'MAE$(\Delta H_\mathrm{d})\,/\,\mathrm{eV\,atom^{-1}}$', fontsize=11)
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, alpha=0.25)

# ── Subplot 3: Mean γ ──────────────────────────────────────────────────────────
ax = axes[2]
ax.plot(lambdas, rms_xis, color=ACCENT, marker=MARKER, ms=MS, lw=LW)
for ref_name, ref in REFERENCES.items():
    ax.axhline(ref["rms_xi"], color=ref["color"], lw=1.2, ls="--",
               alpha=0.75, label=ref_name)
ax.set_xlabel(r"$\lambda$", fontsize=13)
ax.set_ylabel(r'$\sqrt{\langle\xi^2\rangle}$', fontsize=13)
ax.set_xticks(lambdas)
ax.set_xlim(-x_pad, max(lambdas) + x_pad)
ax.legend(fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_FILE}")
plt.show()
