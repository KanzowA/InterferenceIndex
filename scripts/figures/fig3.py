"""
fig3.py
------------------------
Figure 3: Gradient descent field comparison in 2D residual space (N=2).
  (a) -∇MSE   = -2r
  (b) -∇ξ²   = -(2/Q)(S1-ξ²r)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ── Journal style (npj Computational Materials) ──────────────────────────
# source_pt = target_print_pt × (fig_width / journal_col_width)
# Targets 7 pt (body) and 6 pt (minor) at 170 mm double-column.
_COL_W = 6.69          # 170 mm in inches
_FIG_W         = 14.0          # rendering width for font scaling
_FS    = round(7.0 * _FIG_W / _COL_W)   # → 13 pt  (body / axis labels)
_FS_SM = round(6.0 * _FIG_W / _COL_W)   # → 11 pt  (minor annotations)
_FS_LEG, _FS_CB = _FS, _FS
_FS_PANEL = 30   # panel labels a/b/c — fixed across all figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   _FS,
    'axes.linewidth': 0.8,
})

CMAP   = 'RdBu_r'
SQ2    = np.sqrt(2)
EXTENT = 2.3
norm   = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=SQ2)

# ── Background: ξ on a fine grid ─────────────────────────────────────────────
NF = 500
xf = np.linspace(-EXTENT, EXTENT, NF)
yf = np.linspace(-EXTENT, EXTENT, NF)
R1f, R2f = np.meshgrid(xf, yf)
Qf  = R1f**2 + R2f**2
xi_bg = np.where(Qf > 1e-8, np.abs(R1f + R2f) / np.sqrt(Qf), 0.0)

# ── Arrow grids — left panel (10×10), right panel (13×13) ──────────
def _arrow_grid(NS):
    xs = np.linspace(-EXTENT, EXTENT, NS)
    ys = np.linspace(-EXTENT, EXTENT, NS)
    R1, R2 = np.meshgrid(xs, ys)
    return xs, ys, R1, R2

xs_a, ys_a, R1s_a, R2s_a = _arrow_grid(10)
near_a = R1s_a**2 + R2s_a**2 < 0.07
U_mse = np.where(~near_a, -2.0 * R1s_a, np.nan)
V_mse = np.where(~near_a, -2.0 * R2s_a, np.nan)

xs_b, ys_b, R1s_b, R2s_b = _arrow_grid(13)
Ss_b  = R1s_b + R2s_b
Qs_b  = R1s_b**2 + R2s_b**2
near_b = Qs_b < 0.07
_Qb   = np.where(Qs_b > 1e-8, Qs_b, 1.0)
xi2b  = np.where(Qs_b > 1e-8, Ss_b**2 / _Qb, 0.0)
U_xi2 = np.where(~near_b, -(2.0/_Qb) * (Ss_b - xi2b * R1s_b), np.nan)
V_xi2 = np.where(~near_b, -(2.0/_Qb) * (Ss_b - xi2b * R2s_b), np.nan)

def _unit(U, V):
    """Normalise to unit vectors; NaN where undefined."""
    mag = np.sqrt(U**2 + V**2)
    mag = np.where(mag < 1e-8, np.nan, mag)
    return U / mag, V / mag

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.5, 13))
gs  = gridspec.GridSpec(2, 1,
                        height_ratios=[1, 1],
                        hspace=0.25,
                        left=0.12, right=0.95, top=0.96, bottom=0.10)
axes   = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.012])

ext = EXTENT * 0.93   # reference-line extent
phi_c = np.linspace(0, 2 * np.pi, 400)

panels = [
    (axes[0], U_mse,  V_mse,  xs_a, ys_a, 'a',
     r'$\mathcal{L}_\mathrm{MSE} = \|\mathbf{r}\|^2$',
     r'$-\nabla_\mathbf{r}\,\mathcal{L}_\mathrm{MSE}$'),
    (axes[1], U_xi2,  V_xi2,  xs_b, ys_b, 'b',
     r'$\mathcal{L}_{\xi^2} = \xi^2 = N\cdot \cos^2 \varphi$',
     r'$-\nabla_\mathbf{r}\,\mathcal{L}_{\xi^2}$'),
]

for ax, U, V, xs, ys, panel_lbl, title_str, eq_str in panels:

    # ── ξ background ─────────────────────────────────────────────────────────
    ax.pcolormesh(xf, yf, xi_bg, cmap=CMAP, norm=norm,
                  rasterized=True, zorder=0)

    # ── Quiver (unit-normalised direction, regular grid) ─────────────────────
    Un, Vn = _unit(U, V)
    ax.quiver(xs, ys, Un, Vn,
              color='black', alpha=0.75,
              scale=14, width=0.01,
              headwidth=5, headlength=5, headaxislength=3,
              zorder=3)

    # ── Reference lines ───────────────────────────────────────────────────────

    ax.axvline(0.00, ymin=0.00, ymax=1.00, color = 'k', alpha = 0.3)
    ax.axhline(0.00, xmin=0.00, xmax=1.00, color = 'k', alpha = 0.3)

    # origin dot
    ax.plot(0, 0, 'ko', ms=3.5, zorder=5)

    # ── Labels / equations ────────────────────────────────────────────────────
    ax.text(-0.12, 1.05, panel_lbl, transform=ax.transAxes,
            fontsize=_FS_PANEL, fontweight='bold', va='bottom')
    #ax.set_title(title_str, fontsize=_FS, pad=7)

    ax.text(0.97, 0.03, eq_str,
            transform=ax.transAxes,
            fontsize=_FS_SM, va='bottom', ha='right', linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.45', fc='white',
                      alpha=0.88, ec='#aaaaaa', lw=0.7))

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect('equal')
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r'$r_1$', fontsize=_FS)
    ax.set_ylabel(r'$r_2$', fontsize=_FS)
    ax.tick_params(top=False, right=False)

# ── Shared colorbar (horizontal, between panels) ──────────────────────────────
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$\xi$', fontsize=_FS_CB, labelpad=2)
cbar.set_ticks([0, 1, SQ2])
cbar.set_ticklabels(['0', '1', r'$\sqrt{2}$'])

def main():
    import argparse
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(_REPO, "figures", "figure3.png"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
