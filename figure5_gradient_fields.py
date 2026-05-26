"""
figure5_gradient_fields.py
--------------------------
Figure 5: Gradient field comparison in 2D residual space (N=2).

  (a) -∇MSE   = -2r          — purely radial,     collapses toward origin
  (b) -∇ξ²   = -(2/Q)(S1-ξ²r) — purely tangential, rotates toward ξ = 0

Background in both panels: ξ heatmap (RdBu_r / TwoSlopeNorm),
consistent with Figures 1–3.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── Appearance ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   11,
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

# ── Stream grid ───────────────────────────────────────────────────────────────
NS = 28
xs = np.linspace(-EXTENT, EXTENT, NS)
ys = np.linspace(-EXTENT, EXTENT, NS)
R1s, R2s = np.meshgrid(xs, ys)
Ss  = R1s + R2s
Qs  = R1s**2 + R2s**2
near_orig = Qs < 0.07

# (a) -∇MSE = -2r
U_mse = np.where(~near_orig, -2.0 * R1s, np.nan)
V_mse = np.where(~near_orig, -2.0 * R2s, np.nan)

# (b) -∇ξ²
_Q    = np.where(Qs > 1e-8, Qs, 1.0)
xi2s  = np.where(Qs > 1e-8, Ss**2 / _Q, 0.0)
U_xi2 = np.where(~near_orig, -(2.0/_Q) * (Ss - xi2s * R1s), np.nan)
V_xi2 = np.where(~near_orig, -(2.0/_Q) * (Ss - xi2s * R2s), np.nan)

def _unit(U, V):
    """Normalise to unit vectors; NaN where undefined."""
    mag = np.sqrt(U**2 + V**2)
    mag = np.where(mag < 1e-8, np.nan, mag)
    return U / mag, V / mag

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.subplots_adjust(left=0.07, right=0.88, wspace=0.32)

ext = EXTENT * 0.93   # reference-line extent
theta_c = np.linspace(0, 2 * np.pi, 400)

panels = [
    (axes[0], U_mse,  V_mse,  'a)',
     r'$\mathcal{L}_\mathrm{MSE} = \|\mathbf{r}\|^2$',
     r'$\nabla_\mathbf{r}\,\mathcal{L}_\mathrm{MSE} = 2\mathbf{r}$'),
    (axes[1], U_xi2,  V_xi2,  'b)',
     r'$\mathcal{L}_{\xi^2} = \xi^2 = \dfrac{S^2}{Q}$',
     (r'$\nabla_\mathbf{r}\,\xi^2 = \dfrac{2}{Q}'
      r'\!\left(S\mathbf{1} - \xi^2\mathbf{r}\right)$'
      '\n'
      r'$S = \sum r_i,\quad Q = \|\mathbf{r}\|^2$')),
]

for ax, U, V, panel_lbl, title_str, eq_str in panels:

    # ── ξ background ─────────────────────────────────────────────────────────
    ax.pcolormesh(xf, yf, xi_bg, cmap=CMAP, norm=norm,
                  rasterized=True, zorder=0)

    # ── Streamplot (unit-normalised direction) ────────────────────────────────
    Un, Vn = _unit(U, V)
    ax.streamplot(xs, ys, Un, Vn,
                  color='black', linewidth=0.6,
                  density=2.2, arrowsize=0.7, zorder=3)

    # ── Reference lines ───────────────────────────────────────────────────────
    # Aggregation direction 1 = (1,1)/√2  (ξ maximum)
    ax.plot([-ext/SQ2, ext/SQ2], [-ext/SQ2, ext/SQ2],
            '--', color='k', lw=1.3, alpha=0.55, zorder=2)
    ax.text(ext/SQ2 + 0.04, ext/SQ2 - 0.12, r'$\mathbf{1}$',
            fontsize=10, color='k', alpha=0.7)

    # ξ = 0 line  (1,-1)/√2
    ax.plot([-ext/SQ2, ext/SQ2], [ext/SQ2, -ext/SQ2],
            ':', color='k', lw=1.3, alpha=0.45, zorder=2)
    ax.text(ext/SQ2 + 0.04, -ext/SQ2 + 0.10, r'$\xi\!=\!0$',
            fontsize=9, color='k', alpha=0.6)

    # ξ = 1 circle
    ax.plot(np.cos(theta_c), np.sin(theta_c),
            '--', color='k', lw=0.7, alpha=0.3, zorder=2)

    # origin dot
    ax.plot(0, 0, 'ko', ms=3.5, zorder=5)

    # ── Labels / equations ────────────────────────────────────────────────────
    ax.text(-0.14, 1.04, panel_lbl, transform=ax.transAxes,
            fontsize=19, fontweight='bold', va='bottom')
    ax.set_title(title_str, fontsize=12, pad=7)

    ax.text(0.97, 0.03, eq_str,
            transform=ax.transAxes,
            fontsize=9.5, va='bottom', ha='right', linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.45', fc='white',
                      alpha=0.88, ec='#aaaaaa', lw=0.7))

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect('equal')
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r'$r_1$', fontsize=12)
    ax.set_ylabel(r'$r_2$', fontsize=12)
    ax.tick_params(top=False, right=False)

# ── Shared colorbar ───────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.70])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r'$\xi$', fontsize=13)
cbar.set_ticks([0, 1, SQ2])
cbar.set_ticklabels(['0', '1', r'$\sqrt{2}$'])

plt.savefig('figure5_gradient_fields.png', dpi=150, bbox_inches='tight')
print('Saved → figure5_gradient_fields.png')
