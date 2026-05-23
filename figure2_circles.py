"""
figure2_circles.py
------------------
Figure 2: Propagation circles + ξ histograms for two limiting regimes.

  a) Error cancellation  — propagation circles (ξ ≪ 1)
  b) Error cancellation  — ξ histogram
  c) Statistical baseline — propagation circles (ξ ≈ 1, i.i.d. null)
  d) Statistical baseline — ξ histogram

Dummy data only — no models, fully reproducible via RNG seed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ── Config ────────────────────────────────────────────────────────────────────
COL_A   = '#2563EB'   # cancellation: blue (matches COL_CANCEL in Figure 1)
COL_C   = '#9CA3AF'   # statistical:  neutral grey
COL_REF = '#6B7280'   # ξ = 1 reference line

RNG    = np.random.default_rng(42)
N_VALS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
N_PER  = 1000   # dummy points per arc ring
CMAP   = 'RdBu_r'
norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=np.sqrt(10))

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.linewidth': 0.8})

# ── Dummy data generators ─────────────────────────────────────────────────────
def gen_cancellation(N, n, rng):
    xi = rng.exponential(scale=0.35, size=n)
    xi = np.clip(xi, 0, np.sqrt(N))
    return xi

def gen_statistical(N, n, rng):
    """i.i.d. N(0,1) residuals — the null hypothesis, E[ξ²] = 1."""
    r     = rng.standard_normal((n, N))
    denom = np.sqrt((r ** 2).sum(axis=1))
    return np.abs(r.sum(axis=1)) / np.where(denom < 1e-9, 1e-9, denom)

# ── Pre-generate all data (shared between circle and histogram panels) ─────────
data = {}
for key, fn in [('cancel', gen_cancellation), ('stat', gen_statistical)]:
    data[key] = {}
    for N in N_VALS:
        data[key][N] = fn(N, N_PER, RNG)

# ── Figure: 2 rows × 2 cols ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
gs  = gridspec.GridSpec(2, 2,
                        width_ratios=[1.15, 0.85],
                        hspace=0.5,
                        wspace=0.38)
ax_a = fig.add_subplot(gs[0, 0])   # a) cancellation circles
ax_b = fig.add_subplot(gs[0, 1])   # b) cancellation histogram
ax_c = fig.add_subplot(gs[1, 0])   # c) statistical circles
ax_d = fig.add_subplot(gs[1, 1])   # d) statistical histogram

# ── Helper: propagation circle panel ─────────────────────────────────────────
arc_theta = np.linspace(0, np.pi / 2, 300)

def draw_circles(ax, key, col):
    all_xi = np.concatenate([data[key][N] for N in N_VALS])
    mean_xi = np.sqrt(np.mean(all_xi**2))
    # Concentric arcs + N labels (style matches propagation_circles.png)
    for N in N_VALS:
        R = np.sqrt(N)
        ax.plot(R * np.cos(arc_theta), R * np.sin(arc_theta),
                color='lightgray', lw=1.0, zorder=0)
        ax.text(R * np.cos(0.24) + 0.04 - R * 0.003,
                R * np.sin(0.24),
                f'N={N}', color='gray', fontsize=7,
                va='top', ha='left', rotation=-80)

    # Scatter data points (single colour, no heatmap)
    all_x, all_y = [], []
    for N in N_VALS:
        xi    = np.clip(data[key][N], 0, np.sqrt(N))
        delta = np.sqrt(np.maximum(N - xi ** 2, 0))
        all_x.extend(xi)
        all_y.extend(delta)

    ax.scatter(all_x, all_y, c=all_xi, cmap=CMAP, norm=norm, s=7, alpha=0.45, zorder=3)

    # Mean ξ per N as black diamond
    for N in N_VALS:
        xi_stab = np.clip(
            (gen_cancellation if key == 'cancel' else gen_statistical)
            (N, N_PER * 8, RNG), 0, np.sqrt(N))
        m  = np.sqrt(np.mean(xi_stab**2))
        md = np.sqrt(max(N - m ** 2, 0))
        if N == 2:
            ax.scatter([m], [md], color='black', s=30, marker='D', zorder=5, label = r'$\sqrt{\langle\xi^2\rangle_N}$')
        ax.scatter([m], [md], color='black', s=30, marker='D', zorder=5)

    # ξ = 1 reference line
    lim = np.sqrt(max(N_VALS)) + 0.42
    ax.axvline(1.0, color=COL_REF, lw=0.9, ls=':', alpha=0.7, zorder=2, label = r'$\xi=1$')
    #ax.text(1.03, lim - 0.30, r'$\xi=1$', color=COL_REF, fontsize=9)

    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel(r'$\xi = \sqrt{N}\cos\theta$', fontsize=11)
    ax.set_ylabel(r'$\sqrt{N}\sin\theta$', fontsize=11)
    ax.set_aspect('equal')
    ax.legend()
# ── Helper: ξ histogram panel ─────────────────────────────────────────────────
HIST_XLIM = (0, 2.6)
HIST_YLIM = (0, 2.6)

def draw_hist(ax, key, col):
    all_xi = np.concatenate([data[key][N] for N in N_VALS])
    mean_xi = np.sqrt(np.mean(all_xi**2))

    counts, edges = np.histogram(all_xi, bins=35, range=HIST_XLIM, density=True)
    cmap_obj = plt.get_cmap(CMAP)
    for left, right, h in zip(edges[:-1], edges[1:], counts):
        xi_mid = (left + right) / 2
        ax.bar(left, h, width=right - left, color=cmap_obj(norm(xi_mid)),
               align='edge', edgecolor='none', alpha=0.85, zorder=3)

    # ξ = 1 reference
    ax.axvline(1.0, color=COL_REF, lw=1.0, ls=':', alpha=0.8, zorder=4,
               label=r'$\xi = 1$')
    # Mean
    ax.axvline(mean_xi, color='black', lw=1.2, ls='--', zorder=5,
               label=r'$\sqrt{\langle{\xi}^2\rangle} = ' + rf'{mean_xi:.2f}$')

    ax.set_xlim(*HIST_XLIM), ax.set_ylim(*HIST_YLIM)
    ax.set_xlabel(r'$\xi$', fontsize=13)
    ax.set_ylabel(r'$\rho(\xi)$', fontsize=11)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.20)

# ── Draw all four panels ───────────────────────────────────────────────────────
draw_circles(ax_a, 'cancel', COL_A)
draw_hist   (ax_b, 'cancel', COL_A)
draw_circles(ax_c, 'stat',   COL_C)
draw_hist   (ax_d, 'stat',   COL_C)

# ── Panel labels (position and size matching figure1_geometry.py) ─────────────
for ax, lbl in [(ax_a, 'a)'), (ax_b, 'b)'), (ax_c, 'c)'), (ax_d, 'd)')]:
    ax.text(-0.25, 1.15, lbl, transform=ax.transAxes,
            fontsize=25, fontweight='bold', va='bottom', ha='left',
            clip_on=False)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig('figure2_circles.png', dpi=150, bbox_inches='tight')
print('Saved → figure2_circles.png')
plt.show()
