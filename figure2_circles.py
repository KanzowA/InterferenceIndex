"""
figure2_circles.py
------------------
Figure 2: Two side-by-side panels (cancellation / statistical baseline).

Each panel is a SINGLE axes whose frame wraps the circle scatter region only
(ylim = 0 to CIRC_TOP).  The x-axis and its label sit at the bottom; all four
spines form a box around the circles.  The xi histogram bars rise above the
top spine using clip_on=False -- they float outside the frame to show the
cumulative xi distribution.

Dummy data only -- fully reproducible via RNG seed.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -- Config -------------------------------------------------------------------
COL_REF = '#6B7280'

RNG    = np.random.default_rng(42)
N_VALS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
N_PER  = 1000
CMAP   = 'RdBu_r'
norm   = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=np.sqrt(10))

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.linewidth': 0.8})

TICK_VALS = [0, 1, 2, 3]
CIRC_TOP  = 3.3        # axes ylim top = top spine = base of histogram

HIST_DENS_MAX = 2.6    # fixed density scale (same for both panels)
HIST_H        = 1.1    # height reserved for histogram in data coords
HIST_SCALE    = HIST_H / HIST_DENS_MAX
FULL_H        = CIRC_TOP + HIST_H   # y extent including histogram

# -- Dummy data generators ----------------------------------------------------
def gen_cancellation(N, n, rng):
    return np.clip(rng.exponential(scale=0.35, size=n), 0, np.sqrt(N))

def gen_statistical(N, n, rng):
    r     = rng.standard_normal((n, N))
    denom = np.sqrt((r ** 2).sum(axis=1))
    return np.abs(r.sum(axis=1)) / np.where(denom < 1e-9, 1e-9, denom)

GEN  = {'cancel': gen_cancellation, 'stat': gen_statistical}
data = {}
for key, fn in GEN.items():
    data[key] = {N: fn(N, N_PER, RNG) for N in N_VALS}

# -- Panel drawing ------------------------------------------------------------
arc_theta = np.linspace(0, np.pi / 2, 300)

def draw_panel(ax, key):
    all_xi  = np.concatenate([data[key][N] for N in N_VALS])
    mean_xi = np.sqrt(np.mean(all_xi ** 2))
    cmap_obj = plt.get_cmap(CMAP)

    # Concentric arcs + N labels
    for N in N_VALS:
        R = np.sqrt(N)
        ax.plot(R * np.cos(arc_theta), R * np.sin(arc_theta),
                color='lightgray', lw=1.0, zorder=0)
        ax.text(R * np.cos(0.24) + 0.04 - R * 0.003,
                R * np.sin(0.24),
                f'N={N}', color='gray', fontsize=7,
                va='top', ha='left', rotation=-80)

    # Scatter coloured by xi
    all_x, all_y = [], []
    for N in N_VALS:
        xi    = np.clip(data[key][N], 0, np.sqrt(N))
        delta = np.sqrt(np.maximum(N - xi ** 2, 0))
        all_x.extend(xi)
        all_y.extend(delta)
    ax.scatter(all_x, all_y, c=all_xi, cmap=CMAP, norm=norm,
               s=7, alpha=0.5, zorder=3)

    # Per-N RMS-xi diamonds
    first = True
    for N in N_VALS:
        xi_s = np.clip(GEN[key](N, N_PER * 8, RNG), 0, np.sqrt(N))
        m    = np.sqrt(np.mean(xi_s ** 2))
        md   = np.sqrt(max(N - m ** 2, 0))
        kw   = dict(color='black', s=30, marker='D', zorder=5)
        if first:
            ax.scatter([m], [md], label=r'$\sqrt{\langle\xi^2\rangle}_N$', **kw)
            first = False
        else:
            ax.scatter([m], [md], **kw)

    # Histogram bars -- outside the frame (clip_on=False, bottom=CIRC_TOP)
    counts, edges = np.histogram(all_xi, bins=35, range=(0, CIRC_TOP), density=True)
    for left, right, h in zip(edges[:-1], edges[1:], counts):
        xi_mid = (left + right) / 2
        bar_h  = h * HIST_SCALE
        ax.bar(left, bar_h, width=right - left, bottom=CIRC_TOP,
               color=cmap_obj(norm(xi_mid)),
               align='edge', edgecolor='none', alpha=0.85,
               zorder=3, clip_on=False)

    # Reference lines through circle region AND histogram (clip_on=False)
    ax.plot([1.0, 1.0], [0, FULL_H], color=COL_REF, lw=0.9, ls=':',
            alpha=0.7, zorder=4, clip_on=False, label=r'$\xi = 1$')
    ax.plot([mean_xi, mean_xi], [0, FULL_H], color='black', lw=1.2, ls='--',
            zorder=5, clip_on=False,
            label=r'$\sqrt{\langle\xi^2\rangle} = ' + rf'{mean_xi:.2f}$')

    # -- Frame wraps circle region only: ylim = (0, CIRC_TOP) ----------------
    ax.set_xlim(0, CIRC_TOP)
    ax.set_ylim(0, CIRC_TOP)
    ax.set_xticks(TICK_VALS)
    ax.set_yticks(TICK_VALS)
    ax.set_xlabel(r'$\xi = \sqrt{N}\cos\theta$', fontsize=11)
    ax.set_ylabel(r'$\eta = \sqrt{N}\sin\theta$', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper right')

# -- Figure -------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 8))

draw_panel(ax_a, 'cancel')
draw_panel(ax_b, 'stat')

# Panel labels
for ax, lbl in [(ax_a, 'a)'), (ax_b, 'b)')]:
    ax.text(-0.20, 1.25, lbl, transform=ax.transAxes,
            fontsize=25, fontweight='bold', va='bottom', ha='left', clip_on=False)

plt.tight_layout()
plt.savefig('figure2_circles.png', dpi=150, bbox_inches='tight')
print('Saved -> figure2_circles.png')
