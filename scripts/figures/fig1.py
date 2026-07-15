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

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.lines as mlines

# -- Config -------------------------------------------------------------------
COL_REF = '#6B7280'

RNG    = np.random.default_rng(1)
N_VALS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
N_PER  = int(1e5/len(N_VALS))  
CMAP   = 'RdBu_r'
norm   = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=np.sqrt(10))

# ── Journal style (npj Computational Materials) ──────────────────────────
# source_pt = target_print_pt × (fig_width / journal_col_width)
# _PT_BODY / _PT_SM set the desired print size in pt — increase if fonts look
# too small on screen (dense figures like this one warrant a larger target).
_JOURNAL_COL_W   = 6.69   # 170 mm in inches
_FIG_W           = 11.0   # rendering width for font scaling
_PT_BODY, _PT_SM = 9.0, 7.5   # target print sizes (pt)
_FS    = round(_PT_BODY * _FIG_W / _JOURNAL_COL_W)   # → 16 pt
_FS_SM = round(_PT_SM   * _FIG_W / _JOURNAL_COL_W)   # → 13 pt
_FS_LEG, _FS_CB = _FS, _FS
_FS_PANEL = 20   # panel labels a/b/c — fixed across all figures
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': _FS,
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
arc_phi = np.linspace(0, np.pi / 2, 300)

def draw_panel(ax, key):
    all_xi   = np.concatenate([data[key][N] for N in N_VALS])
    cmap_obj = plt.get_cmap(CMAP)

    xi_rms = np.sqrt(np.mean(all_xi ** 2))

    # Concentric arcs + N labels
    for N in N_VALS:
        R = np.sqrt(N)
        ax.plot(R * np.cos(arc_phi), R * np.sin(arc_phi),
                color='lightgray', lw=1.0, zorder=0)
        ax.text(R * np.cos(0.24) + 0.01 - R * 0.003,
                R * np.sin(0.24),
                f'N={N}', color='gray', fontsize=_FS_SM + 1,
                va='top', ha='left', rotation=-80)

    # Scatter coloured by xi
    all_x, all_y = [], []
    for N in N_VALS:
        xi    = np.clip(data[key][N], 0, np.sqrt(N))
        delta = np.sqrt(np.maximum(N - xi ** 2, 0))
        all_x.extend(xi)
        all_y.extend(delta)
    ax.scatter(all_x, all_y, c=all_xi, cmap=CMAP, norm=norm,
               s=10, alpha=0.35, zorder=3)

    # Per-N RMS-xi diamonds — computed from the sampled data
    for N in N_VALS:
        xi_N = np.clip(data[key][N], 0, np.sqrt(N))
        m    = np.sqrt(np.mean(xi_N ** 2))
        md   = np.sqrt(max(N - m ** 2, 0))
        ax.scatter([m], [md], color='black', s=30, marker='D', zorder=5)

    # Histogram bars — bin the ~10⁵ sampled points directly
    edges = np.linspace(0, CIRC_TOP, 36)
    counts, _ = np.histogram(all_xi, bins=edges, density=True)

    for left, right, h in zip(edges[:-1], edges[1:], counts):
        xi_mid = (left + right) / 2
        bar_h  = h * HIST_SCALE
        ax.bar(left, bar_h, width=right - left, bottom=CIRC_TOP,
               color=cmap_obj(norm(xi_mid)),
               align='edge', edgecolor='none', alpha=0.85,
               zorder=3, clip_on=False)

    # Reference lines -- stop at the top of their overlapping histogram bar
    def _vline_top(x_val):
        for i in range(len(counts)):
            if edges[i] <= x_val < edges[i + 1]:
                return float(CIRC_TOP + counts[i] * HIST_SCALE)
        return float(CIRC_TOP)

    xi1_top = _vline_top(1.0)
    rms_top = _vline_top(xi_rms)
    ax.plot([1.0, 1.0], [0, xi1_top], color=COL_REF, lw=0.9, ls=':',
            alpha=0.7, zorder=4, clip_on=False)
    ax.plot([xi_rms, xi_rms], [0, rms_top], color='black', lw=1.2, ls='--',
            zorder=5, clip_on=False)
    # rms value as text annotation inside the panel
    ax.text(0.98, 0.98, r'$\xi_\mathrm{rms} =' + rf'{xi_rms:.2f}$',
            transform=ax.transAxes, fontsize=_FS_SM, ha='right', va='top')

    # -- Frame wraps circle region only: ylim = (0, CIRC_TOP) ----------------
    ax.set_xlim(0, CIRC_TOP)
    ax.set_ylim(0, CIRC_TOP)
    ax.set_xticks(TICK_VALS)
    ax.set_yticks(TICK_VALS)
    ax.set_xlabel(r'$\xi = \sqrt{N}\cos\varphi$', fontsize=_FS)
    ax.set_ylabel(r'$\eta = \sqrt{N}\sin\varphi$', fontsize=_FS)
    ax.set_aspect('equal')

# -- Figure -------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.5, 13))
fig.subplots_adjust(hspace=0.35)

draw_panel(ax_a, 'cancel')
draw_panel(ax_b, 'stat')

# Remove x-axis label and tick labels from panel a (shared with b)
ax_a.set_xlabel('')
ax_a.tick_params(labelbottom=False)

# Shared legend — diamond + ξ=1 line, shared across both panels
_leg_handles = [
    mlines.Line2D([], [], color='black', marker='D', markersize=5,
                  linestyle='None', label=r'$\sqrt{\langle\xi^2\rangle}_N$'),
    mlines.Line2D([], [], color=COL_REF, lw=0.9, ls=':', alpha=0.7,
                  label=r'$\xi = 1$'),
    mlines.Line2D([], [], color='black', lw=1.2, ls='--',
                  label=r'$\xi_\mathrm{rms}$'),
]
fig.legend(handles=_leg_handles, loc='lower center', ncol=3,
           fontsize=_FS_LEG, framealpha=0.9, bbox_to_anchor=(0.5, 0.025))

# Colorbar below legend
cbar_ax = fig.add_axes([0.15, -0.015, 0.70, 0.012])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$\xi$', fontsize=_FS_CB)
cbar.set_ticks([0, 1, np.sqrt(10)])
cbar.set_ticklabels(['0', '1', r'$\sqrt{10}$'])

# Panel labels
for ax, lbl in [(ax_a, 'a'), (ax_b, 'b')]:
    ax.text(-0.18, 1.15, lbl, transform=ax.transAxes,
            fontsize=_FS_PANEL, fontweight='bold', va='bottom', ha='left', clip_on=False)

plt.tight_layout(rect=[0, 0.07, 1, 1])


def main():
    import argparse
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(_REPO, "figures", "figure1.png"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
