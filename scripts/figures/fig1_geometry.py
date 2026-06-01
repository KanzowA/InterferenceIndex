"""
figure1_geometry.py
-------------------
Figure 1: Geometric interpretation of the interference score ξ (N=2 case).

Two panels:
  a) 2D heatmap of ξ in (r1, r2) residual space
  b) Quarter-arc coloured by ξ, showing how angle θ from 1̂ determines ξ

Colour scheme — edit the four COL_* variables and CMAP to restyle.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

# ── Colour scheme ─────────────────────────────────────────────────────────────
COL_AMPLIFY = '#DC2626'   # ξ > 1  constructive
COL_CANCEL  = '#2563EB'   # ξ < 1  destructive
COL_NULL    = '#FFFFFF'   # ξ = 1  null
COL_ONES    = '#6B7280' # 1̂ arrow
CMAP        = 'RdBu_r'

# ── Journal style (npj Computational Materials) ────────────────────────────────────────────
# Source pt = target_print_pt × (fig_width / journal_col_width).
# All text targets 7 pt (body) and 6 pt (minor) at 170 mm double-column.
_JOURNAL_COL_W = 6.69          # 170 mm in inches
_FIG_W         = 12.0          # this figure's width in inches
_FS    = round(7.0 * _FIG_W / _JOURNAL_COL_W)   # → 13 pt  (body / axis labels)
_FS_SM = round(6.0 * _FIG_W / _JOURNAL_COL_W)   # → 11 pt  (minor annotations)
_FS_LEG, _FS_CB = _FS, _FS
_FS_PANEL = 20   # panel labels a/b/c — fixed across all figures
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': _FS,
                     'axes.linewidth': 0.8})

N    = 2
R    = np.sqrt(N)                      # max ξ = √2
norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=R)
stroke_black = [pe.withStroke(linewidth=2.5, foreground='black')]

# ── Figure & GridSpec ─────────────────────────────────────────────────────────
# Row 0 (tiny): two separate colorbars, one above each panel
# Row 1 (main): panel a | panel b
fig = plt.figure(figsize=(11, 6.2))
gs  = gridspec.GridSpec(2, 2,
                        height_ratios=[0.07, 1],
                        hspace=0.55,
                        wspace=0.65)
ax_cb1 = fig.add_subplot(gs[0, 0])
ax_cb2 = fig.add_subplot(gs[0, 1])
ax1    = fig.add_subplot(gs[1, 0])
ax2    = fig.add_subplot(gs[1, 1])

# ── Separate colorbars above each panel (horizontal, ticks on top) ─────────
sm = ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])

for ax_cb in (ax_cb1, ax_cb2):
    cb = fig.colorbar(sm, cax=ax_cb, orientation='horizontal',
                      ticks=[0, 1.0, R])
    cb.ax.set_xticklabels(['0', '1', r'$\sqrt{2}$'], fontsize=_FS)
    cb.set_label(r'$\xi$', fontsize=_FS_CB, labelpad=3)
    ax_cb.xaxis.set_ticks_position('top')
    ax_cb.xaxis.set_label_position('top')
    pos = ax_cb.get_position()
    ax_cb.set_position([pos.x0, pos.y0 - 0.15, pos.width, pos.height])

# ── Helper: draw annotated arrow with black contour ───────────────────────────
def draw_vec(ax, x1, y1, color, label, loff, c = 'black'):
    ann = ax.annotate('', xy=(x1, y1), xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->', color=color,
                                      lw=2.3, mutation_scale=14), zorder=7)
    ann.arrow_patch.set_path_effects(
        [pe.withStroke(linewidth=4, foreground=c)])
    ax.text(x1 + loff[0], y1 + loff[1], label, color=color,
            fontsize=_FS, fontweight='bold', zorder=8,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=c)])

# ══════════════════════════════════════════════════════════════════════════════
# Panel a — 2D heatmap of ξ in (r1, r2) space
# ══════════════════════════════════════════════════════════════════════════════
lim, res = 1.1, 500
rv       = np.linspace(-lim, lim, res)
R1, R2   = np.meshgrid(rv, rv)
denom    = np.sqrt(R1**2 + R2**2)
xi_map   = np.abs(R1 + R2) / np.where(denom < 1e-9, 1e-9, denom)

ax1.pcolormesh(R1, R2, xi_map, cmap=CMAP, norm=norm,
               shading='gouraud', rasterized=True, alpha=0.2)

# Reference lines
xl = np.linspace(-lim, lim, 2)

# Line labels — positioned to avoid vectors
ax1.text( -0.5,  -0.85, r'$\xi=\sqrt{2}$', color=COL_AMPLIFY, fontsize=_FS_SM,
          ha='center', fontweight='bold', path_effects=stroke_black)
# ξ=1 label: upper-right corner, above r_C which sits on the x-axis
ax1.text( 0.85,  0.15, r'$\xi=1$',        color=COL_NULL,    fontsize=_FS_SM,
          ha='center', fontweight='bold', path_effects=stroke_black)
ax1.text( -0.5, 0.80, r'$\xi=0$',        color=COL_CANCEL,  fontsize=_FS_SM,
          ha='center', fontweight='bold', path_effects=stroke_black)



# Vectors
draw_vec(ax1, 1/np.sqrt(2), 1/np.sqrt(2), COL_ONES, r'$\boldsymbol{1}$', (0.00, 0.10))
# rA at (-1/√2, -1/√2): constructive diagonal, away from 1̂
draw_vec(ax1, -1/np.sqrt(2), -1/np.sqrt(2), COL_AMPLIFY, r'$\boldsymbol{r}_\mathrm{A}$', (0.00,  0.20))
# rB at (+1/√2, -1/√2): cancellation diagonal
draw_vec(ax1,  -1/np.sqrt(2), 1/np.sqrt(2), COL_CANCEL,  r'$\boldsymbol{r}_\mathrm{B}$', (0.00,  -0.26))
# rC on x-axis: null; label below to avoid ξ=1 text
draw_vec(ax1,  1.0,  0.00, COL_NULL, r'$\boldsymbol{r}_\mathrm{C}$', (-0.05, -0.20))

t_ann = np.linspace(0, np.pi / 4, 50)
r_ann = 0.25
ax1.plot(r_ann * np.cos(t_ann), r_ann * np.sin(t_ann), 'k-', lw=1.5, zorder=6)
ax1.text(r_ann * 0.7 * np.cos(np.pi / 8),
         r_ann * 0.6 * np.sin(np.pi / 8),
         r'$\theta$', fontsize=_FS, va='center', ha='center')

ax1.axvline(0.00, ymin=0.00, ymax=1.00, color = 'k', alpha = 0.3)
ax1.axhline(0.00, xmin=0.00, xmax=1.00, color = 'k', alpha = 0.3)
ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
ax1.set_xlabel(r'$r_1$', fontsize=_FS, labelpad=5)
ax1.set_ylabel(r'$r_2$', fontsize=_FS, labelpad=5)
ax1.set_aspect('equal')

# Panel label: outside, top-left
ax1.text(-0.22, 1.18, 'a', transform=ax1.transAxes,
         fontsize=_FS_PANEL, fontweight='bold', va='bottom', ha='left', clip_on=False)

# ══════════════════════════════════════════════════════════════════════════════
# Panel b — coloured arc: direction of r̂ relative to 1̂ determines ξ
#
# Coordinate system: x = ξ = √N cosθ,  y = √N sinθ
# where θ = ∠(1̂, r̂).  The arc sits at radius R = √N.
# ══════════════════════════════════════════════════════════════════════════════
theta_q = np.linspace(0, np.pi / 2, 600)
x_arc   = R * np.cos(theta_q)   # ξ
y_arc   = R * np.sin(theta_q)

# Coloured arc via LineCollection (colour = ξ = x-coordinate)
pts  = np.array([x_arc, y_arc]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc   = LineCollection(segs, cmap=CMAP, norm=norm,
                      linewidth=4, zorder=5, capstyle='round')
lc.set_array(x_arc[:-1])
ax2.add_collection(lc)

# No 1̂ arrow in panel b — axis label carries the meaning

# θ arc annotation (from x-axis at 0° to rC at 45°)
t_ann = np.linspace(0, np.pi / 4, 50)
r_ann = 0.25
ax2.plot(r_ann * np.cos(t_ann), r_ann * np.sin(t_ann), 'k-', lw=1.5, zorder=6)
ax2.text(r_ann * 0.7 * np.cos(np.pi / 8),
         r_ann * 0.6 * np.sin(np.pi / 8),
         r'$\theta$', fontsize=_FS, va='center', ha='center')

# Example vectors at rightful places: θ=0° (ξ=√2), θ=45° (ξ=1), θ=90° (ξ=0)
examples_circ = [
    (0,          COL_ONES, r'$\boldsymbol{1}$', ( -0.3,  0.10)),
    (0,          COL_AMPLIFY, r'$\boldsymbol{r}_\mathrm{A}$', ( -0.15,  0.10)),   # tip (√2, 0) — label above
    (np.pi / 2,  COL_CANCEL,  r'$\boldsymbol{r}_\mathrm{B}$', ( 0.07,  -0.16)),   # tip (0, √2) — label right
    (np.pi / 4,  COL_NULL,    r'$\boldsymbol{r}_\mathrm{C}$', ( -0.2,  -0.05)),   # tip (1, 1)
]
for angle, col, lbl, loff in examples_circ:
    draw_vec(ax2, R * np.cos(angle), R * np.sin(angle), col, lbl, loff)


ax2.axvline(1.0, ymin=0.08/(np.sqrt(2) * 1.18), ymax=1/(np.sqrt(2) * 1.18) + 0.08, color='k', ls=':', alpha=0.3, zorder=2)

# Ticks and limits
tick_v = [0, 1.0, R]
tick_l = ['0', '1', r'$\sqrt{2}$']
ax2.set_xticks(tick_v); ax2.set_xticklabels(tick_l)
ax2.set_yticks(tick_v); ax2.set_yticklabels(tick_l)
ax2.set_xlim(-0.05 * R, R * 1.1)
ax2.set_ylim(-0.05 * R, R * 1.1)
ax2.set_xlabel(r'$\xi = \sqrt{N}\cos\theta$', fontsize=_FS, labelpad=5)
ax2.set_ylabel(r'$\eta = \sqrt{N}\sin\theta$', fontsize=_FS, labelpad=5)
ax2.set_aspect('equal')

# Panel label: outside, top-left
ax2.text(-0.22, 1.18, 'b', transform=ax2.transAxes,
         fontsize=_FS_PANEL, fontweight='bold', va='bottom', ha='left', clip_on=False)

# ── Save ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(_REPO, "figures", "figure1_geometry.png"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
