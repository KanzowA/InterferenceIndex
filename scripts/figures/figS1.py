"""Figure S1: convergence of the two loss terms during stage two.

    (a) MSE term, relative to the stage-one reference
    (b) xi^2 term, relative to the stage-one reference

Curves are read from the per-run training_curve.csv files written by
train_models.py and averaged over folds and seeds at each epoch.

Usage
-----
    python scripts/figures/figS1.py
    python scripts/figures/figS1.py --lams 0.1 0.2 0.3 0.5
    python scripts/figures/figS1.py --prefix HdLoss_finetune_
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# -- Journal style (npj Computational Materials) --------------------------
# Fonts are scaled so they print at 7 pt (body) and 6 pt (minor) once the
# figure is reduced to a 170 mm double column.
_COL_W    = 6.69
_FIG_W    = 12.0
_FS       = round(7.0 * _FIG_W / _COL_W)
_FS_SM    = round(6.0 * _FIG_W / _COL_W)
_FS_LEG   = _FS
_FS_PANEL = 20
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   _FS,
    "axes.linewidth": 0.8,
})

# -- Panels ---------------------------------------------------------------
METRICS = [
    ("mse_over_ref",
     r"$\mathrm{MSE}(\Delta_\mathrm{f}H)\,/\,\mathrm{MSE}_0(\Delta_\mathrm{f}H)$"),
    ("penalty_over_ref",
     r"$\langle\xi^2\rangle\,/\,\langle\xi^2\rangle_0$"),
]

DEFAULT_LAMS = ["0.1", "0.2", "0.3", "0.4", "0.5",
                "0.6", "0.7", "0.8", "0.9", "1.0"]
DEFAULT_BASE_WIDTH = 1024

# Interpolates between the two objective colours of Fig. 4 through purple, so
# that no value of lam falls on a washed-out midpoint.
_CMAP = LinearSegmentedColormap.from_list(
    "lam", ["#1E46E4", "#7A1FA2", "#E61D60"])


# -- Helpers --------------------------------------------------------------
def _curve_paths(year, target, prefix, lam, width):
    """All per-seed training_curve.csv files for one lam and base width."""
    suffix = "" if width == DEFAULT_BASE_WIDTH else f"_w{width}"
    pattern = os.path.join(REPO_ROOT, "data", year, "ml", target,
                           f"{prefix}{lam}{suffix}_s*", "training_curve.csv")
    return sorted(glob.glob(pattern))


def load_curve(year, target, prefix, lam, width):
    """Mean over folds and seeds at each epoch, or None if nothing is found."""
    paths = _curve_paths(year, target, prefix, lam, width)
    if not paths:
        return None
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df.groupby("epoch").mean(numeric_only=True).reset_index()


# -- Figure ---------------------------------------------------------------
def build_figure(curves):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    values = [float(lam) for lam, _ in curves]
    norm = Normalize(vmin=min(values), vmax=max(values))

    for ax_idx, (ax, (col, ylabel)) in enumerate(zip(axes, METRICS)):
        for lam, df in curves:
            ax.plot(df["epoch"], df[col], lw=1.6,
                    color=_CMAP(norm(float(lam))))

        ax.set_xlabel("epoch", fontsize=_FS)
        ax.set_ylabel(ylabel, fontsize=_FS)
        ax.tick_params(labelsize=_FS)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, which="major", lw=0.4, alpha=0.5)

        ax.text(-0.18, 1.04, "ab"[ax_idx], transform=ax.transAxes,
                fontsize=_FS_PANEL, fontweight="bold", va="bottom", ha="left",
                clip_on=False)

    # A colourbar rather than a legend, which would need one entry per lam.
    scalar_map = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
    scalar_map.set_array([])
    cbar = fig.colorbar(scalar_map, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label(r"$\lambda$", fontsize=_FS)
    cbar.set_ticks(values)
    cbar.ax.tick_params(labelsize=_FS_SM)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2026")
    parser.add_argument("--target", default="Hf")
    parser.add_argument("--prefix", default="iiLoss_finetune_")
    parser.add_argument("--lams", nargs="+", default=DEFAULT_LAMS)
    parser.add_argument("--width", type=int, default=DEFAULT_BASE_WIDTH,
                        help="base width of the runs to plot")
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "figures", "figureS1.png"))
    args = parser.parse_args()

    curves = []
    for lam in args.lams:
        df = load_curve(args.year, args.target, args.prefix, lam, args.width)
        if df is None:
            print(f"  no training_curve.csv for {args.prefix}{lam} "
                  f"at width {args.width}, skipping")
            continue
        curves.append((lam, df))
    if not curves:
        raise SystemExit("No training curves found")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build_figure(curves)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()
