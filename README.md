# Interference Index for ML Formation Energy Models

This repository accompanies the manuscript:

> **[Title]**
> [Authors], *Journal* (year)
> [DOI]

It extends the benchmark framework of [Bartel et al. (2020)](https://www.nature.com/articles/s41524-020-00362-y) by introducing the **interference index ξ** — a scale-invariant metric that quantifies how machine-learned formation energy errors propagate in decomposition reactions — and two training objectives that directly minimise it.

---

## Repository structure

```
├── data/                        # Materials Project hull data
│   ├── 2020/                    # Bartel et al. 2020 MP snapshot
│   └── 2026/                    # Current MP snapshot
│                                # 2026 ML predictions: https://doi.org/10.5281/zenodo.20468539
├── results/                     # Pre-computed interference scores
│   ├── 2020/                    # interference_scores_2020.csv, interference_summary_2020.csv
│   └── 2026/                    # interference_scores_2026.csv, interference_summary_2026.csv
├── figures/                     # Paper figures (PNG outputs)
├── models/                      # Model definitions
│   ├── iiLossNN.py              # iiLoss model (ξ²-regularised)
│   └── HdLossNN.py              # HdLoss model (Hd²-regularised)
└── scripts/                     # All runnable scripts
    ├── train_models.py          # Entry point for training
    ├── process.py               # Data splits and model registry
    ├── interference_score.py    # Compute ξ for any trained model
    ├── generate_figures.py      # Reproduce all paper figures
    ├── sweep.sh                 # iiLoss / HdLoss hyperparameter sweeps
    ├── perovskite_subset.py     # Subset analysis on ABO₃ perovskites
    ├── download_mp_current.py   # Re-fetch MP data (requires API key)
    └── figures/                 # Individual figure scripts
        ├── fig1_geometry.py
        ├── fig2_circles.py
        ├── fig3_model_comparison.py
        ├── fig5_gradient_fields.py
        └── fig6_lambda_sweep.py
```

---

## Installation

```bash
git clone https://github.com/KanzowA/InterferenceIndex
cd InterferenceIndex
conda env create -f environment.yml
conda activate interference
pip install -e .
```

---

## Reproducing results

**1. Train models** (GPU recommend, but also works on CPU; runs 5-fold CV on the 2026 MP dataset):
```bash
# iiLoss sweep
bash scripts/sweep.sh iiLoss

# HdLoss + PCGrad sweep
bash scripts/sweep.sh HdLoss pcgrad
```

**2. Compute interference scores:**
```bash
python scripts/interference_score.py 2026
python scripts/interference_score.py 2020
```

Results are written to `results/2026/` and `results/2020/` respectively.
Pre-computed results for all models in the paper are already provided there.

**3. Reproduce figures:**
```bash
python scripts/generate_figures.py        # all figures
python scripts/figures/fig6_lambda_sweep.py --csv results/2026/interference_summary_2026.csv
python scripts/figures/fig5_gradient_fields.py
# etc.
```

---

## The interference index

For a decomposition reaction $A = Σ_i \nu_i \Delta_\mathrm{f} H_i$, define the error vector **c** with components $c_i = \nu_i · \delta_i$ where $\delta_i = \Delta_\mathrm{f} H_i^\mathrm{ML} − \Delta_\mathrm{f} H_i^\mathrm{DFT}$. The interference index is:

\[\xi = \frac{\left|\sum_i c_i\right|}{\left|\left|\mathbf{c}\right|\right|} = \sqrt{N}\cdot \cos{\theta}\]

$\xi \rightarrow 0$: errors cancel across the reaction (favourable)
$\xi \rightarrow \sqrt{N}$: errors align constructively (unfavourable)

ξ is scale-in