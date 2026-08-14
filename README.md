# Interference Index for ML Formation Energy Models

This repository accompanies the manuscript:

> **The Interference Index: Quantifying and Improving Error Cancellation in Machine-Learned Thermodynamic Stability Predictions**
> Alexander Kanzow, Cesare Boriosi, Carlos R. Jacinto-Mejía, Loriano Storchi and Giovanni Bistoni, *Journal* (2026)
> [DOI]

It extends the benchmark framework of [Bartel et al. (2020)](https://www.nature.com/articles/s41524-020-00362-y) by introducing the *interference index ξ*, a scale-invariant metric that quantifies how errors in machine-learned predictions propagate in derived quantities. It also features a model which minimises *ξ* through regularised training and compares the results to direct decomposition enthalpy training.

---

## Repository structure

```
├── data/                        # Materials Project hull data
│   ├── 2020/                    # Bartel et al. 2020 MP snapshot
│   └── 2026/                    # MP snapshot from 2026/05/27
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
        ├── fig1.py              # Fig. 1 — Interference circles (conceptual)
        ├── fig2.py              # Fig. 2 — Model comparison (allMP 2020)
        ├── fig3.py              # Fig. 3 — Gradient field panel (MSE vs ξ²)
        └── fig4.py              # Fig. 4 — Lambda sweep (iiLoss / HdLoss / PCGrad)
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
bash scripts/sweep.sh Hf iiLoss

# HdLoss + PCGrad sweep
bash scripts/sweep.sh Hf HdLoss pcgrad
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
python scripts/figures/fig4.py --csv results/2026/interference_summary_2026.csv
python scripts/figures/fig3.py
# etc.
```