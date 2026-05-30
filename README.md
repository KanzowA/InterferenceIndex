# Interference Index for ML Formation Energy Models

This repository accompanies the manuscript:

> **[Title]**
> [Authors], *Journal* (year)
> [DOI]

It extends the benchmark framework of [Bartel et al. (2020)](https://www.nature.com/articles/s41524-020-00362-y) by introducing the **interference index ξ** — a scale-invariant metric that quantifies how machine-learned formation energy errors propagate in decomposition reactions — and two training objectives that directly minimise it.

---

## Repository structure

```
├── data/                        # Materials Project dataset (2026 snapshot)
├── results/                     # Pre-computed interference scores
├── figures/                     # Paper figures (PDF/PNG)
├── scripts/                     # Figure generation scripts
├── sweeps/                      # Cluster training sweep scripts
├── interference_score.py        # Compute ξ for any trained model
├── perovskite_subset.py         # Subset analysis on ABO₃ perovskites
├── download_mp_current.py       # Re-fetch MP data (requires API key)
└── mlstabilitytest/             # Core package
    ├── train_models.py          # Entry point for training
    └── training/
        ├── iiLossNN.py          # iiLoss model (ξ²-regularised)
        ├── EdLossNN.py          # EdLoss model (Ed²-regularised)
        ├── CHGNetModel.py       # CHGNet wrapper (pre-trained)
        └── process.py          # Data splits and model registry
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
bash sweeps/sweep_iiLoss.sh

# EdLoss + PCGrad sweep
bash sweeps/sweep_EdLoss_pcgrad.sh
```

**2. Compute interference scores:**
```bash
python scripts/interference_score.py allMP_2026
python scripts/interference_score.py allMP_2020
```

Results are written to `results/2026/` and `results/2020/` respectively.
Pre-computed results for all models in the paper are already provided there.

**3. Reproduce figures:**
```bash
python figures/generate_figures.py        # all figures
python scripts/fig6_lambda_sweep.py --csv results/2026/interference_summary_2026.csv
python scripts/fig5_gradient_fields.py
# etc.
```

---

## The interference index

For a decomposition reaction X → Σ_k v_k P_k, define the error vector **c** with components c_i = v_i · δ_i where δ_i = ΔH_f^ML - ΔH_f^DFT. The interference index is:

ξ = |Σ c_i| / ‖**c**‖ = √N · |cos θ|

ξ → 0: errors cancel across the reaction (favourable)
ξ → √N: errors align constructively (unfavourable)

ξ is scale-invariant and invariant to systematic biases (since reaction weights sum to zero), making it a more informative diagnostic than decomposition energy MAE alone.

---

## Citation

If you use this repository, please cite both this work and Bartel et al.:

```bibtex
@article{[key],
  title   = {[Title]},
  author  = {[Authors]},
  journal = {npj Computational Materials},
  year    = {2025},
  doi     = {[DOI]}
}

@article{bartel2020,
  title   = {A critical examination of compound stability predictions from machine-learned formation energies},
  author  = {Bartel, C. and Trewartha, A. and Wang, Q. and Dunn, A. and Jain, A. and Ceder, G.},
  journal = {npj Computational Materials},
  volume  = {6},
  pages   = {97},
  year    = {2020}
}
```
