# Interference Index for ML Formation Energy Models

This repository accompanies the manuscript

> **The Interference Index: Quantifying and Improving Error Cancellation in
> Machine-Learned Thermodynamic Stability Predictions**
> Alexander Kanzow, Cesare Boriosi, Carlos R. Jacinto-Mejía, Loriano Storchi
> and Giovanni Bistoni, *Journal* (2026), [DOI]

Machine-learned formation enthalpies approach DFT accuracy while remaining
unreliable for the stability predictions derived from them. The interference
index ξ measures how strongly per-compound errors reinforce or cancel when they
are combined into a reaction enthalpy. This repository provides the metric, two
training objectives that act on it, and everything needed to reproduce the
figures and tables of the paper.

It extends the benchmark study by
[Bartel et al. (2020)](https://www.nature.com/articles/s41524-020-00362-y),
whose seven models and 2020 Materials Project snapshot are included here for
comparison.

If you use this code, please cite

```
Kanzow, A., Boriosi, C., Jacinto-Mejia, C. R., Storchi, L., Bistoni, G.,
The Interference Index: Quantifying and Improving Error Cancellation in
Machine-Learned Thermodynamic Stability Predictions, Journal (2026)
```

---

## Repository structure

```
├── data/                        # Materials Project hull data
│   ├── 2020/                    # Bartel et al. snapshot, seven literature models at ml/
│   └── 2026/                    # MP snapshot from 2026/05/27
├── checkpoints/                 # Stage-one weights (from Zenodo)
├── results/                     # Interference scores and summaries
├── figures/                     # Paper and SI figures
├── models/
│   ├── iiLossNN.py              # iiLoss, penalty on xi^2
│   └── HdLossNN.py              # HdLoss, penalty on Hd^2
└── scripts/
    ├── train_models.py          # Train one model
    ├── sweep.sh                 # Full sweep over lambda and variant
    ├── process.py               # Data splits and model registry
    ├── interference_score.py    # Score predictions, writes results/
    ├── summarise_runs.py        # Table 2 of the main text
    ├── si_tables.py             # Supporting Information tables
    ├── reaction_coverage.py     # Fold coverage statistics
    ├── check_convergence.py     # Constant learning rate diagnostic
    ├── audit_runs.py            # Inventory of run directories
    ├── download_mp_current.py   # Re-fetch MP data, needs an API key
    ├── generate_figures.py      # Rebuild every figure
    └── figures/
        ├── fig1.py              # Interference circles, conceptual
        ├── fig2.py              # Model comparison, allMP 2020
        ├── fig3.py              # Gradient fields, MSE against xi squared
        ├── fig4.py              # Lambda sweep
        ├── fig5.py              # xi before and after training
        ├── figS1.py             # Convergence, iiLoss
        ├── figS2.py             # Convergence, HdLoss
        └── figS3.py             # Capacity dependence
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

Dependency versions are pinned. The pip section installs torch from the cu130
index, which matches CUDA 13.1. For older drivers change it to cu126, or delete
the line for a CPU-only install.

---

## Data

The repository carries the 2020 snapshot, both convex hulls, and the summary
CSVs behind every table. Files too large for git come from Zenodo,
<https://doi.org/10.5281/zenodo.20468539>.

| Needed for | Source |
|---|---|
| Table 1, figures 1, 2, 3 | in the repository |
| Table 2, all SI tables, figures 4 and S3 | in the repository |
| Figure 5 | `predictions_2026.tar.gz`, then rescore |
| Figures S1 and S2 | `training_curves_2026.tar.gz` |
| Retraining | `checkpoints.tar.gz` |

Unpack the archives at the repository root and every file lands where the
scripts expect it.

```bash
tar xzf predictions_2026.tar.gz
tar xzf training_curves_2026.tar.gz
tar xzf checkpoints.tar.gz
```

`generate_figures.py` checks its inputs before running and names the missing
file rather than failing partway.

---

## Reproducing the published results

**1. Score the predictions.** Writes `interference_scores_<year>.csv` and
`interference_summary_<year>.csv` into `results/<year>/`.

```bash
python scripts/interference_score.py 2020
python scripts/interference_score.py 2026
```

The summary files are tracked, so a clean `git status` afterwards confirms your
environment reproduces them exactly. The per-compound score files are
regenerated rather than stored.

**2. Rebuild the tables.**

```bash
python scripts/summarise_runs.py      # Table 2
python scripts/si_tables.py           # Tables S1 to S8
python scripts/reaction_coverage.py   # fold coverage percentages
```

**3. Rebuild the figures.**

```bash
python scripts/generate_figures.py            # all of them
python scripts/generate_figures.py --fig 4    # one
python scripts/generate_figures.py --si       # supporting figures only
```

---

## Retraining

Training is deterministic given the seed, so the published predictions are
reproducible from the stage-one checkpoints.

`sweep.sh` runs stage one, the matched lambda = 0 control, and lambda from 0.1
to 1.0 crossed with the four two-stage variants. That is the main sweep behind
Table 2 and Figure 4.

```bash
bash scripts/sweep.sh              # target Hf, seed 0
bash scripts/sweep.sh Hf 0 1 2     # three seeds, as in the paper
```

A single configuration, with optional seed and base width:

```bash
python scripts/train_models.py allMP_2026 Hf iiLoss_finetune_0.2 0
python scripts/train_models.py allMP_2026 Hf HdLoss_pcgrad_0.3 1 256
```

Run names follow `<objective>_<variant>_<lambda>_s<seed>`. Base width 1024 is
the default and carries no suffix.

### Runs outside the sweep

The Supporting Information also needs the capacity ladder, the denominator
stabiliser check and the conditional surgery comparison. `sweep.sh` does not
cover these, so without them the corresponding SI tables come out empty.

```bash
# Capacity ladder, Table S5 and Figure S3
for w in 32 64 128 256 512; do
  for s in 0 1 2; do
    python scripts/train_models.py allMP_2026 Hf iiLoss_save_0.0 $s $w
    for lam in 0.0 0.1 0.2 0.3 0.4 0.5; do
      python scripts/train_models.py allMP_2026 Hf iiLoss_finetune_$lam $s $w
    done
  done
done

# Denominator stabiliser, Table S3
for e in 1e-3 1e-6; do
  for s in 0 1 2; do
    python scripts/train_models.py allMP_2026 Hf iiLoss_finetune_eps${e}_0.2 $s
  done
done

# Conditional gradient surgery, Table S4
for lam in 0.1 0.2 0.3 0.4 0.5; do
  for s in 0 1 2; do
    python scripts/train_models.py allMP_2026 Hf iiLoss_pcgradc_$lam $s
  done
done

# Convergence diagnostic, Section 3 of the SI
python scripts/check_convergence.py --model iiLoss --lam 0.2
python scripts/check_convergence.py --model HdLoss --lam 0.1
```

---

## Scoring your own model

The interference index needs nothing but predictions and a reference, so any model can be
assessed without touching the training code.

1. Produce cross-validated, out-of-sample predictions as a JSON dictionary
   mapping formula to formation enthalpy in eV/atom, matching the keys in any
   existing `data/2026/ml/Hf/*/ml_input.json`.
2. Save it as `data/2026/ml/Hf/<your_model>_s0/ml_input.json`.
3. Run `python scripts/interference_score.py 2026`.

Your model then appears alongside the others in the summary CSV, with ξ, the
formation and decomposition enthalpy errors, and the stability classification
metrics.

---

## A note on the Materials Project

The Materials Project database changes over time. Results here are pinned to
two snapshots, the 2020 snapshot used by Bartel et al. and a 2026 snapshot retrieved on
27 May 2026, and both hulls are stored in the repository. Re-fetching current
data with `download_mp_current.py` will therefore not reproduce the published
numbers, which is the point of storing the snapshots.

---

## Licence

MIT, see `LICENSE.txt`.
