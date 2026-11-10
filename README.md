# Interference Index for ML Formation Energy Models

This repository accompanies the manuscript

> **The Interference Index: Quantifying and Improving Error Cancellation in
> Machine-Learned Thermodynamic Stability Predictions**
> Alexander Kanzow, Cesare Boriosi, Carlos R. Jacinto-Mejía, Loriano Storchi
> and Giovanni Bistoni, *ChemRxiv* (2026), DOI: 10.26434/chemrxiv.15008797/v1

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
A. Kanzow, C. Boriosi, C. R. Jacinto-Mejía, L. Storchi, G. Bistoni, The Interference Index: Quantifying and Improving Error Cancellation in Machine-Learned Thermodynamic Stability Predictions, ChemRxiv, 2026.
```

---

## The revision

This branch holds the revised manuscript. What changed, and why:

- **The objective.** The ξ² penalty of the preprint can be lowered without
  cancelling anything, by shifting every formation enthalpy along a linear
  function of composition (the elemental-reference gauge) or by growing the
  per-reaction denominators. `iiLoss` now projects the gauge out of the penalty
  and freezes the denominators at the reference model's own spreads, so it is
  lowered only by smaller decomposition errors. The preprint's objective is the
  same class with both switches off and is registered as `iiLossRaw_*`;
  `scripts/check_equivalence.py` shows it is the preprint's model to the last bit.
- **The data.** The cross-validation now runs on the MP GGA/GGA+U hull of
  Matbench Discovery (`data/2025`), whose reactions reproduce its own stored
  decomposition enthalpies exactly and whose energies follow the convention of
  the WBM test set. The 2026 snapshot stays in the repository for the preprint's
  analyses.
- **λ is chosen without the test set.** `scripts/nested.sh` selects λ on a
  validation split inside every training fold (`docs/NESTED_PROTOCOL.md`).
- **The hull rebuilt from predictions** (Bartel et al.) is scored next to the
  reaction-based metrics, as are max-F1, ROC-AUC, PR-AUC, MCC and the
  gauge-fixed index Ξ²_Q.
- **A prospective test.** One model per seed is trained on all of MP and
  predicts the 257k materials of the WBM discovery set, under hypotheses
  registered before the runs (`docs/PREREGISTRATION.md`).
- **The Matbench Discovery models.** The diagnostic is applied to the published
  predictions of 67 models, 57 of them interatomic potentials, on a closed WBM
  hull.
- **The capacity ladder** is repeated with the nested protocol.

Every number of the revised manuscript is listed, next to its value in the
preprint, in `results/paper_numbers.md`, written by `scripts/paper_numbers.py`.

---

## Repository structure

```
├── data/                        # Materials Project hull data
│   ├── 2020/                    # Bartel et al. snapshot, seven literature models at ml/
│   ├── 2025/                    # MP GGA/GGA+U hull of Matbench Discovery, the revision's data
│   ├── 2026/                    # MP snapshot from 2026/05/27, the preprint's data
│   └── wbm/                     # WBM discovery set: closed hull, model manifest, predictions
├── checkpoints/                 # Stage-one weights (from Zenodo)
├── docs/                        # Nested protocol and preregistration of the prospective test
├── results/                     # Scores, summaries, paper_numbers.md
├── figures/                     # Paper and SI figures
├── models/
│   ├── iiLossNN.py              # iiLoss, gauge-fixed penalty on xi^2 (iiLossRaw: the preprint's)
│   └── HdLossNN.py              # HdLoss, penalty on Hd^2
└── scripts/
    ├── train_models.py          # Train one model
    ├── process.py               # Data splits and model registry
    ├── build_hull.py            # hullout files of the revision from published energies
    ├── nested_cv.py             # One seed and fold of the nested protocol; merge
    ├── nested.sh                # Nested cross-validation, Table 2
    ├── capacity.sh              # Capacity ladder with the nested protocol
    ├── prospective.py           # Train on all of MP, predict WBM
    ├── prospective.sh           # The preregistered prospective runs
    ├── sweep.sh                 # Full sweep over lambda and variant
    ├── interference_score.py    # Score predictions, writes results/
    ├── spectral.py              # Shared scoring machinery (structure matrix, gauge, LP hull)
    ├── spectral_score.py        # Revision metrics: Xi2_Q, threshold-free, ML hull
    ├── prospective_score.py     # Matbench Discovery protocol and closed WBM hull
    ├── mlip_score.py            # Diagnostic for the Matbench Discovery models
    ├── summarise_revision.py    # Paired comparisons behind the revision's tables
    ├── paper_numbers.py         # Every number of the manuscript, against the preprint
    ├── check_equivalence.py     # iiLossRaw is the preprint's model, to the bit
    ├── import_revision_runs.py  # Provenance: names of the cluster runs
    ├── summarise_runs.py        # Table 2 of the preprint
    ├── si_tables.py             # Supporting Information tables of the preprint
    ├── reaction_coverage.py     # Fold coverage statistics
    ├── check_convergence.py     # Constant learning rate diagnostic
    ├── audit_runs.py            # Inventory of run directories
    ├── download_mp_current.py   # Re-fetch MP data, needs an API key
    ├── generate_figures.py      # Rebuild every figure
    └── figures/
        ├── fig1.py              # Interference circles, conceptual
        ├── fig2.py              # Model comparison, allMP 2020
        ├── fig3.py              # Gradient fields, MSE against xi squared
        ├── fig4.py              # Lambda on the validation split; S5 adds iiLossRaw
        ├── fig5.py              # xi, dH_d errors and their xi / |c| split; S6 for iiLossRaw
        ├── figS1.py             # Convergence, iiLoss (preprint)
        ├── figS2.py             # Convergence, HdLoss (preprint)
        ├── figS3.py             # Capacity dependence
        └── figS4.py             # Matbench Discovery models on a closed hull
```

---

## Installation

```bash
git clone https://github.com/KanzowA/InterferenceIndex
cd InterferenceIndex
git checkout revision
conda env create -f environment.yml
conda activate interference
pip install -e .
```

Dependency versions are pinned. The pip section installs torch from the cu130
index, which matches CUDA 13.1. For older drivers change it to cu126, or delete
the line for a CPU-only install.

---

## Data

The repository carries the 2020 and 2026 snapshots, the 2025 hull every
revision run was trained on, the closed WBM hull, the selection logs of the
nested runs, and the summary CSVs behind every table. Files too large for git
come from Zenodo, <https://doi.org/10.5281/zenodo.20468539>, or from their
original sources.

| Needed for | Source |
|---|---|
| Table 1, figures 1, 2, 3 | in the repository |
| Figure 4 | in the repository (`results/2025/nested/`) |
| Table 2, figures 5 and S3 | `predictions_2025.tar.gz`, then rescore |
| Prospective test | `predictions_wbm.tar.gz` and the WBM summary, see below |
| Figure S4 | the Matbench Discovery predictions, see below |
| Preprint: figures S1 and S2 | `training_curves_2026.tar.gz` |
| Retraining | `checkpoints.tar.gz` |

Unpack the archives at the repository root and every file lands where the
scripts expect it.

```bash
tar xzf predictions_2025.tar.gz
tar xzf predictions_wbm.tar.gz
```

The WBM summary comes from Matbench Discovery, and `prospective.sh` fetches it
if it is absent:

```bash
curl -L -o data/wbm/2023-12-13-wbm-summary.csv.gz \
  https://github.com/janosh/matbench-discovery/raw/main/data/wbm/2023-12-13-wbm-summary.csv.gz
```

The published predictions of the Matbench Discovery models are listed with
their figshare links and MD5 sums in `data/wbm/mbd_manifest.csv`; save each as
`data/wbm/mbd/<model_key>.csv.gz`. Six models of the current leaderboard have
no published predictions and are absent.

The two hullout files of the revision are rebuilt from published energies by

```bash
python scripts/build_hull.py mp  data/2025/2025-02-01-mp-energies.csv.gz
python scripts/build_hull.py wbm data/wbm/2023-12-13-wbm-summary.csv.gz
```

The first prints a content hash, which must match the one of the file the
runs were trained on (`9ee82af58674dcc57bdd28fc035caaa8`); the job scripts
check it before training.

---

## Reproducing the published results

**1. Score the predictions.** Writes the summaries into `results/<year>/` and
`results/wbm/`.

```bash
python scripts/interference_score.py 2020
python scripts/interference_score.py 2025 --summary-only
python scripts/spectral_score.py 2020
python scripts/spectral_score.py 2025        # about an hour: the ML hull is an LP per compound
python scripts/prospective_score.py
python scripts/mlip_score.py
```

The summary files are tracked, so a clean `git status` afterwards confirms your
environment reproduces them exactly. The per-compound score files are
regenerated rather than stored; for 2025, `--summary-only` skips them, since
they would take several GB.

**2. Rebuild the tables and the number list.**

```bash
python scripts/summarise_revision.py  # Table 2, ML hull, capacity, prospective
python scripts/reaction_coverage.py --dataset 2025
python scripts/paper_numbers.py       # results/paper_numbers.md
```

**3. Rebuild the figures.**

```bash
python scripts/generate_figures.py            # all of them
python scripts/generate_figures.py --fig 4    # one
python scripts/generate_figures.py --si       # supporting figures only
```

**4. Check the objective.** Trains small models on CPU and compares
`iiLossRaw` with the preprint's class from `main`; every difference must be 0.

```bash
python scripts/check_equivalence.py
```

---

## Retraining

Training is deterministic given the seed, up to the order of floating-point
sums on the GPU, so retrained predictions agree with the published ones to
float32 rounding. Each job checks the hull, logs to `logs/`, skips what has
finished and stops at the first failure.

```bash
bash scripts/nested.sh           # Table 2: seeds 0 to 4, 5 outer folds each
bash scripts/capacity.sh         # capacity ladder: widths 32 to 512, seeds 0 1 2
bash scripts/prospective.sh      # prospective test: seeds 0 to 4
```

`nested.sh` trains, per seed and fold, the inner selection runs and the final
refits; one fold is

```bash
python scripts/nested_cv.py run 0 3              # seed 0, fold 3
python scripts/nested_cv.py run 0 3 256 --objectives iiLoss,HdLoss --no-eps
python scripts/nested_cv.py merge                # join the folds
```

A single configuration on the 2025 hull, with optional seed and base width:

```bash
python scripts/train_models.py allMP_2025 Hf iiLoss_finetune_0.2 0
python scripts/train_models.py allMP_2025 Hf iiLossRaw_finetune_0.2 1 256
```

Run names follow `<objective>_<variant>_<lambda>_s<seed>`, with `_w<width>`
for base widths other than 1024. A nested run, whose λ is chosen per fold, is
named `<objective>_nested_s<seed>`. `sweep.sh` trains the full λ sweep; its λ
values are scored on the test folds, so they are for looking at the response,
not for choosing an operating point.

The nested, capacity and prospective runs of the paper were trained on the
cluster from the working repository of the revision, whose models are those
of this branch; `import_revision_runs.py` records how their names map onto the
names here.

### The preprint's runs

The predictions in `data/2026/ml/` predate the revision: the `iiLoss_*` runs
stored there were trained with the preprint's objective, `iiLossRaw` in this
branch's registry. `summarise_runs.py`, `si_tables.py`, `check_convergence.py`
and figures S1 and S2 still read them and reproduce the preprint.

---

## Scoring your own model

The interference index needs nothing but predictions and a reference, so any model can be
assessed without touching the training code.

1. Produce cross-validated, out-of-sample predictions as a JSON dictionary
   mapping formula to formation enthalpy in eV/atom, matching the keys of
   `data/2025/hullout_2025.json`.
2. Save it as `data/2025/ml/Hf/<your_model>_s0/ml_input.json`.
3. Run `python scripts/spectral_score.py 2025 --mlhull <your_model>`.

Your model then appears alongside the others in the summary CSV, with ξ, the
gauge-fixed index, the formation and decomposition enthalpy errors, the
stability classification metrics and those of the hull rebuilt from its
predictions. Predictions for the WBM set in the format of Matbench Discovery
are scored by `prospective_score.py <file.csv.gz>`.

---

## A note on the Materials Project

The Materials Project database changes over time. Results here are pinned to
three hulls: the 2020 snapshot used by Bartel et al., a 2026 snapshot retrieved
on 27 May 2026, and the GGA/GGA+U energies distributed with Matbench Discovery
(2025-02-01). All three are stored in the repository. Re-fetching current data
with `download_mp_current.py` will therefore not reproduce the published
numbers, which is the point of storing the snapshots.

---

## Licence

MIT, see `LICENSE.txt`.
