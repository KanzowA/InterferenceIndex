<!--
Copied unchanged from the working repository of the revision (ii-revision),
committed in bcc0880 on 2026-11-09 15:52:27 +0100, before the nested job was
started. scripts/nested_cv.py implements it; it is the working repository's
nested/run_nested.py under this repository's names and checked against it
prediction by prediction.

Names in this document are those of the working repository:
  iiPerpFrozen  ->  iiLoss here      (the revision's objective)
  iiLoss        ->  iiLossRaw here   (the preprint's objective)
  data/mp_hull_gga.json  ->  data/2025/hullout_2025.json
  score_paper1.py        ->  scripts/spectral.py
  paper 1       ->  the first preprint, chemRxiv 10.26434/chemrxiv.15008797/v1
-->

# Nested in-distribution comparison: protocol fixed before running (2026-11-09)

Purpose: replace the in-distribution numbers of the revision with values that answer
R1.4 (no λ chosen on reported metrics), R1.3 (ML-hull metrics) and the dataset
inconsistency found in AUDIT.md §D. Everything is shared with the prospective test.

- **Data:** `data/mp_hull_gga.json`, the same file the prospective test trained on (MP
  GGA/GGA+U, MP2020-corrected, ΔfH < 1 eV/atom, ground state per formula, 103,580
  compounds, reactions by LP, |S·ΔfH + ΔdH| < 1e-10). The job prints its content hash.
  Expected `9ee82af58674dcc57bdd28fc035caaa8`; a different hash from LP-degenerate
  reaction strings is harmless, as long as the prospective test used the same file.
- **Model and training:** paper 1 unchanged (ResidualMLP width 1024, stage 1 500 epochs
  at λ = 0, 200-epoch fine-tune at lr 1e-4, Adam, weight decay 1e-4, cosine, clip 1.0).
  Model code `paper1_regime/models` is the same as in the prospective test.
- **Outer CV:** KFold(5, shuffle=True, random_state=10), as in paper 1. Seeds 0–4.
- **Inner selection:** 10 % of each outer training fold is held out as validation
  (rng seeded by (seed, fold)). Stage 1 is trained on the rest, and each penalised arm
  is fine-tuned at λ ∈ {0.05, 0.1, 0.2, 0.3, 0.5}.
  - **Criterion:** validation MAE(ΔdH) over reactions whose reactant is a validation
    compound, every participant predicted by the same inner model.
  - λ* = argmin; ties go to the smaller λ. The F1-based choice is logged for reference
    only.
- **Outer refit:** stage 1 on the full outer training fold, then fine-tunes:
  - the control (λ = 0)
  - each arm at its λ*
  - each arm at the fixed published value (iiLoss 0.2, HdLoss 0.1, iiPerpFrozen 0.2)
  - iiPerpFrozen at λ = 0.2 with ε ∈ {1e-5, 1e-3} (ε sensitivity; default 1e-4)

  Predictions are made for the untouched outer test fold only.
- **Scoring (paper-1 conventions, `score_paper1.py`):**
  - MAE(ΔfH); MAE(ΔdH) over all reactions; ξ_rms over N ≥ 2; Ξ²_Q and p_gauge
  - F1 at 0, max-F1, ROC-AUC, PR-AUC, MCC
  - ML-hull-reconstructed F1 (R1.3)
  - all paired against the control, per seed
- **Primary comparison:** the nested arms vs the control. The fixed-λ arms show that the
  prospective λ choice is representative.
- **Reporting:** all arms and metrics are reported regardless of outcome.
