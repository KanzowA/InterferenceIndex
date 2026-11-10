<!--
Copied unchanged from the working repository of the revision (ii-revision),
where it was committed in 8b9279f on 2026-11-09 13:33:43 +0100, before the
prospective runs started (13:54) and before any result came back (14:24). The
Deviations section was appended after the runs, in e37e367 (15:44:53).

Names in this document are those of the working repository:
  iiPerpFrozen  ->  iiLoss here      (the revision's objective)
  iiLoss        ->  iiLossRaw here   (the preprint's objective)
  paper 1       ->  the first preprint, chemRxiv 10.26434/chemrxiv.15008797/v1
-->

# Prospective WBM test: hypotheses fixed before any result (written 2026-11-09)

## Question
Does interference-aware training, as fixed in paper 1 and without retuning, improve
stability prediction for genuinely new materials outside the training database?
This answers Reviewer 2's "retrospective cross-validation only / no prospective
validation", and the manuscript's own stated limitation ("the present evaluation is
in-distribution").

## Protocol (fixed)
- Model: paper 1's ResidualMLP on element fractions (width 1024), paper 1's two-stage
  schedule (500 epochs λ = 0 → 200-epoch fine-tune, lr 1e-4), no hyperparameter changes.
- Training data: all Materials Project GGA/GGA+U (MP2020-corrected) compositions,
  dfH < 1 eV/atom, ground state per formula (Matbench Discovery `mp_energies`).
  Hull reactions are solved by LP over MP phases.
- Test data: Matbench Discovery WBM set (256,963 structures; ~215k unique prototypes),
  never seen in training. Composition model, so polymorphs share a prediction.
- Arms, λ fixed in advance from paper 1 (none tuned on WBM): matched control (λ = 0),
  iiLoss λ = 0.2 (published), HdLoss λ = 0.1 (published), iiPerpFrozen λ = 0.2.
- 5 seeds, paired against the control from the same stage-1 checkpoint.
- Metrics:
  - (a) Matbench Discovery's official protocol (screen against the DFT MP hull): F1,
    DAF, precision, recall, MAE of the hull distance, ROC-AUC, PR-AUC, on the full set
    and the unique-prototype subset.
  - (b) The closed WBM-only hull: MAE(ΔdH), xi_rms, Xi2_Q, F1.
  - (c) MAE(ΔfH) on WBM.

## Hypotheses (directional, iiPerpFrozen vs control unless stated)
- **H1** closed MAE(ΔdH) decreases (≥ 4/5 seeds); MAE(ΔfH) on WBM rises by no more than 2 %.
- **H2** official MBD F1 and DAF (unique prototypes) are not worse. The candidate's own
  error decides the mixed protocol, so with ΔfH accuracy preserved no loss is expected.
- **H3** published iiLoss λ = 0.2 *worsens* the official mixed-protocol MAE of the hull
  distance, because its ΔfH error roughly doubles and nothing cancels against a DFT hull.
  This is a prediction of the closed/mixed analysis.
- **H4** iiPerpFrozen beats HdLoss on closed MAE(ΔdH) and on MAE(ΔfH).

## Commitments
All arms and all metrics are reported whatever the outcome. No arm, λ or seed is added
or dropped after seeing results. Any deviation will be listed here with its reason.

## Deviations (added after the runs, 2026-11-09)
- **Data-loading bug.** pandas read the formula of sodium nitride ("NaN") as a missing
  value, which crashed the hull build. Fixed by reading with `na_filter=False`. No
  compound is selected or dropped differently than the protocol specifies.
- **Job-script fix.** The script now accepts the unpacked `.csv` and stops if the hull
  build fails. Runs from the first failed attempt produced no predictions.
- **Official metrics.** Rescored with the exact Matbench Discovery leaderboard
  convention (3-decimal rounding, unrounded prevalence for unique-prototype DAF). No
  conclusion changed.
- **Additional robustness analysis, not pre-registered and reported as such:**
  restriction to WBM compositions absent from the MP training set.
