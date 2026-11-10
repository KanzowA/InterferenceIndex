"""Train on all of MP and predict the WBM discovery set (prospective test).

Usage
-----
    python scripts/prospective.py <model> [seed] [base_width]
    python scripts/prospective.py iiLoss_save_0.0 0          # stage one
    python scripts/prospective.py iiLoss_finetune_0.2 0      # stage two

The architecture and two-stage schedule of the cross-validation runs, but one
model per seed trained on every compound of data/2025/hullout_2025.json, the MP
GGA/GGA+U hull, whose energies follow the convention of the WBM test set. Stage
one writes checkpoints/prospective/s<seed>/, which every stage-two model of that
seed reloads.

A composition model gives every polymorph of a formula the same prediction, so
each WBM entry gets the prediction for its reduced formula. Written in the
format of Matbench Discovery,

    data/wbm/ml/<model>[_w<width>]_s<seed>.csv.gz    material_id, e_form_per_atom

The hypotheses, the runs and the metrics were registered before any of these
runs started; see the revision's PREREGISTRATION.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from process import DEFAULT_BASE_WIDTH, MODEL_DICTIONARY, hidden_from_base

HULLOUT = os.path.join(REPO_ROOT, "data", "2025", "hullout_2025.json")
WBM_SUMMARY = os.path.join(REPO_ROOT, "data", "wbm", "2023-12-13-wbm-summary.csv.gz")
OUT_DIR = os.path.join(REPO_ROOT, "data", "wbm", "ml")
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints", "prospective")


def main(argv):
    try:
        model_name = argv[1]
    except IndexError:
        print("Arguments should be Model [Seed] [BaseWidth]")
        exit(1)
    seed = int(argv[2]) if len(argv) > 2 else 0
    base_width = int(argv[3]) if len(argv) > 3 else DEFAULT_BASE_WIDTH
    hidden = hidden_from_base(base_width)

    width_tag = "" if base_width == DEFAULT_BASE_WIDTH else f"_w{base_width}"
    out_file = os.path.join(OUT_DIR, f"{model_name}{width_tag}_s{seed}.csv.gz")
    if os.path.exists(out_file):
        print(f"{out_file} exists, skipping")
        return

    from matminer.featurizers.composition import ElementFraction
    from pymatgen.core import Composition

    ckpt = os.path.join(CKPT_DIR + width_tag, f"s{seed}")
    model = MODEL_DICTIONARY[model_name]("Hf", seed, hidden)
    if model_name.startswith("iiLoss_save_"):
        model.checkpoint_dir = ckpt
    else:
        need = os.path.join(ckpt, f"Hf_fold0_w{hidden[0]}_seed{seed}.pt")
        if not os.path.exists(need):
            raise SystemExit(f"stage-one checkpoint missing: {need}")
        model.finetune_from = ckpt
    model.history_path = os.path.join(OUT_DIR, "curves",
                                      f"{model_name}{width_tag}_s{seed}.csv")

    print(f"Training {model_name} on all of MP (seed {seed}, hidden {hidden})")
    t0 = time.time()
    with open(HULLOUT) as f:
        features, targets, labels = model.preprocess(json.load(f))
    model.fit(features, targets)

    wbm = pd.read_csv(WBM_SUMMARY)
    reduced = wbm.formula.map(lambda f: Composition(f).reduced_formula)
    formulas = sorted(set(reduced))
    featurizer = ElementFraction()
    X = np.array([featurizer.featurize(Composition(f)) for f in formulas], dtype=np.float32)
    X = np.hstack([np.zeros((len(formulas), 1), np.float32), X])   # row-index column
    pred = dict(zip(formulas, model.predict(X)))

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({"material_id": wbm.material_id,
                  "e_form_per_atom": reduced.map(pred)}).to_csv(out_file, index=False)
    print(f"Trained on {len(labels)} MP compounds, predicted {len(formulas)} WBM "
          f"formulas in {time.time() - t0:.0f} s -> {out_file}")


if __name__ == "__main__":
    main(sys.argv)
