"""Nested cross-validation: lam is chosen on a validation split, never on test.

Usage
-----
    python scripts/nested_cv.py run <seed> <fold> [base_width]
    python scripts/nested_cv.py run 0 3 256 --objectives iiLoss,HdLoss --no-eps
    python scripts/nested_cv.py merge [base_width]

One `run` call is one (seed, outer fold) of the protocol below; `merge` joins the
five outer folds of every seed into one out-of-fold prediction set per run, in
the layout train_models.py writes, and checks that every compound is predicted
exactly once.

Protocol (fixed before the runs, see the revision's NESTED_PROTOCOL)
    outer   KFold(5, shuffle=True, random_state=10), the folds of allMP_2025
    inner   10 % of the outer training fold held out for validation, drawn
            with an RNG seeded by (seed, fold)
    select  on the inner training part: stage one (iiLoss_save_0.0), then each
            objective fine-tuned at every lam in GRID. The criterion is the
            validation MAE(dH_d), over the reactions whose reactant is a
            validation compound, every participant predicted by the same inner
            model. Ties go to the smaller lam.
    refit   stage one again on the full outer training fold, then
                iiLoss_finetune_0.0          the matched control
                <objective>_nested           at the selected lam
                <objective>_finetune_<lam>   at the preprint's fixed lam
                iiLoss_finetune_eps<e>_0.2   denominator stabiliser, e = 1e-5, 1e-3
            each predicting the untouched outer test fold.

Output
    data/2025/ml/Hf/nested[_w<width>]/s<seed>/f<fold>/   one JSON per run, the
                                                         selection log, DONE
    data/2025/ml/Hf/<run>[_w<width>]_s<seed>/ml_input.json         after merge
    results/2025/nested/selection[_w<width>]_s<seed>_f<fold>.json  after merge
    checkpoints/nested[_w<width>]/s<seed>/f<fold>/{inner,outer}/

A fold that finished is skipped, so an interrupted job resumes where it stopped.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from process import DEFAULT_BASE_WIDTH, MODEL_DICTIONARY, hidden_from_base
from spectral import load_hull, reaction_sums

HULLOUT = os.path.join(REPO_ROOT, "data", "2025", "hullout_2025.json")
ML_DIR = os.path.join(REPO_ROOT, "data", "2025", "ml", "Hf")
LOG_DIR = os.path.join(REPO_ROOT, "results", "2025", "nested")
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints", "nested")

GRID = [0.05, 0.1, 0.2, 0.3, 0.5]
# The preprint's operating points, kept as a fixed-lam reference for each objective.
FIXED = {"iiLoss": 0.2, "iiLossRaw": 0.2, "HdLoss": 0.1}
OBJECTIVES = ["iiLoss", "iiLossRaw", "HdLoss"]
EPS_CHECK = [1e-5, 1e-3]
VAL_FRAC = 0.1


def width_tag(width):
    return "" if width == DEFAULT_BASE_WIDTH else f"_w{width}"


# -- One (seed, fold) -----------------------------------------------------

def run_fold(seed, fold, width, objectives, eps_check):
    tag = width_tag(width)
    hidden = hidden_from_base(width)
    fdir = os.path.join(ML_DIR, f"nested{tag}", f"s{seed}", f"f{fold}")
    if os.path.exists(os.path.join(fdir, "DONE")):
        print(f"s{seed} f{fold}: done, skipping")
        return
    os.makedirs(fdir, exist_ok=True)
    t0 = time.time()

    with open(HULLOUT) as f:
        data = json.load(f)
    H = load_hull(HULLOUT)

    # Featurised once; identical to what preprocess() gives every model below.
    probe = MODEL_DICTIONARY["iiLoss_save_0.0"]("Hf", seed, hidden)
    features, targets, labels = probe.preprocess(data)

    def train_predict(name, ckpt, train_idx):
        """Fit one registry model on train_idx; predictions for every compound."""
        model = MODEL_DICTIONARY[name]("Hf", seed, hidden)
        if name.startswith("iiLoss_save_"):
            model.checkpoint_dir = ckpt
        else:
            # HdLossNN only warns on a missing checkpoint and trains from
            # scratch; refuse instead.
            need = os.path.join(ckpt, f"Hf_fold{fold}_w{hidden[0]}_seed{seed}.pt")
            if not os.path.exists(need):
                raise FileNotFoundError(f"stage-one checkpoint missing: {need}")
            model.finetune_from = ckpt
        model._data, model._labels = data, labels
        model._fold_counter = fold          # checkpoint names carry the outer fold
        model.fit(features[train_idx], targets[train_idx])
        return np.asarray(model.predict(features), dtype=float)

    def validation(pred, reactants):
        """MAE(dH_d) and F1 over the reactions of the validation compounds."""
        d = np.zeros(H["M"])
        for lbl, p in zip(labels, pred):
            i = H["idx"][lbl]
            d[i] = p - H["hf"][i]
        y, _ = reaction_sums(H, d)
        m = np.isin(H["rc"], reactants)
        truth = H["hd"][m] <= 0
        pos = (H["hd"][m] - y[m]) <= 0
        tp = (pos & truth).sum()
        fp = (pos & ~truth).sum()
        fn = (~pos & truth).sum()
        return (float(np.abs(y[m]).mean()), float(2 * tp / max(2 * tp + fp + fn, 1)),
                int(m.sum()))

    # -- Splits, identical for every objective ----------------------------
    outer = list(KFold(n_splits=5, shuffle=True, random_state=10).split(features))
    train, test = outer[fold]
    rng = np.random.default_rng([seed, fold, 2026])
    val = np.sort(rng.choice(train, int(round(VAL_FRAC * len(train))), replace=False))
    inner = np.setdiff1d(train, val)
    val_reactants = np.array([H["idx"][labels[i]] for i in val if labels[i] in H["idx"]])
    log = dict(seed=seed, fold=fold, width=width, objectives=objectives,
               n_train=len(train), n_test=len(test), n_inner_train=len(inner),
               n_val=len(val), grid=GRID, criterion="validation MAE(dH_d)",
               val={}, selected={}, selected_by_F1={})

    # -- Inner loop: choose lam on the validation split -------------------
    ck_in = os.path.join(CKPT_DIR + width_tag(width), f"s{seed}", f"f{fold}", "inner")
    train_predict("iiLoss_save_0.0", ck_in, inner)
    log["val"]["control"] = validation(
        train_predict("iiLoss_finetune_0.0", ck_in, inner), val_reactants)
    for obj in objectives:
        for lam in GRID:
            log["val"][f"{obj}_{lam:g}"] = validation(
                train_predict(f"{obj}_finetune_{lam}", ck_in, inner), val_reactants)
        scores = [(log["val"][f"{obj}_{lam:g}"][0], lam) for lam in GRID]
        log["selected"][obj] = min(scores)[1]
        log["selected_by_F1"][obj] = -max((log["val"][f"{obj}_{lam:g}"][1], -lam)
                                          for lam in GRID)[1]
    print(f"s{seed} f{fold}: selected {log['selected']} ({time.time() - t0:.0f} s)",
          flush=True)

    # -- Outer: refit on the full training fold, predict the test fold ----
    ck_out = os.path.join(CKPT_DIR + width_tag(width), f"s{seed}", f"f{fold}", "outer")
    train_predict("iiLoss_save_0.0", ck_out, train)
    final = [("iiLoss_finetune_0.0", "iiLoss_finetune_0.0")]
    final += [(f"{obj}_nested", f"{obj}_finetune_{log['selected'][obj]}") for obj in objectives]
    final += [(f"{obj}_finetune_{FIXED[obj]}",) * 2 for obj in objectives]
    if "iiLoss" in objectives and eps_check:
        final += [(f"iiLoss_finetune_eps{e:g}_0.2", f"iiLoss_finetune_eps{e:g}_0.2")
                  for e in EPS_CHECK]
    for run_name, model_name in final:
        pred = train_predict(model_name, ck_out, train)
        with open(os.path.join(fdir, f"{run_name}.json"), "w") as f:
            json.dump({labels[i]: float(pred[i]) for i in test}, f)

    with open(os.path.join(fdir, "selection.json"), "w") as f:
        json.dump(log, f, indent=1)
    with open(os.path.join(fdir, "DONE"), "w") as f:
        f.write(f"{time.time() - t0:.0f}s\n")
    print(f"s{seed} f{fold}: finished in {time.time() - t0:.0f} s", flush=True)


# -- Merge ----------------------------------------------------------------

def merge(width):
    tag = width_tag(width)
    root = os.path.join(ML_DIR, f"nested{tag}")
    with open(HULLOUT) as f:
        keys = set(json.load(f))
    os.makedirs(LOG_DIR, exist_ok=True)
    for sdir in sorted(glob.glob(os.path.join(root, "s*"))):
        seed = os.path.basename(sdir)
        folds = sorted(glob.glob(os.path.join(sdir, "f*")))
        if len(folds) != 5 or not all(os.path.exists(os.path.join(f, "DONE")) for f in folds):
            raise SystemExit(f"{sdir}: incomplete")
        runs = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(folds[0], "*.json"))
                      if not p.endswith("selection.json"))
        for run in runs:
            merged = {}
            for fdir in folds:
                with open(os.path.join(fdir, f"{run}.json")) as f:
                    part = json.load(f)
                if set(part) & set(merged):
                    raise SystemExit(f"{seed} {run}: a compound is in two folds")
                merged.update(part)
            if set(merged) != keys:
                raise SystemExit(f"{seed} {run}: {len(keys - set(merged))} compounds missing")
            out = os.path.join(ML_DIR, f"{run}{tag}_{seed}")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "ml_input.json"), "w") as f:
                json.dump(merged, f)
        for fdir in folds:
            name = f"selection{tag}_{seed}_{os.path.basename(fdir)}.json"
            shutil.copy(os.path.join(fdir, "selection.json"), os.path.join(LOG_DIR, name))
        print(f"{seed}: merged {len(runs)} runs, {len(keys)} compounds each")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["run", "merge"])
    parser.add_argument("args", nargs="*", type=int,
                        help="run: seed fold [base_width]; merge: [base_width]")
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument("--no-eps", action="store_true",
                        help="skip the denominator-stabiliser refits")
    args = parser.parse_args()

    objectives = [o for o in OBJECTIVES if o in args.objectives.split(",")]
    if not objectives or set(args.objectives.split(",")) - set(OBJECTIVES):
        raise SystemExit(f"objectives must be among {OBJECTIVES}")

    if args.mode == "run":
        if len(args.args) not in (2, 3):
            raise SystemExit("run needs: seed fold [base_width]")
        width = args.args[2] if len(args.args) == 3 else DEFAULT_BASE_WIDTH
        run_fold(args.args[0], args.args[1], width, objectives, not args.no_eps)
    else:
        merge(args.args[0] if args.args else DEFAULT_BASE_WIDTH)


if __name__ == "__main__":
    main()
