"""Bring the revision's cluster runs into this repository's layout and names.

The nested, capacity and prospective runs were trained on the cluster from the
working repository of the revision (ii-revision), whose models are bit-identical
to models/iiLossNN.py and models/HdLossNN.py here (scripts/check_equivalence.py)
and whose nested runner is scripts/nested_cv.py under another name. This script
only renames and moves their predictions; it changes no number.

Usage
-----
    python scripts/import_revision_runs.py --nested   <ii-revision>/runs_nested
    python scripts/import_revision_runs.py --capacity <ii-revision>/runs_capacity
    python scripts/import_revision_runs.py --prospective <ii-revision>/runs_wbm

Names in ii-revision -> names here

    control                         iiLoss_finetune_0.0
    iiperpfrozen_nested             iiLoss_nested
    iiperpfrozen_fixed0.2           iiLoss_finetune_0.2
    iiperpfrozen_fixed0.2_eps<e>    iiLoss_finetune_eps<e>_0.2
    iiloss_nested                   iiLossRaw_nested
    iiloss_fixed0.2                 iiLossRaw_finetune_0.2
    hdloss_nested                   HdLoss_nested
    hdloss_fixed0.1                 HdLoss_finetune_0.1

and for the prospective runs (one model per seed, trained on all of MP)

    stage1                          iiLoss_save_0.0
    mse_0                           iiLoss_finetune_0.0
    iiperpfrozen_0.2                iiLoss_finetune_0.2
    iiloss_0.2                      iiLossRaw_finetune_0.2
    hdloss_0.1                      HdLoss_finetune_0.1
"""

import argparse
import glob
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

ML_DIR = os.path.join(REPO_ROOT, "data", "2025", "ml", "Hf")
LOG_DIR = os.path.join(REPO_ROOT, "results", "2025", "nested")
WBM_ML_DIR = os.path.join(REPO_ROOT, "data", "wbm", "ml")

NESTED = {"control": "iiLoss_finetune_0.0",
          "iiperpfrozen_nested": "iiLoss_nested",
          "iiperpfrozen_fixed0.2": "iiLoss_finetune_0.2",
          "iiperpfrozen_fixed0.2_eps1e-05": "iiLoss_finetune_eps1e-05_0.2",
          "iiperpfrozen_fixed0.2_eps0.001": "iiLoss_finetune_eps0.001_0.2",
          "iiloss_nested": "iiLossRaw_nested",
          "iiloss_fixed0.2": "iiLossRaw_finetune_0.2",
          "hdloss_nested": "HdLoss_nested",
          "hdloss_fixed0.1": "HdLoss_finetune_0.1"}
OBJECTIVE = {"iiperpfrozen": "iiLoss", "iiloss": "iiLossRaw", "hdloss": "HdLoss"}
PROSPECTIVE = {"stage1": "iiLoss_save_0.0", "mse_0": "iiLoss_finetune_0.0",
               "iiperpfrozen_0.2": "iiLoss_finetune_0.2",
               "iiloss_0.2": "iiLossRaw_finetune_0.2", "hdloss_0.1": "HdLoss_finetune_0.1"}


def convert_selection(log, width):
    """Selection log with this repository's objective names."""
    val = {}
    for key, value in log["val"].items():
        if key == "mse_0":
            val["control"] = value
        else:
            obj, lam = key.rsplit("_", 1)
            val[f"{OBJECTIVE[obj]}_{lam}"] = value
    out = {k: v for k, v in log.items() if k not in ("val", "selected", "selected_by_F1", "arms")}
    out["width"] = width                  # absent from the logs of the 1024 run
    arms = log.get("arms", ["iiloss", "hdloss", "iiperpfrozen"])
    out["objectives"] = [OBJECTIVE[a] for a in arms]
    out["val"] = val
    out["selected"] = {OBJECTIVE[k]: v for k, v in log["selected"].items()}
    out["selected_by_F1"] = {OBJECTIVE[k]: v for k, v in log["selected_by_F1"].items()}
    return out


def import_cv(root, tag):
    n = 0
    for path in sorted(glob.glob(os.path.join(root, "preds", "*_s*", "ml_input.json"))):
        run = os.path.basename(os.path.dirname(path))
        arm, seed = re.match(r"^(.*)_(s\d+)$", run).groups()
        dst = os.path.join(ML_DIR, f"{NESTED[arm]}{tag}_{seed}")
        os.makedirs(dst, exist_ok=True)
        shutil.copy(path, os.path.join(dst, "ml_input.json"))
        n += 1
    os.makedirs(LOG_DIR, exist_ok=True)
    m = 0
    for path in sorted(glob.glob(os.path.join(root, "folds", "s*", "f*", "selection.json"))):
        fold = os.path.basename(os.path.dirname(path))
        seed = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as f:
            log = convert_selection(json.load(f), int(tag[2:]) if tag else 1024)
        with open(os.path.join(LOG_DIR, f"selection{tag}_{seed}_{fold}.json"), "w") as f:
            json.dump(log, f, indent=1)
        m += 1
    print(f"{root}: {n} prediction sets, {m} selection logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nested")
    parser.add_argument("--capacity")
    parser.add_argument("--prospective")
    args = parser.parse_args()

    if args.nested:
        import_cv(args.nested, "")
    if args.capacity:
        for wdir in sorted(glob.glob(os.path.join(args.capacity, "w*"))):
            import_cv(wdir, f"_{os.path.basename(wdir)}")
    if args.prospective:
        os.makedirs(WBM_ML_DIR, exist_ok=True)
        n = 0
        for path in sorted(glob.glob(os.path.join(args.prospective, "preds", "*.csv.gz"))):
            run, seed = re.match(r"^(.*)_(s\d+)\.csv\.gz$", os.path.basename(path)).groups()
            shutil.copy(path, os.path.join(WBM_ML_DIR, f"{PROSPECTIVE[run]}_{seed}.csv.gz"))
            n += 1
        print(f"{args.prospective}: {n} prediction sets")


if __name__ == "__main__":
    main()
