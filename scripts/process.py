"""Dataset splits and the model registry used by train_models.py.

Problems are cross-validation protocols over an MP snapshot; models are named
by their configuration, e.g. "iiLoss_pcgrad_0.3", and constructed on lookup.
"""

import json
import os
import sys

from sklearn.model_selection import KFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# models/ lives at the repository root, which is not on the path when this
# module is imported from scripts/.
sys.path.insert(0, REPO_ROOT)

from models.HdLossNN import HdLossNN
from models.iiLossNN import iiLossNN

# Model names encode their configuration, e.g. "iiLoss_pcgrad_0.3", and are
# resolved on lookup rather than enumerated.
_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")

# Hidden widths halve at each residual block, so one base width fixes the
# whole stack. Varying it traces the capacity axis with the shape held fixed.
DEFAULT_BASE_WIDTH = 1024


def hidden_from_base(base, n_layers=4):
    return tuple(base >> k for k in range(n_layers))

class _ModelDict(dict):
    def __missing__(self, key):
        # Stage 1: train with lam and save a checkpoint.
        if key.startswith("iiLoss_save_"):
            try:
                lam = float(key.split("_", 2)[2])
                return lambda target, seed=0, hidden=None, l=lam: iiLossNN(
                    target, lam=l, seed=seed,
                    hidden=hidden or hidden_from_base(DEFAULT_BASE_WIDTH),
                    checkpoint_dir=os.path.join(_CHECKPOINT_DIR, target),
                )
            except (ValueError, IndexError):
                pass
        # Stage 2 with a non-default denominator stabiliser, for the
        # sensitivity check: "iiLoss_finetune_eps1e-3_0.2".
        if key.startswith("iiLoss_finetune_eps"):
            try:
                eps_tag, lam_tag = key[len("iiLoss_finetune_eps"):].split("_")
                eps, lam = float(eps_tag), float(lam_tag)
                return lambda target, seed=0, hidden=None, l=lam, e=eps: iiLossNN(
                    target, lam=l, eps=e, seed=seed,
                    hidden=hidden or hidden_from_base(DEFAULT_BASE_WIDTH),
                    finetune_from=os.path.join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                )
            except (ValueError, IndexError):
                pass
        # Stage 2: reload the checkpoint and fine-tune with lam.
        if key.startswith("iiLoss_finetune_"):
            try:
                lam = float(key.split("_", 2)[2])
                return lambda target, seed=0, hidden=None, l=lam: iiLossNN(
                    target, lam=l, seed=seed,
                    hidden=hidden or hidden_from_base(DEFAULT_BASE_WIDTH),
                    finetune_from=os.path.join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                )
            except (ValueError, IndexError):
                pass
        # Stage 2 with gradient surgery.
        if key.startswith("iiLoss_pcgrad_"):
            try:
                lam = float(key.rsplit("_", 1)[1])
                return lambda target, seed=0, hidden=None, l=lam: iiLossNN(
                    target, lam=l, seed=seed,
                    hidden=hidden or hidden_from_base(DEFAULT_BASE_WIDTH),
                    finetune_from=os.path.join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                    use_pcgrad=True,
                )
            except (ValueError, IndexError):
                pass
        # Stage 2 with gradient surgery applied only on conflicting steps.
        if key.startswith("iiLoss_pcgradc_"):
            try:
                lam = float(key.rsplit("_", 1)[1])
                return lambda target, seed=0, hidden=None, l=lam: iiLossNN(
                    target, lam=l, seed=seed,
                    hidden=hidden or hidden_from_base(DEFAULT_BASE_WIDTH),
                    finetune_from=os.path.join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                    use_pcgrad=True,
                    pcgrad_conditional=True,
                )
            except (ValueError, IndexError):
                pass
        # Single-stage training.
        if key.startswith("iiLoss_"):
            try:
                lam = float(key.split("_", 1)[1])
                return lambda target, seed=0, hidden=None, l=lam: iiLossNN(
                    target, lam=l, seed=seed,
                    hidden=hidden or hidden_from_base(DEFAULT_BASE_WIDTH))
            except ValueError:
                pass
        # Stage 2: reload the MSE checkpoint and fine-tune with Hd^2.
        if key.startswith("HdLoss_finetune_"):
            try:
                alpha = float(key.split("_", 2)[2])
                return lambda target, seed=0, hidden=None, a=alpha: HdLossNN(
                    target, lam=a, seed=seed,
                    hidden=hidden or hidden_from_base(DEFAULT_BASE_WIDTH),
                    finetune_from=os.path.join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                )
            except (ValueError, IndexError):
                pass
        # Stage 2 with gradient surgery.
        if key.startswith("HdLoss_pcgrad_"):
            try:
                alpha = float(key.rsplit("_", 1)[1])
                return lambda target, seed=0, hidden=None, a=alpha: HdLossNN(
                    target, lam=a, seed=seed,
                    hidden=hidden or hidden_from_base(DEFAULT_BASE_WIDTH),
                    finetune_from=os.path.join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                    use_pcgrad=True,
                )
            except (ValueError, IndexError):
                pass
        # Single-stage training.
        if key.startswith("HdLoss_"):
            try:
                alpha = float(key.split("_", 1)[1])
                return lambda target, seed=0, hidden=None, a=alpha: HdLossNN(
                    target, lam=a, seed=seed,
                    hidden=hidden or hidden_from_base(DEFAULT_BASE_WIDTH))
            except ValueError:
                pass
        known = list(self.keys()) + ["iiLoss_<lam>", "iiLoss_save_<lam>",
                                     "iiLoss_finetune_<lam>", "iiLoss_pcgrad_<lam>",
                                     "iiLoss_pcgradc_<lam>",
                                     "iiLoss_finetune_eps<eps>_<lam>",
                                     "HdLoss_<alpha>", "HdLoss_finetune_<alpha>",
                                     "HdLoss_pcgrad_<alpha>"]
        raise KeyError(f"Unknown model '{key}'. Available: {known}")

MODEL_DICTIONARY = _ModelDict({})

TARGET_LIST = ["Hd", "Hf"]


# -- Problem definitions --------------------------------------------------
def allMP_2020(model, target):

    input_file = os.path.join(REPO_ROOT, "data", "2020", "hullout_2020.json")

    print(f"Reading input data from {input_file}")

    with open(input_file, "r") as f:
        input_data = json.load(f)

    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

    predictions = dict()

    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    iFold = 0
    for train_indices, test_indices in kf.split(features):
        print(f"Training on fold {iFold}")
        iFold += 1

        features_train = features[train_indices]
        targets_train = targets[train_indices]

        features_test = features[test_indices]
        labels_test = labels[test_indices]

        predictions_this_fold = model.fit_and_predict(
            Xtrain=features_train, Ytrain=targets_train, Xtest=features_test)

        predictions = {**predictions, **{labels_test[i]: predictions_this_fold[i] for i in range(len(test_indices))}}

    return predictions


def allMP_2020_single(model, target, test_size=0.2, random_state=10):
    """80/20 train/test split - faster iteration than 5-fold CV."""
    input_file = os.path.join(REPO_ROOT, "data", "2020", "hullout_2020.json")
    print(f"Reading input data from {input_file}")
    with open(input_file, "r") as f:
        input_data = json.load(f)

    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

    idx = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=True)

    print(f"Training on 80/20 split ({len(train_idx)} train, "
          f"{len(test_idx)} test)")
    preds = model.fit_and_predict(
        Xtrain=features[train_idx],
        Ytrain=targets[train_idx],
        Xtest=features[test_idx])

    return {labels[test_idx[i]]: preds[i] for i in range(len(test_idx))}


def allMP_2026(model, target):
    """Five-fold cross-validation on the 2026 MP snapshot."""
    input_file = os.path.join(REPO_ROOT, "data", "2026", "hullout_2026.json")

    print(f"Reading 2026 MP data from {input_file}")

    with open(input_file, "r") as f:
        input_data = json.load(f)

    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

    predictions = dict()

    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    iFold = 0
    for train_indices, test_indices in kf.split(features):
        print(f"Training on fold {iFold}")
        iFold += 1

        features_train = features[train_indices]
        targets_train  = targets[train_indices]
        features_test  = features[test_indices]
        labels_test    = labels[test_indices]

        predictions_this_fold = model.fit_and_predict(
            Xtrain=features_train, Ytrain=targets_train, Xtest=features_test)

        predictions = {**predictions,
                       **{labels_test[i]: predictions_this_fold[i]
                          for i in range(len(test_indices))}}

    return predictions


def allMP_2026_single(model, target, test_size=0.2, random_state=10):
    """80/20 split on the 2026 MP snapshot, for faster sweeps."""
    input_file = os.path.join(REPO_ROOT, "data", "2026", "hullout_2026.json")
    print(f"Reading 2026 MP data from {input_file}")
    with open(input_file, "r") as f:
        input_data = json.load(f)

    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

    idx = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=True)

    print(f"Training on 80/20 split ({len(train_idx)} train, "
          f"{len(test_idx)} test)")
    preds = model.fit_and_predict(
        Xtrain=features[train_idx],
        Ytrain=targets[train_idx],
        Xtest=features[test_idx])

    return {labels[test_idx[i]]: preds[i] for i in range(len(test_idx))}


PROBLEM_DICTIONARY = {"allMP_2020":           allMP_2020,
                      "allMP_2020_single":    allMP_2020_single,
                      "allMP_2026":           allMP_2026,
                      "allMP_2026_single":    allMP_2026_single}
