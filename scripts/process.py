from os.path import dirname, abspath, join
import json
import sys
from sklearn.model_selection import KFold

# Add root directory to path so we can import models
root_path = dirname(dirname(abspath(__file__)))
sys.path.insert(0, root_path)

from models.iiLossNN import iiLossNN
from models.HdLossNN import HdLossNN

base_path = dirname(dirname(abspath(__file__)))

# Dictionary of available models — supports dynamic iiLoss_<lam> names,
# e.g. "iiLoss_0.0", "iiLoss_0.1", "iiLoss_0.25", "iiLoss_0.5"
_CHECKPOINT_DIR = join(base_path, "checkpoints")

class _ModelDict(dict):
    def __missing__(self, key):
        # iiLoss_save_<lam>  — stage 1: train with λ, save checkpoint
        if key.startswith("iiLoss_save_"):
            try:
                lam = float(key.split("_", 2)[2])
                return lambda target, l=lam: iiLossNN(
                    target, lam=l,
                    checkpoint_dir=join(_CHECKPOINT_DIR, target),
                )
            except (ValueError, IndexError):
                pass
        # iiLoss_finetune_<lam>  — stage 2: load checkpoint, fine-tune with λ
        if key.startswith("iiLoss_finetune_"):
            try:
                lam = float(key.split("_", 2)[2])
                return lambda target, l=lam: iiLossNN(
                    target, lam=l,
                    finetune_from=join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                )
            except (ValueError, IndexError):
                pass
        # iiLoss_pcgrad_<lam>  — stage 2: fine-tune with gradient surgery (PCGrad)
        if key.startswith("iiLoss_pcgrad_"):
            try:
                lam = float(key.rsplit("_", 1)[1])
                return lambda target, l=lam: iiLossNN(
                    target, lam=l,
                    finetune_from=join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                    use_pcgrad=True,
                )
            except (ValueError, IndexError):
                pass
        # iiLoss_<lam>  — original single-stage training
        if key.startswith("iiLoss_"):
            try:
                lam = float(key.split("_", 1)[1])
                return lambda target, l=lam: iiLossNN(target, lam=l)
            except ValueError:
                pass
        # HdLoss_finetune_<alpha>  — stage 2: load MSE checkpoint, fine-tune with Ed²
        if key.startswith("HdLoss_finetune_"):
            try:
                alpha = float(key.split("_", 2)[2])
                return lambda target, a=alpha: HdLossNN(
                    target, lam=a,
                    finetune_from=join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                )
            except (ValueError, IndexError):
                pass
        # HdLoss_pcgrad_<alpha>  — stage 2: fine-tune with gradient surgery
        if key.startswith("HdLoss_pcgrad_"):
            try:
                alpha = float(key.rsplit("_", 1)[1])
                return lambda target, a=alpha: HdLossNN(
                    target, lam=a,
                    finetune_from=join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                    use_pcgrad=True,
                )
            except (ValueError, IndexError):
                pass
        # HdLoss_<alpha>  — original single-stage training
        if key.startswith("HdLoss_"):
            try:
                alpha = float(key.split("_", 1)[1])
                return lambda target, a=alpha: HdLossNN(target, lam=a)
            except ValueError:
                pass
        raise KeyError("Unknown model '{}'. Available: {}".format(
            key, list(self.keys()) + ["iiLoss_<lam>", "iiLoss_save_<lam>",
                                      "iiLoss_finetune_<lam>", "iiLoss_pcgrad_<lam>",
                                      "HdLoss_<alpha>", "HdLoss_finetune_<alpha>",
                                      "HdLoss_pcgrad_<alpha>"]))

model_dictionary = _ModelDict({})

# List of available target properties
target_list = ["Hd", "Hf"]


# Functions to perform training and prediction for a problem, given an input model and target property
def allMP_2020(model, target):

    input_file = join(base_path, "data", "2020", "hullout_2020.json")

    print("Reading input data from {}".format(input_file))

    with open(input_file, 'r') as f:
        input_data = json.load(f)

    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

    predictions = dict()

    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    iFold = 0
    for train_indices, test_indices in kf.split(features):
        print("Training on fold {}".format(iFold))
        iFold += 1

        features_train = features[train_indices]
        targets_train = targets[train_indices]

        features_test = features[test_indices]
        targets_test = targets[test_indices]
        labels_test = labels[test_indices]

        predictions_this_fold = model.fit_and_predict(
            Xtrain=features_train, Ytrain=targets_train, Xtest=features_test)

        predictions = {**predictions, **{labels_test[i]: predictions_this_fold[i] for i, x in enumerate(test_indices)}}

    return predictions


def allMP_2020_single(model, target, test_size=0.2, random_state=10):
    """80/20 train/test split — faster iteration than 5-fold CV."""
    from sklearn.model_selection import train_test_split

    input_file = join(base_path, "data", "2020", "hullout_2020.json")
    print("Reading input data from {}".format(input_file))
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

    idx = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=True)

    print("Training on 80/20 split ({} train, {} test)".format(
        len(train_idx), len(test_idx)))
    preds = model.fit_and_predict(
        Xtrain=features[train_idx],
        Ytrain=targets[train_idx],
        Xtest=features[test_idx])

    return {labels[test_idx[i]]: preds[i] for i in range(len(test_idx))}


def allMP_2026(model, target):
    """
    Same 5-fold CV as allMP, but trained/evaluated on the 2026 MP dataset
    (hullout_2026.json).
    """
    input_file = join(base_path, "data", "2026", "hullout_2026.json")

    print("Reading 2026 MP data from {}".format(input_file))

    with open(input_file, 'r') as f:
        input_data = json.load(f)

    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

    predictions = dict()

    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    iFold = 0
    for train_indices, test_indices in kf.split(features):
        print("Training on fold {}".format(iFold))
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
    """
    80/20 single split on the 2026 MP dataset — 5× faster than allMP_2026.
    Use for sweeps where relative ranking matters more than CV stability.
    """
    from sklearn.model_selection import train_test_split

    input_file = join(base_path, "data", "2026", "hullout_2026.json")
    print("Reading 2026 MP data from {}".format(input_file))
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

    idx = list(range(len(labels)))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=True)

    print("Training on 80/20 split ({} train, {} test)".format(
        len(train_idx), len(test_idx)))
    preds = model.fit_and_predict(
        Xtrain=features[train_idx],
        Ytrain=targets[train_idx],
        Xtest=features[test_idx])

    return {labels[test_idx[i]]: preds[i] for i in range(len(test_idx))}


# Dictionary of available problem functions
problem_dictionary = {"allMP_2020":           allMP_2020,
                      "allMP_2020_single":    allMP_2020_single,
                      "allMP_2026":           allMP_2026,
                      "allMP_2026_single":    allMP_2026_single}
