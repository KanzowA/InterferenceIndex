from os.path import dirname, abspath, join
import json
from sklearn.model_selection import KFold
from mlstabilitytest.training.MatminerModel import MatminerModel
from mlstabilitytest.training.iiLossNN import iiLossNN
from mlstabilitytest.training.EdLossNN import EdLossNN

try:
    from mlstabilitytest.training.ElemNetModel import ElemNet
except ImportError:
    ElemNet = None
    print("[process.py] ElemNet unavailable (keras/tensorflow not installed)")

try:
    from mlstabilitytest.training.AutoMatModel import AutoMat
except ImportError:
    AutoMat = None
    print("[process.py] AutoMat unavailable (automatminer not installed)")

try:
    from mlstabilitytest.training.ALIGNNModel import ALIGNNModel
except ImportError:
    ALIGNNModel = None
    print("[process.py] ALIGNN unavailable (alignn/jarvis-tools not installed)")

try:
    from mlstabilitytest.training.CHGNetModel import CHGNetModel
except ImportError:
    CHGNetModel = None
    print("[process.py] CHGNet unavailable (pip install chgnet)")

base_path = dirname(dirname(abspath(__file__)))
data_path = join(base_path, "mp_data", "data")

# Root of the repository (one level above mlstabilitytest/)
repo_path = dirname(base_path)

# Dictionary of available models — supports dynamic iiLoss_<lam> names,
# e.g. "iiLoss_0.0", "iiLoss_0.1", "iiLoss_0.25", "iiLoss_0.5"
_CHECKPOINT_DIR = join(repo_path, "checkpoints")

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
        # EdLoss_finetune_<alpha>  — stage 2: load MSE checkpoint, fine-tune with Ed²
        if key.startswith("EdLoss_finetune_"):
            try:
                alpha = float(key.split("_", 2)[2])
                return lambda target, a=alpha: EdLossNN(
                    target, alpha=a,
                    finetune_from=join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                )
            except (ValueError, IndexError):
                pass
        # EdLoss_pcgrad_<alpha>  — stage 2: fine-tune with gradient surgery
        if key.startswith("EdLoss_pcgrad_"):
            try:
                alpha = float(key.rsplit("_", 1)[1])
                return lambda target, a=alpha: EdLossNN(
                    target, alpha=a,
                    finetune_from=join(_CHECKPOINT_DIR, target),
                    finetune_lr=1e-4,
                    finetune_epochs=200,
                    warmup_frac=0.0,
                    renorm_every=0,
                    use_pcgrad=True,
                )
            except (ValueError, IndexError):
                pass
        # EdLoss_<alpha>  — original single-stage training
        if key.startswith("EdLoss_"):
            try:
                alpha = float(key.split("_", 1)[1])
                return lambda target, a=alpha: EdLossNN(target, alpha=a)
            except ValueError:
                pass
        raise KeyError("Unknown model '{}'. Available: {}".format(
            key, list(self.keys()) + ["iiLoss_<lam>", "iiLoss_save_<lam>",
                                      "iiLoss_finetune_<lam>", "iiLoss_pcgrad_<lam>",
                                      "EdLoss_<alpha>", "EdLoss_finetune_<alpha>",
                                      "EdLoss_pcgrad_<alpha>"]))

model_dictionary = _ModelDict({
    "Deml":     lambda target: MatminerModel('Deml', target),
    "ElFrac":   lambda target: MatminerModel('ElFrac', target),
    "Magpie":   lambda target: MatminerModel('Magpie', target),
    "Meredig":  lambda target: MatminerModel('Meredig', target),
    **( {"ElemNet": lambda target: ElemNet(target)} if ElemNet else {} ),
    **( {"AutoMat": lambda target: AutoMat(target)} if AutoMat else {} ),
    **( {"ALIGNN":  lambda target: ALIGNNModel(target)} if ALIGNNModel else {} ),
    **( {"CHGNet":  lambda target: CHGNetModel(target)} if CHGNetModel else {} ),
})

# List of available target properties
target_list = ["Ed", "Ef"]


# Functions to perform training and prediction for a problem, given an input model and target property
def allMP(model, target):

    input_file = join(data_path, "hullout.json")

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


def allMP_single(model, target, test_size=0.2, random_state=10):
    """80/20 train/test split — faster iteration than 5-fold CV."""
    from sklearn.model_selection import train_test_split

    input_file = join(data_path, "hullout.json")
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


def LiMnTMO(model, target):
    input_file_all_mp = join(data_path, "hullout.json")

    input_file_LiMnTMO = join(data_path, "mp_LiMnTMO.json")

    print("Reading input data from {},{}".format(
        input_file_all_mp, input_file_LiMnTMO))

    # Load the training set
    with open(input_file_all_mp, 'r') as f:
        input_data_all_mp = json.load(f)

    # Load the test set
    with open(input_file_LiMnTMO, 'r') as f:
        input_data_LiMnTMO = json.load(f)

    # Make sure to exclude the test set from the training set
    input_data_all_mp = {k: v for k, v in input_data_all_mp.items(
    ) if k not in input_data_LiMnTMO.keys()}

    print("Preprocessing data")
    features_train, targets_train, _ = model.preprocess(input_data_all_mp)

    # Save training-time state before the test preprocess call overwrites it.
    # iiLossNN stores self._data and self._labels during preprocess; fit()
    # needs these to reflect the *training* set when building the reaction graph.
    _saved_data   = getattr(model, '_data',   None)
    _saved_labels = getattr(model, '_labels', None)

    features_test, _, labels_test = model.preprocess(input_data_LiMnTMO)

    # Restore training state so fit() builds the reaction graph correctly.
    if _saved_data is not None:
        model._data   = _saved_data
    if _saved_labels is not None:
        model._labels = _saved_labels

    print("Training")
    predictions = model.fit_and_predict(
        Xtrain=features_train, Ytrain=targets_train, Xtest=features_test)

    predictions = {labels_test[i]: predictions[i]
                   for i, x in enumerate(labels_test)}

    return predictions


def smact(model, target):
    input_file_all_mp = join(data_path, "hullout.json")

    input_file_LiMnTMO = join(data_path, "mp_LiMnTMO.json")

    input_file_smact_LiMnTMO = join(data_path, "smact_LiMnTMO.json")

    print("Reading input data from {},{}, {}".format(
        input_file_all_mp, input_file_LiMnTMO, input_file_smact_LiMnTMO))

    # Load the training set
    with open(input_file_all_mp, 'r') as f:
        input_data_all_mp = json.load(f)

    # Load the test set
    with open(input_file_LiMnTMO, 'r') as f:
        input_data_LiMnTMO = json.load(f)

    # Load the smact formulae
    with open(input_file_smact_LiMnTMO, 'r') as f:
        input_data_smact_LiMnTMO = json.load(f)['smact']

    # The smact formulae are stored as a list of formulae
    # Convert to a dictionary of formulae: dummy value for consistency of data format
    # The dummy value is irrelevant since we only use these for testing
    dummy_dict = {target: 0 for target in target_list}

    input_data_smact_LiMnTMO = {
        formula: dummy_dict for formula in input_data_LiMnTMO}

    # Combine the smact and MP test sets
    input_data_LiMnTMO = {**input_data_LiMnTMO, **input_data_smact_LiMnTMO}

    # Make sure to exclude the test set from the training set
    input_data_all_mp = {k: v for k, v in input_data_all_mp.items(
    ) if k not in input_data_LiMnTMO.keys()}

    print("Preprocessing data")
    features_train, targets_train, _ = model.preprocess(input_data_all_mp)

    features_test, _, labels_test = model.preprocess(input_data_LiMnTMO)

    print("Training")
    predictions = model.fit_and_predict(
        Xtrain=features_train, Ytrain=targets_train, Xtest=features_test)

    predictions = {labels_test[i]: predictions[i]
                   for i, x in enumerate(labels_test)}

    return predictions


def allMP_current(model, target):
    """
    Same 5-fold CV as allMP, but trained/evaluated on the 2026 MP dataset
    (hullout_current.json, downloaded via download_mp_current.py).
    Saves predictions to ml_data/Ef/allMP_current/<model>/ml_input.json.
    Run interference_score.py with --split allMP_current --hullout hullout_current.json.
    """
    input_file = join(repo_path, "hullout_current.json")

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


def allMP_current_single(model, target, test_size=0.2, random_state=10):
    """
    80/20 single split on the 2026 MP dataset — 5× faster than allMP_current.
    Use for sweeps where relative ranking matters more than CV stability.
    """
    from sklearn.model_selection import train_test_split

    input_file = join(repo_path, "hullout_current.json")
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


def allMP_current_finetune(model, target):
    """
    Two-stage fine-tuning on allMP_current.
    Uses the same 5-fold CV split as allMP_current (random_state=10).
    The model (iiLoss_finetune_<lam>) loads fold-specific pretrained weights
    from checkpoints/<target>/ before computing references and fine-tuning.
    Run after saving checkpoints with iiLoss_save_0.0.
    """
    return allMP_current(model, target)


def allMP_current_v2(model, target):
    """
    Same 5-fold CV as allMP_current, but reads hullout_current_v2.json which
    includes material_id fields required by structure-based models (ALIGNN etc.).
    """
    input_file = join(repo_path, "hullout_current_v2.json")

    print("Reading 2026 MP data (v2, with material_ids) from {}".format(input_file))

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


# Dictionary of available problem functions
problem_dictionary = {"allMP":                   allMP,
                      "allMP_single":            allMP_single,
                      "allMP_current":           allMP_current,
                      "allMP_current_single":    allMP_current_single,
                      "allMP_current_finetune":  allMP_current_finetune,
                      "allMP_current_v2":        allMP_current_v2,
                      "LiMnTMO":                 LiMnTMO,
                      "smact":                   smact}
