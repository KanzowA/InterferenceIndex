from os.path import dirname, abspath, join
import json
from sklearn.model_selection import KFold
from mlstabilitytest.training.MatminerModel import MatminerModel
from mlstabilitytest.training.GammaLossNN import GammaLossNN
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

base_path = dirname(dirname(abspath(__file__)))
data_path = join(base_path, "mp_data", "data")

# Dictionary of available models — supports dynamic GammaLoss_<lam> names,
# e.g. "GammaLoss_0.0", "GammaLoss_0.1", "GammaLoss_0.25", "GammaLoss_0.5"
class _ModelDict(dict):
    def __missing__(self, key):
        if key.startswith("GammaLoss_"):
            try:
                lam = float(key.split("_", 1)[1])
                return lambda target, l=lam: GammaLossNN(target, lam=l)
            except ValueError:
                pass
        if key.startswith("EdLoss_"):
            try:
                alpha = float(key.split("_", 1)[1])
                return lambda target, a=alpha: EdLossNN(target, alpha=a)
            except ValueError:
                pass
        raise KeyError("Unknown model '{}'. Available: {}".format(
            key, list(self.keys()) + ["GammaLoss_<lam>", "EdLoss_<alpha>"]))

model_dictionary = _ModelDict({
    "Deml":     lambda target: MatminerModel('Deml', target),
    "ElFrac":   lambda target: MatminerModel('ElFrac', target),
    "Magpie":   lambda target: MatminerModel('Magpie', target),
    "Meredig":  lambda target: MatminerModel('Meredig', target),
    **( {"ElemNet": lambda target: ElemNet(target)} if ElemNet else {} ),
    **( {"AutoMat": lambda target: AutoMat(target)} if AutoMat else {} ),
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
    # GammaLossNN stores self._data and self._labels during preprocess; fit()
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


# Dictionary of available problem functions
problem_dictionary = {"allMP": allMP,
                      "allMP_single": allMP_single,
                      "LiMnTMO": LiMnTMO,
                      "smact": smact}
