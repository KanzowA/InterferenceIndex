"""Train one model on the MP data and write its out-of-sample predictions.

Usage
-----
    python scripts/train_models.py <problem> <target> <model>
    python scripts/train_models.py allMP_2026 Hf iiLoss_pcgrad_0.3

Predictions are written to data/<year>/ml/<target>/<model>/ml_input.json in the
format the hull analysis expects.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# process.py sits beside this file, which is not on the path when the script
# is run from the repository root.
sys.path.insert(0, HERE)

from process import MODEL_DICTIONARY, PROBLEM_DICTIONARY, TARGET_LIST


def main(argv):
    try:
        problem = argv[1]
        target = argv[2]
        model_name = argv[3]
    except IndexError:
        print("Arguments should be Problem Target Model")
        exit(1)

    try:
        train_func = PROBLEM_DICTIONARY[problem]
    except KeyError:
        choices = ", ".join(PROBLEM_DICTIONARY.keys())
        print(f"Invalid problem selection {problem}. Valid choices are {choices}")
        exit(1)
    try:
        if not target in TARGET_LIST:
            raise ValueError
    except ValueError:
        choices = ", ".join(TARGET_LIST)
        print(f"Invalid target selection. Valid choices are {choices}")
        exit(1)

    try:
        model = MODEL_DICTIONARY[model_name](target)
    except KeyError:
        choices = ", ".join(MODEL_DICTIONARY.keys())
        print(f"Invalid model selection. Valid choices are {choices}")
        exit(1)

    year = "2026" if "2026" in problem else "2020"
    output_dir = os.path.join(REPO_ROOT, "data", year, "ml", target, model_name)
    output_file = os.path.join(output_dir, "ml_input.json")

    print(f"Training {model_name} to predict {target} using the {problem} dataset")

    predictions = train_func(model, target)

    print(f"Training complete, saving predictions to {output_file}")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main(sys.argv)
