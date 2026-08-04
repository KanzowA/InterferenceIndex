"""Report how much of each decomposition reaction the training penalty sees.

The xi^2 penalty is built only from compounds in the training fold, so products
that fall in the held-out fold are dropped and reactions left with fewer than
two participants are skipped. This script reproduces that construction for
every fold and reports the retained fraction, for the sentence in Methods.

Runs on CPU and trains nothing.

Usage
-----
    python scripts/reaction_coverage.py
    python scripts/reaction_coverage.py --dataset 2020
"""

import argparse
import json
import os
import re
import sys
from statistics import mean

from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from models.iiLossNN import _is_element, _parse_rxn, iiLossNN

_TOKEN = re.compile(r"^([\d.]+)_(.+)$")


def _n_products(rxn_str):
    """Count non-elemental products, whether or not they are in the fold."""
    total = 0
    for token in rxn_str.split("+"):
        match = _TOKEN.match(token.strip())
        if match and not _is_element(match.group(2).strip()):
            total += 1
    return total


def coverage(data, labels, train_indices):
    """Retained reaction count and per-reaction product coverage for one fold."""
    train_labels = labels[train_indices]
    label_to_pos = {lbl: i for i, lbl in enumerate(train_labels)}

    n_with_rxn, n_retained, fractions = 0, 0, []
    for lbl in train_labels:
        rxn_str = data.get(lbl, {}).get("rxn", "")
        if not rxn_str:
            continue
        n_with_rxn += 1

        total = _n_products(rxn_str)
        kept = _parse_rxn(rxn_str, label_to_pos)
        # The reactant itself always contributes, with coefficient -1.
        if len(kept) + 1 < 2:
            continue
        n_retained += 1
        if total:
            fractions.append(len(kept) / total)

    return n_with_rxn, n_retained, fractions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2026", choices=["2020", "2026"])
    args = parser.parse_args()

    path = os.path.join(REPO_ROOT, "data", args.dataset,
                        f"hullout_{args.dataset}.json")
    with open(path) as handle:
        data = json.load(handle)

    # preprocess fixes the label order that KFold then slices, so the folds
    # match those used in training exactly.
    features, _, labels = iiLossNN().preprocess(data)
    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    print(f"{'fold':>5}{'with rxn':>11}{'retained':>11}{'retained %':>13}"
          f"{'mean products kept %':>23}")
    totals_with, totals_kept, all_fracs = 0, 0, []
    for fold, (train_indices, _) in enumerate(kf.split(features)):
        n_with, n_kept, fracs = coverage(data, labels, train_indices)
        totals_with += n_with
        totals_kept += n_kept
        all_fracs.extend(fracs)
        print(f"{fold:>5}{n_with:>11}{n_kept:>11}"
              f"{100 * n_kept / n_with:>12.1f}%"
              f"{100 * mean(fracs):>22.1f}%")

    print(f"\n  reactions retained:        {100 * totals_kept / totals_with:.1f}%"
          f"  ({totals_kept} of {totals_with})")
    print(f"  mean products kept:        {100 * mean(all_fracs):.1f}%")
    print(f"  reactions with all products kept: "
          f"{100 * sum(f == 1.0 for f in all_fracs) / len(all_fracs):.1f}%")


if __name__ == "__main__":
    main()
