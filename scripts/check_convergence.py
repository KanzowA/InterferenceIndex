"""Diagnostic: is the stage-two plateau set by the objective or by the schedule?

Runs one fine-tuning configuration at constant learning rate for a longer
budget than the paper protocol, so that flatness at the end cannot be an
artefact of cosine annealing. Not part of the pipeline that produces the
reported results.

Usage
-----
    python scripts/check_convergence.py                     # HdLoss, lam 0.5
    python scripts/check_convergence.py --model iiLoss --lam 0.2
    python scripts/check_convergence.py --epochs 600
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

from process import PROBLEM_DICTIONARY, DEFAULT_BASE_WIDTH, hidden_from_base
from models.HdLossNN import HdLossNN
from models.iiLossNN import iiLossNN

MODELS = {"iiLoss": iiLossNN, "HdLoss": HdLossNN}


class _ConstantLR:
    """Stand-in for CosineAnnealingLR that never changes the rate."""

    def __init__(self, optimizer, *args, **kwargs):
        self.optimizer = optimizer

    def step(self):
        pass


def report(path, window=20):
    """Fraction of the total decrease that falls in the last window."""
    import pandas as pd
    df = pd.read_csv(path).groupby("epoch").mean(numeric_only=True)
    last = df.index.max()
    for col in ("mse_over_ref", "penalty_over_ref"):
        series = df[col]
        total = series.iloc[0] - series.loc[last]
        tail = series.loc[last - window] - series.loc[last]
        share = 100 * tail / total if total else float("nan")
        print(f"  {col:<18} start {series.iloc[0]:.4f}  end {series.loc[last]:.4f}"
              f"  last {window} epochs {tail:+.5f} ({share:.2f}% of total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HdLoss", choices=list(MODELS))
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="Hf")
    parser.add_argument("--problem", default="allMP_2026")
    args = parser.parse_args()

    out_dir = os.path.join(REPO_ROOT, "logs", "convergence_check")
    os.makedirs(out_dir, exist_ok=True)
    history = os.path.join(
        out_dir, f"{args.model}_{args.lam}_ep{args.epochs}_constlr.csv")
    if os.path.exists(history):
        os.remove(history)

    # Constant learning rate, so a flat tail cannot come from the schedule.
    torch.optim.lr_scheduler.CosineAnnealingLR = _ConstantLR

    model = MODELS[args.model](
        args.target, lam=args.lam, seed=args.seed,
        hidden=hidden_from_base(DEFAULT_BASE_WIDTH),
        finetune_from=os.path.join(REPO_ROOT, "checkpoints", args.target),
        finetune_lr=1e-4,
        finetune_epochs=args.epochs,
        warmup_frac=0.0,
        renorm_every=0,
        history_path=history,
    )

    print(f"{args.model} lam={args.lam}, {args.epochs} epochs, constant lr 1e-4")
    PROBLEM_DICTIONARY[args.problem](model, args.target)

    print(f"\nWrote {history}")
    report(history)


if __name__ == "__main__":
    main()
