"""Check that iiLossRaw is the preprint's objective, to the last bit.

Usage
-----
    python scripts/check_equivalence.py              # against main:models/iiLossNN.py
    python scripts/check_equivalence.py --ref <git revision>

models/iiLossNN.py carries the gauge projection and the frozen normalisation,
on by default. With both switched off it must be the model of the first
preprint, unchanged; and at lam = 0 the two objectives must not differ at all.
This script trains small models both ways on CPU, single-threaded so the sums
are taken in a fixed order, and compares the predictions:

    1. stage one, preprint class vs this class
    2. lam = 0 fine-tune, preprint class vs this class (default switches)
    3. lam = 0.3 fine-tune, preprint class vs this class with both switches off
    4. the same with gradient surgery

Every difference must be exactly zero. The preprint's class is read from git,
so nothing of it needs to be kept in the working tree.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)


def load_reference(ref):
    """The preprint's models/iiLossNN.py at `ref`, as a module."""
    source = subprocess.run(["git", "show", f"{ref}:models/iiLossNN.py"], cwd=REPO_ROOT,
                            check=True, capture_output=True, text=True).stdout
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(source)
    tmp.close()
    spec = importlib.util.spec_from_file_location("iiLossNN_preprint", tmp.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.unlink(tmp.name)
    return module.iiLossNN


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="main", help="git revision of the preprint's model")
    parser.add_argument("--n", type=int, default=4000, help="compounds in the test set")
    args = parser.parse_args()

    import numpy as np
    import torch
    torch.set_num_threads(1)
    from models.iiLossNN import iiLossNN
    Preprint = load_reference(args.ref)

    with open(os.path.join(REPO_ROOT, "data", "2025", "hullout_2025.json")) as f:
        hull = json.load(f)
    data = {k: hull[k] for k in sorted(hull)[:args.n]}
    common = dict(hidden=(64, 32, 16, 8), seed=3, device="cpu")

    with tempfile.TemporaryDirectory() as ckpt:
        def fit(cls, **kw):
            model = cls("Hf", **common, **kw)
            X, y, _ = model.preprocess(data)
            test = np.arange(0, len(y), 7)
            train = np.setdiff1d(np.arange(len(y)), test)
            model.fit(X[train], y[train])
            return np.array(model.predict(X[test]))

        stage2 = dict(finetune_from=ckpt, finetune_lr=1e-4, finetune_epochs=15,
                      warmup_frac=0.0, renorm_every=0)
        checks = [
            ("stage one", dict(lam=0.0, epochs=20), dict(lam=0.0, epochs=20)),
            ("lam = 0, default switches", dict(lam=0.0, **stage2), dict(lam=0.0, **stage2)),
            ("lam = 0.3, both switches off", dict(lam=0.3, **stage2),
             dict(lam=0.3, project=False, freeze_den=False, **stage2)),
            ("lam = 0.3, both off, PCGrad", dict(lam=0.3, use_pcgrad=True, **stage2),
             dict(lam=0.3, project=False, freeze_den=False, use_pcgrad=True, **stage2)),
        ]
        fit(Preprint, lam=0.0, epochs=20, checkpoint_dir=ckpt)
        failed = 0
        for label, ref_kw, new_kw in checks:
            diff = np.abs(fit(Preprint, **ref_kw) - fit(iiLossNN, **new_kw)).max()
            ok = diff == 0.0
            failed += not ok
            print(f"  {label:32s} max |difference| = {diff:.1e}  {'OK' if ok else 'FAILED'}")
    sys.exit(failed)


if __name__ == "__main__":
    main()
