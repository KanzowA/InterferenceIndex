"""Classify the run directories under data/<year>/ml/<target>.

Every script that produces a figure or table requires the _s<seed> suffix, so
runs without one are not used by any reported result. This lists what is
present, what each group is for, and how much space it occupies, so that the
legacy generation can be removed deliberately rather than by guesswork.

Nothing is deleted. Use --legacy to emit a plain list of paths for xargs.

Usage
-----
    python scripts/audit_runs.py
    python scripts/audit_runs.py --legacy
    python scripts/audit_runs.py --legacy | xargs -r rm -rf
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Each pattern is matched against the run name with the _s<seed> suffix
# already removed.
KINDS = [
    (re.compile(r"^iiLoss_save_0\.0(_w\d+)?$"),
     "stage 1 checkpoint source"),
    (re.compile(r"^(ii|Hd)Loss_0\.0(_w\d+)?$"),
     "stage 1 baseline, not fine-tuned"),
    (re.compile(r"^(ii|Hd)Loss_finetune_\d+(\.\d+)?$"),
     "main sweep"),
    (re.compile(r"^(ii|Hd)Loss_pcgradc?_\d+(\.\d+)?$"),
     "gradient surgery"),
    (re.compile(r"^(ii|Hd)Loss_finetune_\d+(\.\d+)?_w\d+$"),
     "capacity ladder"),
    (re.compile(r"^iiLoss_finetune_eps[\d.e-]+_\d+(\.\d+)?$"),
     "epsilon sensitivity"),
    (re.compile(r"^(ii|Hd)Loss_finetune_conv\d+_\d+(\.\d+)?$"),
     "convergence diagnostic"),
]


def size_of(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def human(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def classify(name):
    """Return (kind, seeded). Kind is None if no pattern matches."""
    stem, _, tail = name.rpartition("_s")
    seeded = bool(stem) and tail.isdigit()
    key = stem if seeded else name
    for pattern, label in KINDS:
        if pattern.match(key):
            return label, seeded
    return None, seeded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2026")
    parser.add_argument("--target", default="Hf")
    parser.add_argument("--legacy", action="store_true",
                        help="print unseeded run paths only, one per line")
    args = parser.parse_args()

    base = os.path.join(REPO_ROOT, "data", args.year, "ml", args.target)
    if not os.path.isdir(base):
        raise SystemExit(f"{base} not found")

    groups = {}
    legacy = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        kind, seeded = classify(name)
        label = kind or "unrecognised"
        key = (label, seeded)
        entry = groups.setdefault(key, {"n": 0, "bytes": 0, "example": name})
        entry["n"] += 1
        entry["bytes"] += size_of(path)
        if not seeded:
            legacy.append(path)

    if args.legacy:
        for path in legacy:
            print(path)
        return

    print(f"{base}\n")
    print(f"  {'kind':<26}{'seeded':>8}{'runs':>7}{'size':>9}   example")
    for (label, seeded), e in sorted(groups.items()):
        print(f"  {label:<26}{'yes' if seeded else 'NO':>8}{e['n']:>7}"
              f"{human(e['bytes']):>9}   {e['example']}")

    used = sum(e["bytes"] for (l, s), e in groups.items() if s)
    unused = sum(e["bytes"] for (l, s), e in groups.items() if not s)
    print(f"\n  seeded (used by the paper):   {human(used)}")
    print(f"  unseeded (legacy, unused):    {human(unused)}")
    if legacy:
        print(f"\n  remove with:  python {os.path.relpath(__file__, REPO_ROOT)}"
              f" --legacy | xargs -r rm -rf")


if __name__ == "__main__":
    main()
