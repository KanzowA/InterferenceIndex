"""Build the hullout files of the revision from published DFT energies.

Usage
-----
    python scripts/build_hull.py mp  data/2025/2025-02-01-mp-energies.csv.gz
    python scripts/build_hull.py wbm data/wbm/2023-12-13-wbm-summary.csv.gz

mp   Matbench Discovery's MP energies (thermo type GGA/GGA+U, MP2020-corrected,
     the convention of the WBM test set) -> data/2025/hullout_2025.json.
     As for the 2020 and 2026 snapshots: compositions with dfH < 1 eV/atom, the
     ground state per reduced formula, elements as references at 0 that are not
     themselves entries.
wbm  The WBM discovery set (unique prototypes, lowest energy per composition)
     -> data/wbm/hullout_wbm.json, a closed hull in which every reaction
     participant is a WBM entry. Each entry keeps its material_id, so the
     Matbench Discovery predictions, which are keyed by it, can be scored on it.

Every compound gets its decomposition reaction into the hull of all other
phases, solved exactly by LP (spectral.HullSolver), with per-atom weights as in
the 2026 file. Written per entry: Hf, Hd, rxn, stability, material_id.

The content hash printed at the end is over (Hf, Hd, rxn) and identifies the
file the nested, capacity and prospective runs were trained on.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from spectral import HullSolver, parse_formula

OUT = {"mp":  os.path.join(REPO_ROOT, "data", "2025", "hullout_2025.json"),
       "wbm": os.path.join(REPO_ROOT, "data", "wbm", "hullout_wbm.json")}

# The hullout_2025.json every revision run was trained on.
REFERENCE_HASH = {"mp": "9ee82af58674dcc57bdd28fc035caaa8"}

_SOLVER, _ENERGIES = None, None


def content_hash(hull):
    canon = json.dumps({k: [round(v["Hf"], 8), round(v.get("Hd", 0), 8), v.get("rxn", "")]
                        for k, v in sorted(hull.items())}, sort_keys=True)
    return hashlib.md5(canon.encode()).hexdigest()


def load_mp(path):
    """Ground state per reduced formula, dfH < 1 eV/atom, compounds only."""
    from pymatgen.core import Composition
    # na_filter: the formula 'NaN' (sodium nitride) is not a missing value.
    df = pd.read_csv(path, na_filter=False)
    for col in ("formation_energy_per_atom", "decomposition_enthalpy"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    idcol = "material_id" if "material_id" in df else df.columns[0]
    fcol = "formula" if "formula" in df else "formula_pretty"
    df = df[df.formation_energy_per_atom < 1.0].copy()
    comps = [Composition(f) for f in df[fcol]]
    df["red"] = [c.reduced_formula for c in comps]
    df["nel"] = [len(c.elements) for c in comps]
    df = (df[df.nel > 1].sort_values("formation_energy_per_atom")
          .drop_duplicates("red").reset_index(drop=True))
    check = (df["decomposition_enthalpy"].to_numpy(float)
             if "decomposition_enthalpy" in df else None)
    return (list(df.red), df.formation_energy_per_atom.to_numpy(float),
            df[idcol].astype(str).tolist(), check)


def load_wbm(path):
    """Unique prototypes, lowest MP2020-corrected dfH per composition."""
    from pymatgen.core import Composition
    df = pd.read_csv(path)
    df = df[df.unique_prototype]
    df = df.assign(comp=df.formula.map(parse_formula))
    df = df[df.comp.map(len) > 1]
    df["key"] = df.comp.map(lambda c: tuple(sorted((k, round(v, 6)) for k, v in c.items())))
    df = (df.sort_values("e_form_per_atom_mp2020_corrected")
          .drop_duplicates("key", keep="first").reset_index(drop=True))
    names = [Composition(f).reduced_formula for f in df.formula]
    if len(set(names)) != len(names):
        raise SystemExit("two WBM compositions share a reduced formula")
    return (names, df.e_form_per_atom_mp2020_corrected.to_numpy(float),
            df.material_id.astype(str).tolist(), None)


def _init(solver, energies):
    global _SOLVER, _ENERGIES
    _SOLVER, _ENERGIES = solver, energies


def _decompose(i):
    return _SOLVER.decompose(_ENERGIES, i)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=["mp", "wbm"])
    parser.add_argument("source", help="MP energies or WBM summary CSV")
    parser.add_argument("--out", default=None)
    parser.add_argument("--procs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    out = args.out or OUT[args.which]

    names, hf, ids, check = (load_mp if args.which == "mp" else load_wbm)(args.source)
    solver = HullSolver(names)
    energies = np.r_[hf, np.zeros(solver.nE)]
    labels = names + solver.elements
    print(f"{len(names)} compounds, {solver.nE} elements")

    t0 = time.time()
    with Pool(args.procs, initializer=_init, initargs=(solver, energies)) as pool:
        results = pool.map(_decompose, range(len(names)), chunksize=500)

    hull, dev = {}, []
    for i, (prods, x, hd) in enumerate(results):
        entry = {"Hf": float(hf[i]), "material_id": ids[i]}
        if prods is not None:
            entry["Hd"] = float(hd)
            entry["stability"] = bool(hd <= 0)
            entry["rxn"] = " + ".join(f"{w:.6f}_{labels[j]}" for j, w in zip(prods, x))
            if check is not None:
                dev.append(abs(hd - check[i]))
        hull[names[i]] = entry

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(hull, f)

    n_rxn = sum("rxn" in v for v in hull.values())
    print(f"{n_rxn} reactions in {time.time() - t0:.0f} s -> {out}")
    if dev:
        dev = np.array(dev)
        print(f"  against MP's own decomposition_enthalpy: median |diff| "
              f"{np.median(dev):.1e} eV/atom, {np.mean(dev > 0.01):.3f} above 10 meV")
    digest = content_hash(hull)
    ref = REFERENCE_HASH.get(args.which)
    note = "" if ref is None else ("  (matches the revision runs)" if digest == ref
                                   else "  (DIFFERS from the file the revision runs used)")
    print(f"  content hash {digest}{note}")


if __name__ == "__main__":
    main()
