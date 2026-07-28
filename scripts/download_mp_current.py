"""Build a hullout file from the current Materials Project database.

Keeps non-elemental compounds with Hf < 1 eV/atom, one ground-state entry per
formula. MP corrections are applied by the API.

Output matches the mlstabilitytest hullout format: per formula, the formation
enthalpy Hf and decomposition enthalpy Hd in eV/atom, the decomposition
reaction string, and a stability flag (Hd <= 0).

Modes
-----
  fast (default)
      Uses mpr.summary.search() + mpr.thermo.search().
      Hd sign and rxn string from MP's pre-computed ThermoDoc.
      Runtime: ~30-60 min.

  pymatgen
      Downloads ComputedEntry objects and rebuilds convex hulls via
      pymatgen PhaseDiagram.  Exact Hd values, exact rxn strings.
      Matches Bartel's procedure most closely.
      Runtime: ~4-8 hours.

Usage
-----
    python download_mp_current.py --api-key YOUR_KEY
    python download_mp_current.py --api-key YOUR_KEY --mode pymatgen
    python download_mp_current.py --api-key YOUR_KEY --out my_file.json
    python download_mp_current.py --api-key YOUR_KEY --limit 500   # quick test
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

def format_rxn(decomposes_to) -> str:
    """Convert ThermoDoc DecompositionProduct list to rxn string."""

    if not decomposes_to:
        return ""
    if isinstance(decomposes_to, str):
        return decomposes_to

    try:
        from pymatgen.core import Composition
        def normalize(f):
            return Composition(f).reduced_formula
    except ImportError:
        def normalize(f):
            f = f.replace(" ", "")
            f = re.sub(r"([A-Za-z])1([A-Z]|$)", r"\1\2", f) # Remove '1' only when between element symbols or at end
            return f

    parts = []
    for p in decomposes_to:
        amt = getattr(p, "amount", getattr(p, "amt", 1.0))
        raw_formula = getattr(p, "formula", getattr(p, "name", str(p)))
        parts.append(f"{amt:.4f}_{normalize(raw_formula)}")
    return " + ".join(parts)


def is_elemental(formula: str) -> bool:
    """Return True if formula contains only one distinct element."""
    # Strip numbers/spaces; if only one token remains -> elemental
    tokens = re.findall(r"[A-Z][a-z]?", formula)
    return len(set(tokens)) == 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key", required=True, help="Materials Project API key")
    p.add_argument("--out", default="hullout_current.json",
                   help="Output JSON path (default: hullout_current.json)")
    p.add_argument("--mode", choices=["fast", "pymatgen"], default="fast",
                   help="'fast' uses MP pre-computed data; 'pymatgen' rebuilds hulls locally")
    p.add_argument("--hf-cutoff", type=float, default=1.0,
                   help="Formation-enthalpy upper cutoff eV/atom (Bartel used 1.0)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit entries for testing (default: no limit)")
    return p.parse_args()


# -- Fast mode ------------------------------------------------------------

def download_fast(api_key, hf_cutoff=1.0, limit=None):
    """Download Hf from the summary endpoint and Hd from the thermo endpoint.

    Where several polymorphs share a formula, the one with the most negative
    Hf is kept as the ground state.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("pip install mp-api")

    print(f"[fast] Step 1/2: downloading summary data (Hf < {hf_cutoff} eV/atom)...")
    t0 = time.time()

    sum_fields = ["material_id", "formula_pretty", "formation_energy_per_atom",
                  "energy_above_hull", "is_stable", "chemsys"]

    with MPRester(api_key) as mpr:
        summary_docs = mpr.materials.summary.search(
            formation_energy=(-100, hf_cutoff),
            fields=sum_fields,
            num_chunks=None,
        )

    print(f"  -> {len(summary_docs)} entries in {(time.time()-t0)/60:.1f} min")

    if limit:
        summary_docs = summary_docs[:limit]
        print(f"  -> limited to {len(summary_docs)} for testing")

    print("[fast] Step 2/2: downloading thermo data (filtered to our material IDs)...")
    t0 = time.time()

    thermo_fields = ["material_id", "decomposition_enthalpy",
                     "decomposition_enthalpy_decomposes_to", "is_stable"]

    with MPRester(api_key) as mpr:
        thermo_docs = mpr.materials.thermo.search(
            thermo_types=["GGA_GGA+U"], # Filter by GGA/GGA+U only
            fields=thermo_fields,
            num_chunks=None,
        )

    print(f"  -> {len(thermo_docs)} thermo entries in {(time.time()-t0)/60:.1f} min")

    thermo_map = {td.material_id: td for td in thermo_docs}

    print("[fast] Building hullout (deduplication by formula, ground-state Hf)...")

    by_formula = defaultdict(list)
    for doc in summary_docs:
        formula = doc.formula_pretty
        if formula is None or is_elemental(formula):
            continue
        if doc.formation_energy_per_atom is None:
            continue
        by_formula[formula].append(doc)

    hullout = {}
    n_no_thermo = 0

    for formula, docs in by_formula.items():
        # Ground state = most negative Hf
        gs = min(docs, key=lambda d: d.formation_energy_per_atom)
        Ef = gs.formation_energy_per_atom

        td = thermo_map.get(gs.material_id)
        if td is not None:
            Ed = td.decomposition_enthalpy
            rxn = format_rxn(td.decomposition_enthalpy_decomposes_to) or gs.chemsys or ""
            stable = bool(td.is_stable) if td.is_stable is not None else (Ed is not None and Ed <= 0)
            if Ed is None:
                e_hull = gs.energy_above_hull or 0.0
                stable = bool(gs.is_stable) if gs.is_stable is not None else (e_hull <= 1e-6)
                Ed = 0.0 if stable else e_hull
        else:
            n_no_thermo += 1
            e_hull = gs.energy_above_hull or 0.0
            stable = bool(gs.is_stable) if gs.is_stable is not None else (e_hull <= 1e-6)
            Ed = 0.0 if stable else e_hull
            rxn = gs.chemsys or ""

        hullout[formula] = {
            "Hf": Ef,
            "Hd": Ed,
            "rxn": rxn,
            "stability": stable,
        }

    if n_no_thermo:
        print(f"  Warning: {n_no_thermo} formulas had no ThermoDoc -> Ed approximated")

    n_stable = sum(1 for v in hullout.values() if v["stability"])
    print(f"  -> {len(hullout)} nonelemental formulas "
          f"({n_stable} stable, {len(hullout)-n_stable} unstable)")
    return hullout


# Pymatgen mode

def download_pymatgen(api_key, hf_cutoff=1.0, limit=None):
    """Rebuild the convex hulls with pymatgen for exact Hd and rxn strings.

    Hf comes from PhaseDiagram.get_form_energy_per_atom(); Hd is signed
    negative for stable and positive for unstable; the rxn strings carry the
    stoichiometric coefficients the xi weights need.
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram
    except ImportError:
        raise ImportError("pip install mp-api pymatgen")

    print("[pymatgen] Step 1/3: collecting chemical systems...")
    t0 = time.time()

    with MPRester(api_key) as mpr:
        chemsys_docs = mpr.materials.summary.search(
            fields=["chemsys"],
            num_chunks=None,
        )

    all_elements = sorted({el
                           for doc in chemsys_docs
                           for el in doc.chemsys.split("-")})
    print(f"  -> {len(chemsys_docs)} materials, {len(all_elements)} unique elements "
          f"in {(time.time()-t0)/60:.1f} min")

    print(f"[pymatgen] Step 2/3: bulk downloading all ComputedEntries "
          f"({len(all_elements)} elements)...")

    t0 = time.time()

    with MPRester(api_key) as mpr:
        all_entries = mpr.get_entries_in_chemsys(
            all_elements,
            compatible_only=True,
        )

    print(f"  -> {len(all_entries)} ComputedEntries in {(time.time()-t0)/60:.1f} min")

    if limit:
        all_entries = all_entries[:limit]
        print(f"  -> limited to {len(all_entries)} for testing")

    print("[pymatgen] Step 3/3: building phase diagrams and extracting Ed/rxn...")
    t0 = time.time()

    entries_by_elems = defaultdict(list)
    for entry in all_entries:
        elems = frozenset(str(e) for e in entry.composition.elements)
        entries_by_elems[elems].append(entry)

    target_chemsys = {
        frozenset(elems)
        for elems in entries_by_elems
        if len(elems) > 1
    }

    def get_pd_entries(chemsys_frozenset):
        """All entries whose elements are a subset of chemsys_frozenset (needed for hull)."""
        return [e
                for elems, entries in entries_by_elems.items()
                if elems.issubset(chemsys_frozenset)
                for e in entries]

    hullout = {}
    n_errors = 0
    total = len(target_chemsys)

    for i, chemsys_set in enumerate(target_chemsys):
        if i % 500 == 0:
            elapsed_min = (time.time() - t0) / 60
            est_min = (elapsed_min / max(i, 1)) * total
            print(f"  [{i}/{total}] {elapsed_min:.1f} min elapsed, "
                  f"~{est_min:.0f} min total | {len(hullout)} entries")

        try:
            pd_entries = get_pd_entries(chemsys_set)
            if not pd_entries:
                continue
            pd = PhaseDiagram(pd_entries)
        except Exception:
            n_errors += 1
            continue

        for entry in entries_by_elems[chemsys_set]:
            formula = entry.composition.reduced_formula
            if is_elemental(formula):
                continue

            try:
                Ef = pd.get_form_energy_per_atom(entry)
            except Exception:
                continue

            if Ef >= hf_cutoff:
                continue

            try:
                decomp, e_hull = pd.get_decomp_and_e_above_hull(entry)
            except Exception:
                continue

            stable = e_hull <= 1e-6

            Ed = -1e-9 if stable else e_hull

            rxn_parts = [f"{amt:.4f}_{comp.reduced_formula}"
                         for comp, amt in decomp.items()]
            rxn = " + ".join(rxn_parts)

            # Deduplication: keep most negative Hf per formula
            if formula not in hullout or Ef < hullout[formula]["Hf"]:
                hullout[formula] = {
                    "Hf": Ef,
                    "Hd": Ed,
                    "rxn": rxn,
                    "stability": stable,
                }

    n_stable = sum(1 for v in hullout.values() if v["stability"])
    print(f"\n[pymatgen] Done. {len(hullout)} nonelemental formulas "
          f"({n_stable} stable, {len(hullout)-n_stable} unstable)")
    if n_errors:
        print(f"  {n_errors} chemical systems had PhaseDiagram errors (skipped)")
    return hullout


# Main

def main():
    args = parse_args()

    out_path = args.out
    if os.path.exists(out_path):
        print(f"Output file already exists: {out_path}")
        print("Delete it first to re-download.")
        return

    print("=" * 60)
    print(f"Mode:        {args.mode}")
    print(f"Ef cutoff:   < {args.hf_cutoff} eV/atom  (Bartel: 1.0)")
    print(f"Output:      {out_path}")
    if args.limit:
        print(f"Limit:       {args.limit} entries (TEST MODE)")
    print("=" * 60)

    t0 = time.time()

    if args.mode == "fast":
        hullout = download_fast(args.api_key,
                                hf_cutoff=args.hf_cutoff,
                                limit=args.limit)
    else:
        hullout = download_pymatgen(args.api_key,
                                    hf_cutoff=args.hf_cutoff,
                                    limit=args.limit)

    elapsed_min = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed_min:.1f} min")

    print(f"Saving {len(hullout)} entries to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(hullout, f, indent=2)

    print("Done.")
    print()



if __name__ == "__main__":
    main()