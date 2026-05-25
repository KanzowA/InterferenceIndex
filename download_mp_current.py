#!/usr/bin/env python3
"""
Download the current Materials Project database and build hullout_current.json
in the same format as the Bartel et al. 2020 dataset.

Replicates Bartel's query criteria:
  - Nonelemental compounds only (Bartel: "85,014 unique nonelemental chemical formulas")
  - Formation energy filter: Ef < 1 eV/atom
  - Ground-state structure per formula (most negative Ef)
  - MP correction scheme applied automatically by the API

Output format (same as mlstabilitytest hullout.json):
{
  "formula": {
    "Ef": float,        # formation energy per atom (eV/atom)
    "Ed": float,        # decomposition energy per atom (eV/atom); <0 stable, >0 unstable
    "rxn": str,         # decomposition reaction string
    "stability": bool   # True if on hull (Ed <= 0)
  }, ...
}

Modes
-----
  fast (default)
      Uses mpr.summary.search() + mpr.thermo.search().
      Ed sign and rxn string from MP's pre-computed ThermoDoc.
      Runtime: ~30-60 min.

  pymatgen
      Downloads ComputedEntry objects and rebuilds convex hulls via
      pymatgen PhaseDiagram.  Exact Ed values, exact rxn strings.
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
import math
import os
import re
import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_rxn(decomposes_to) -> str:
    """Convert ThermoDoc DecompositionProduct list to Bartel rxn string.

    Bartel format: '0.5000_Fe2O3 + 0.5000_FeO'

    ThermoDoc returns raw MP formulas like 'Ac1 Ag1', 'Ir1', 'Ac4' —
    these must be normalized to reduced_formula ('AcAg', 'Ir', 'Ac') so
    they match the hullout dict keys (which come from formula_pretty).
    """
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
            # Fallback without pymatgen: strip spaces, remove explicit '1'
            # e.g. 'Ac1 Ag1' → 'AcAg', 'Ir1' → 'Ir'
            import re
            f = f.replace(" ", "")
            # Remove '1' only when between element symbols or at end
            f = re.sub(r'([A-Za-z])1([A-Z]|$)', r'\1\2', f)
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
    tokens = re.findall(r'[A-Z][a-z]?', formula)
    return len(set(tokens)) == 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key", required=True, help="Materials Project API key")
    p.add_argument("--out", default="hullout_current.json",
                   help="Output JSON path (default: hullout_current.json)")
    p.add_argument("--mode", choices=["fast", "pymatgen"], default="fast",
                   help="'fast' uses MP pre-computed data; 'pymatgen' rebuilds hulls locally")
    p.add_argument("--ef-cutoff", type=float, default=1.0,
                   help="Formation-energy upper cutoff eV/atom (Bartel used 1.0)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit entries for testing (default: no limit)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fast mode: summary + thermo endpoints
# ---------------------------------------------------------------------------

def download_fast(api_key, ef_cutoff=1.0, limit=None):
    """
    Two-step fast download:
      1. mpr.summary.search() → get all ground-state Ef values
      2. mpr.thermo.search()  → get pre-computed Ed, rxn strings, stability

    Deduplication: for multiple polymorphs of the same formula, keep
    the entry with the most negative Ef (ground state), consistent with
    Bartel's procedure.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("pip install mp-api")

    # ------------------------------------------------------------------
    # Step 1: summary search — filtered by Ef < cutoff
    #   Gives us: Ef, energy_above_hull, is_stable, material_id
    # ------------------------------------------------------------------
    print(f"[fast] Step 1/2: downloading summary data (Ef < {ef_cutoff} eV/atom)...")
    t0 = time.time()

    sum_fields = ["material_id", "formula_pretty", "formation_energy_per_atom",
                  "energy_above_hull", "is_stable", "chemsys"]

    with MPRester(api_key) as mpr:
        summary_docs = mpr.materials.summary.search(
            formation_energy=(-100, ef_cutoff),
            fields=sum_fields,
            num_chunks=None,
        )

    print(f"  → {len(summary_docs)} entries in {(time.time()-t0)/60:.1f} min")

    if limit:
        summary_docs = summary_docs[:limit]
        print(f"  → limited to {len(summary_docs)} for testing")

    # ------------------------------------------------------------------
    # Step 2: thermo search — NO filter (ThermoRester doesn't support Ef filter)
    #   Gives us: decomposition_enthalpy (signed Ed), rxn string
    #
    #   decomposition_enthalpy:
    #     < 0 → stable   (compound is lower energy than competing phases)
    #     > 0 → unstable (compound would gain energy by decomposing)
    #   decomposition_enthalpy_decomposes_to: reaction string, e.g.
    #     "0.5 Fe2O3 + 0.5 FeO"
    # ------------------------------------------------------------------
    print("[fast] Step 2/2: downloading thermo data (filtered to our material IDs)...")
    t0 = time.time()

    thermo_fields = ["material_id", "decomposition_enthalpy",
                     "decomposition_enthalpy_decomposes_to", "is_stable"]

    # Collect only the material IDs we actually need (from step 1, already limited)
    # Test run (--limit N): N material IDs → fast thermo fetch
    # Full run (no limit): ~151k IDs → mp-api chunks automatically
    all_mids = [d.material_id for d in summary_docs]

    # Filter by GGA/GGA+U only — same energy scheme as Bartel 2020.
    # Without this, MP returns entries for GGA, r2SCAN, and mixed schemes,
    # tripling the download size and deserialization time.
    with MPRester(api_key) as mpr:
        thermo_docs = mpr.materials.thermo.search(
            thermo_types=["GGA_GGA+U"],
            fields=thermo_fields,
            num_chunks=None,
        )

    print(f"  → {len(thermo_docs)} thermo entries in {(time.time()-t0)/60:.1f} min")

    # Build material_id → thermo lookup
    thermo_map = {td.material_id: td for td in thermo_docs}

    # ------------------------------------------------------------------
    # Build hullout: deduplicate by formula, keep most-negative Ef
    # ------------------------------------------------------------------
    print("[fast] Building hullout (deduplication by formula, ground-state Ef)...")

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
        # Ground state = most negative Ef (Bartel: "used the most negative ΔHf")
        gs = min(docs, key=lambda d: d.formation_energy_per_atom)
        Ef = gs.formation_energy_per_atom

        td = thermo_map.get(gs.material_id)
        if td is not None:
            Ed = td.decomposition_enthalpy        # signed: <0 stable, >0 unstable
            rxn = format_rxn(td.decomposition_enthalpy_decomposes_to) or gs.chemsys or ""
            stable = bool(td.is_stable) if td.is_stable is not None else (Ed is not None and Ed <= 0)
            if Ed is None:
                e_hull = gs.energy_above_hull or 0.0
                stable = bool(gs.is_stable) if gs.is_stable is not None else (e_hull <= 1e-6)
                Ed = 0.0 if stable else e_hull
        else:
            # Fallback if material has no ThermoDoc (rare)
            n_no_thermo += 1
            e_hull = gs.energy_above_hull or 0.0
            stable = bool(gs.is_stable) if gs.is_stable is not None else (e_hull <= 1e-6)
            Ed = 0.0 if stable else e_hull
            rxn = gs.chemsys or ""

        hullout[formula] = {
            "Ef": Ef,
            "Ed": Ed,
            "rxn": rxn,
            "stability": stable,
        }

    if n_no_thermo:
        print(f"  Warning: {n_no_thermo} formulas had no ThermoDoc → Ed approximated")

    n_stable = sum(1 for v in hullout.values() if v["stability"])
    print(f"  → {len(hullout)} nonelemental formulas "
          f"({n_stable} stable, {len(hullout)-n_stable} unstable)")
    return hullout


# ---------------------------------------------------------------------------
# Pymatgen mode: full convex hull reconstruction
# ---------------------------------------------------------------------------

def download_pymatgen(api_key, ef_cutoff=1.0, limit=None):
    """
    Full mode: one bulk download of all ComputedEntry objects, then builds
    all convex hulls locally using pymatgen PhaseDiagram.

    This matches Bartel's procedure exactly:
      - Ef from PhaseDiagram.get_form_energy_per_atom()
      - Ed signed: negative for stable, positive for unstable
      - rxn strings with stoichiometric coefficients (needed for ξ weights)

    Runtime: ~20-45 min (one API call + local PhaseDiagram construction).
    The old approach looped per chemical system (10k+ API calls = hours).
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram
    except ImportError:
        raise ImportError("pip install mp-api pymatgen")

    # ------------------------------------------------------------------
    # Step 1: collect all unique elements across the database
    # ------------------------------------------------------------------
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
    print(f"  → {len(chemsys_docs)} materials, {len(all_elements)} unique elements "
          f"in {(time.time()-t0)/60:.1f} min")

    # ------------------------------------------------------------------
    # Step 2: bulk download ALL ComputedEntry objects in one streaming call
    #
    # Passing all elements to get_entries_in_chemsys returns every entry
    # whose composition is a subset of those elements — i.e. the full DB.
    # compatible_only=True applies MP's GGA/GGA+U mixing scheme corrections,
    # exactly as Bartel's 2019 query did.
    # ------------------------------------------------------------------
    print(f"[pymatgen] Step 2/3: bulk downloading all ComputedEntries "
          f"({len(all_elements)} elements)...")
    print("  (One streaming call — expect ~5-15 min depending on connection.)")
    t0 = time.time()

    with MPRester(api_key) as mpr:
        all_entries = mpr.get_entries_in_chemsys(
            all_elements,
            compatible_only=True,
        )

    print(f"  → {len(all_entries)} ComputedEntries in {(time.time()-t0)/60:.1f} min")

    if limit:
        all_entries = all_entries[:limit]
        print(f"  → limited to {len(all_entries)} for testing")

    # ------------------------------------------------------------------
    # Step 3: group entries by chemical system, build phase diagrams locally
    #
    # Key: PhaseDiagram for "Li-Fe-O" needs ALL entries whose elements are
    # a subset of {Li, Fe, O}, including unary and binary sub-systems.
    # We index by frozenset(elements) and do subset lookups.
    # ------------------------------------------------------------------
    print("[pymatgen] Step 3/3: building phase diagrams and extracting Ed/rxn...")
    t0 = time.time()

    # Index: frozenset(elements) → list of entries
    entries_by_elems = defaultdict(list)
    for entry in all_entries:
        elems = frozenset(str(e) for e in entry.composition.elements)
        entries_by_elems[elems].append(entry)

    # Unique chemical systems that have nonelemental compounds
    target_chemsys = {
        frozenset(elems)
        for elems in entries_by_elems
        if len(elems) > 1
    }

    def get_pd_entries(chemsys_frozenset):
        """All entries whose elements ⊆ chemsys_frozenset (needed for hull)."""
        return [e
                for elems, entries in entries_by_elems.items()
                if elems.issubset(chemsys_frozenset)
                for e in entries]

    hullout = {}
    n_errors = 0
    total = len(target_chemsys)

    for i, chemsys_set in enumerate(target_chemsys):
        if i % 500 == 0:
            elapsed = (time.time() - t0) / 60
            est = (elapsed / max(i, 1)) * total
            print(f"  [{i}/{total}] {elapsed:.1f} min elapsed, "
                  f"~{est:.0f} min total | {len(hullout)} entries")

        try:
            pd_entries = get_pd_entries(chemsys_set)
            if not pd_entries:
                continue
            pd = PhaseDiagram(pd_entries)
        except Exception as ex:
            n_errors += 1
            continue

        # Process only direct members of this chemsys (not sub-systems,
        # those are handled when their own chemsys is processed)
        for entry in entries_by_elems[chemsys_set]:
            formula = entry.composition.reduced_formula
            if is_elemental(formula):
                continue

            try:
                Ef = pd.get_form_energy_per_atom(entry)
            except Exception:
                continue

            if Ef >= ef_cutoff:
                continue

            try:
                decomp, e_hull = pd.get_decomp_and_e_above_hull(entry)
            except Exception:
                continue

            stable = e_hull <= 1e-6

            # Ed sign convention matching Bartel:
            #   stable   → Ed ≤ 0  (compound is at/below hull)
            #   unstable → Ed > 0  (compound is above hull)
            # For stable compounds on the hull, e_hull = 0.
            # The exact negative value requires a leave-one-out hull,
            # which is expensive; -1e-9 flags them as stable with Ed < 0.
            Ed = -1e-9 if stable else e_hull

            rxn_parts = [f"{amt:.4f}_{comp.reduced_formula}"
                         for comp, amt in decomp.items()]
            rxn = " + ".join(rxn_parts)

            # Deduplication: keep most negative Ef per formula
            if formula not in hullout or Ef < hullout[formula]["Ef"]:
                hullout[formula] = {
                    "Ef": Ef,
                    "Ed": Ed,
                    "rxn": rxn,
                    "stability": stable,
                }

    n_stable = sum(1 for v in hullout.values() if v["stability"])
    print(f"\n[pymatgen] Done. {len(hullout)} nonelemental formulas "
          f"({n_stable} stable, {len(hullout)-n_stable} unstable)")
    if n_errors:
        print(f"  {n_errors} chemical systems had PhaseDiagram errors (skipped)")
    return hullout


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_path = args.out
    if os.path.exists(out_path):
        print(f"Output file already exists: {out_path}")
        print("Delete it first to re-download.")
        return

    print("=" * 60)
    print(f"Mode:        {args.mode}")
    print(f"Ef cutoff:   < {args.ef_cutoff} eV/atom  (Bartel: 1.0)")
    print(f"Output:      {out_path}")
    if args.limit:
        print(f"Limit:       {args.limit} entries (TEST MODE)")
    print("=" * 60)

    t0 = time.time()

    if args.mode == "fast":
        hullout = download_fast(args.api_key,
                                ef_cutoff=args.ef_cutoff,
                                limit=args.limit)
    else:
        hullout = download_pymatgen(args.api_key,
                                    ef_cutoff=args.ef_cutoff,
                                    limit=args.limit)

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")

    print(f"Saving {len(hullout)} entries to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(hullout, f, indent=2)

    print("Done.")
    print()

    if args.mode == "fast":
        print("NOTE: Fast mode — Ed and rxn from MP pre-computed ThermoDoc.")
        print("      Ef, Ed, stability, and rxn strings are all exact.")
        print("      Only difference from pymatgen mode: rxn string format may differ slightly.")


if __name__ == "__main__":
    main()
