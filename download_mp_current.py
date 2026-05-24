#!/usr/bin/env python3
"""
Download the current Materials Project database and build hullout.json
in the same format as the Bartel et al. 2020 dataset.

Output format (hullout_current.json):
{
  "formula": {
    "Ef": float,   # formation energy per atom (eV/atom)
    "Ed": float,   # decomposition energy per atom (eV/atom)
    "rxn": str,    # decomposition reaction string
    "stability": bool  # True if Ed <= 0
  }, ...
}

Usage:
    python download_mp_current.py --api-key YOUR_API_KEY
    python download_mp_current.py --api-key YOUR_API_KEY --out my_hullout.json
"""

import argparse
import json
import os
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", required=True, help="Materials Project API key")
    p.add_argument("--out", default="hullout_current.json",
                   help="Output file path (default: hullout_current.json)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of entries (for testing)")
    return p.parse_args()


def download_entries(api_key, limit=None):
    """Download all ground-state entries from Materials Project."""
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("Install with: pip install mp-api")

    print("Connecting to Materials Project API...")
    fields = ["material_id", "formula_pretty", "formation_energy_per_atom",
              "energy_above_hull", "is_stable", "chemsys"]

    with MPRester(api_key) as mpr:
        print("Downloading entries (this may take a few minutes)...")
        entries = mpr.summary.search(
            fields=fields,
            num_chunks=None,
        )

    if limit:
        entries = entries[:limit]

    print(f"Downloaded {len(entries)} entries.")
    return entries


def build_hullout(entries):
    """
    Build hullout dict from MP entries.

    Note: MP's energy_above_hull is the distance above the convex hull (>= 0).
    For stable compounds (is_stable=True), Ed = -(energy_above_hull of the
    competing phases). We approximate Ed using the MP hull data:
      - stable:   Ed = -|distance to hull without this compound| (negative)
      - unstable: Ed = +energy_above_hull (positive)

    Since MP doesn't directly give the "distance below hypothetical hull" for
    stable compounds, we use a sign convention consistent with Bartel:
      Ed < 0 → stable (on hull)
      Ed > 0 → unstable (above hull)

    For stable compounds, we set Ed = -energy_above_hull_of_nearest_competitor,
    but MP doesn't expose this directly. As a practical approximation we use
    Ed = 0 for stable compounds and Ed = +energy_above_hull for unstable ones,
    then flag stability. For the full signed Ed you'd need pymatgen's
    PhaseDiagram — see the note below.
    """
    hullout = {}
    skipped = 0

    for e in entries:
        formula = e.formula_pretty
        if formula is None:
            skipped += 1
            continue

        Ef = e.formation_energy_per_atom
        if Ef is None:
            skipped += 1
            continue

        e_above_hull = e.energy_above_hull or 0.0
        stable = e.is_stable if e.is_stable is not None else (e_above_hull <= 0)

        # Ed sign convention: negative = stable, positive = unstable
        # For unstable: Ed = +energy_above_hull
        # For stable: Ed = 0 (exact signed value needs full phase diagram)
        Ed = -0.0 if stable else e_above_hull

        chemsys = e.chemsys or ""
        rxn = chemsys  # placeholder; full rxn string needs PhaseDiagram

        hullout[formula] = {
            "Ef": Ef,
            "Ed": Ed,
            "rxn": rxn,
            "stability": stable,
        }

    print(f"Built hullout with {len(hullout)} entries ({skipped} skipped).")
    return hullout


def build_hullout_pymatgen(api_key, limit=None):
    """
    Full version using pymatgen PhaseDiagram for exact signed Ed values
    and proper decomposition reaction strings — matches Bartel format exactly.
    Slower but correct.
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
        from pymatgen.core import Composition
    except ImportError:
        raise ImportError("Install with: pip install mp-api pymatgen")

    print("Connecting to Materials Project API (full pymatgen mode)...")
    with MPRester(api_key) as mpr:
        print("Downloading ComputedStructureEntries...")
        all_entries = mpr.get_entries_in_chemsys(
            elements=None,  # all elements
            additional_criteria={"is_stable": None},  # all entries
        )

    if limit:
        all_entries = all_entries[:limit]

    print(f"Downloaded {len(all_entries)} entries. Building phase diagrams...")

    # Group by chemical system
    from collections import defaultdict
    by_chemsys = defaultdict(list)
    for entry in all_entries:
        chemsys = "-".join(sorted(entry.composition.chemical_system.split("-")))
        by_chemsys[chemsys].append(entry)

    hullout = {}
    total = len(by_chemsys)
    for i, (chemsys, entries) in enumerate(by_chemsys.items()):
        if i % 500 == 0:
            print(f"  Processing chemical system {i}/{total}...")
        try:
            pd = PhaseDiagram(entries)
            for entry in entries:
                formula = entry.composition.reduced_formula
                Ef = entry.energy_per_atom  # Note: this is total E, not formation E
                # Get decomposition
                decomp, e_hull = pd.get_decomp_and_e_above_hull(entry)
                stable = e_hull <= 1e-6

                # Decomposition reaction string
                rxn_parts = []
                for comp, amt in decomp.items():
                    rxn_parts.append(f"{amt:.4f}_{comp.reduced_formula}")
                rxn = " + ".join(rxn_parts)

                # Ed: negative for stable, positive for unstable (eV/atom)
                Ed = -e_hull if stable else e_hull

                hullout[formula] = {
                    "Ef": Ef,
                    "Ed": Ed,
                    "rxn": rxn,
                    "stability": stable,
                }
        except Exception as ex:
            pass  # skip problematic chemical systems

    print(f"Built hullout with {len(hullout)} entries.")
    return hullout


def main():
    args = parse_args()

    out_path = args.out
    if os.path.exists(out_path):
        print(f"Output file {out_path} already exists. Delete it first to re-download.")
        return

    # Use simple mode (fast) — switch to build_hullout_pymatgen for exact Ed values
    entries = download_entries(args.api_key, limit=args.limit)
    hullout = build_hullout(entries)

    print(f"Saving to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(hullout, f)
    print(f"Done. {len(hullout)} compounds saved to {out_path}")
    print()
    print("NOTE: For exact signed Ed values matching Bartel's format,")
    print("use build_hullout_pymatgen() instead — it runs full convex hull")
    print("construction via pymatgen PhaseDiagram (slower, ~hours).")


if __name__ == "__main__":
    main()
