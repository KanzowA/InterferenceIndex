"""
patch_material_ids.py
---------------------
Adds material_id to an existing hullout JSON without re-downloading everything.

Usage:
    python patch_material_ids.py --api-key YOUR_KEY
    python patch_material_ids.py --api-key YOUR_KEY --hullout hullout_current_v2.json
"""

import argparse
import json
import time
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", required=True)
    p.add_argument("--hullout", default="hullout_current_v2.json")
    p.add_argument("--batch-size", type=int, default=1000)
    return p.parse_args()

def main():
    args = parse_args()

    print(f"Loading {args.hullout}...")
    with open(args.hullout) as f:
        hull = json.load(f)

    # Only patch formulas that are missing material_id
    to_patch = [f for f, v in hull.items() if "material_id" not in v]
    print(f"  {len(hull)} total formulas, {len(to_patch)} need material_id")

    if not to_patch:
        print("Already complete — nothing to do.")
        return

    from mp_api.client import MPRester

    batches = [to_patch[i:i+args.batch_size]
               for i in range(0, len(to_patch), args.batch_size)]
    n_found = 0
    t0 = time.time()

    for b_idx, batch in enumerate(batches):
        print(f"  Batch {b_idx+1}/{len(batches)} ({len(batch)} formulas)...")

        with MPRester(args.api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=batch,
                fields=["material_id", "formula_pretty",
                        "formation_energy_per_atom"],
                num_chunks=None,
            )

        # Group by formula, pick ground state (most negative Ef)
        by_formula = defaultdict(list)
        for doc in docs:
            if doc.formation_energy_per_atom is not None:
                by_formula[doc.formula_pretty].append(
                    (doc.formation_energy_per_atom, doc.material_id))

        for formula, entries in by_formula.items():
            if formula in hull:
                _, mid = min(entries, key=lambda x: x[0])
                hull[formula]["material_id"] = mid
                n_found += 1

        # Save progress after each batch
        with open(args.hullout, "w") as f:
            json.dump(hull, f)

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min — {n_found} material_ids added.")
    print(f"Saved to {args.hullout}")

if __name__ == "__main__":
    main()
