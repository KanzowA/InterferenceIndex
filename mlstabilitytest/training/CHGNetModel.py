"""
CHGNetModel.py — CHGNet wrapper for the stability-prediction pipeline.

Pre-trained CHGNet (Deng et al., Nature Machine Intelligence 2023) for
structure-based formation energy prediction. No training required — pure
inference. fit_and_predict ignores Xtrain/Ytrain entirely.

Install:
    pip install chgnet --break-system-packages

Structures are fetched from the MP API and cached under:
    <repo_root>/structures_cache/<material_id>.json
(same cache used by ALIGNNModel, so no re-fetching if ALIGNN already ran)

Set your API key:
    export MP_API_KEY="your-key-here"

Usage via train_models.py:
    python mlstabilitytest/train_models.py allMP_current_v2 Ef CHGNet

Note on CHGNet's 'e' output:
    CHGNet.predict_structure() returns {'e': eV/atom, ...}.  On MP-sourced
    structures this corresponds to the formation energy as stored in MP
    (the model is benchmarked against MP Ef with MAE ~0.05 eV/atom).
    If you see a systematic offset, subtract the mean residual on the
    training fold.
"""

import os
import json
import numpy as np
from os.path import dirname, abspath, join

# ── repo paths ────────────────────────────────────────────────────────────────
_HERE       = dirname(abspath(__file__))   # mlstabilitytest/training/
_BASE       = dirname(_HERE)               # mlstabilitytest/
_REPO       = dirname(_BASE)              # repo root
_STRUCT_DIR = join(_REPO, "structures_cache")


class CHGNetModel:
    """
    Pre-trained CHGNet for formation-energy prediction.

    Implements the preprocess / fit_and_predict interface used by
    process.py / train_models.py.  Training data is ignored — CHGNet
    is applied as a fixed pre-trained model.
    """

    model_type = 'chgnet'

    def __init__(self, target, structures_dir=_STRUCT_DIR):
        self.target         = target
        self.structures_dir = structures_dir
        self._structures    = None   # list of pymatgen Structures (indexed)

    # ── structure helpers (shared with ALIGNNModel) ───────────────────────────

    def _load_cached(self, material_id: str):
        from pymatgen.core import Structure
        path = os.path.join(self.structures_dir, f"{material_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                return Structure.from_dict(json.load(f))
        return None

    def _save_structure(self, material_id: str, structure):
        os.makedirs(self.structures_dir, exist_ok=True)
        path = os.path.join(self.structures_dir, f"{material_id}.json")
        with open(path, "w") as f:
            json.dump(structure.as_dict(), f)

    def _batch_fetch(self, material_ids: list) -> dict:
        import requests
        from pymatgen.core import Structure

        api_key  = os.environ.get("MP_API_KEY", "")
        base_url = "https://api.materialsproject.org/materials/core/"
        headers  = {"X-Api-Key": api_key}
        result   = {}
        chunk_size = 50

        for start in range(0, len(material_ids), chunk_size):
            chunk  = material_ids[start:start + chunk_size]
            params = {"material_ids": ",".join(chunk),
                      "_fields": "material_id,structure"}
            resp = requests.get(base_url, headers=headers,
                                params=params, timeout=120)
            if not resp.ok:
                print(f"  Warning: HTTP {resp.status_code} — {resp.text[:200]}")
                resp.raise_for_status()
            for item in resp.json().get("data", []):
                mid       = item["material_id"]
                structure = Structure.from_dict(item["structure"])
                self._save_structure(mid, structure)
                result[mid] = structure
            print(f"  [CHGNet] fetched {start + len(chunk)}/{len(material_ids)} structures")

        return result

    # ── pipeline interface ────────────────────────────────────────────────────

    def preprocess(self, input_data: dict):
        """
        Load / fetch structures for every entry that has a material_id.

        Returns
        -------
        indices : np.ndarray of int   — positions into self._structures
        targets : np.ndarray of float
        labels  : np.ndarray of str   — formula strings
        """
        entries     = []
        needs_fetch = []

        for formula, props in input_data.items():
            t_val = props.get(self.target)
            mid   = props.get("material_id")
            if t_val is None or mid is None:
                continue
            entries.append((formula, float(t_val), mid))
            if self._load_cached(mid) is None:
                needs_fetch.append(mid)

        if needs_fetch:
            print(f"  [CHGNet] fetching {len(needs_fetch)} structures from MP…")
            self._batch_fetch(list(set(needs_fetch)))

        structures, targets_list, labels_list = [], [], []
        for formula, t_val, mid in entries:
            structure = self._load_cached(mid)
            if structure is None:
                print(f"  [CHGNet] Warning: no structure for {formula} ({mid}), skipping")
                continue
            structures.append(structure)
            targets_list.append(t_val)
            labels_list.append(formula)

        self._structures = structures
        n = len(entries)
        print(f"  [CHGNet] preprocess done — {len(structures)}/{n} entries kept")

        indices = np.arange(len(structures), dtype=np.int64)
        return (indices,
                np.array(targets_list, dtype=np.float32),
                np.array(labels_list))

    def fit_and_predict(self, Xtrain, Ytrain, Xtest):
        """
        Ignore training data — run CHGNet inference on Xtest structures.

        Parameters
        ----------
        Xtrain, Xtest : array-like of int — indices into self._structures
        Ytrain        : array-like of float — ignored (pre-trained model)

        Returns
        -------
        preds : np.ndarray of float, shape (len(Xtest),)
        """
        import torch
        from chgnet.model import CHGNet

        chgnet = CHGNet.load()
        use_gpu = torch.cuda.is_available()
        print(f"  [CHGNet] pre-trained model loaded, "
              f"predicting {len(Xtest)} structures… (GPU={'yes' if use_gpu else 'no'})")

        n_oom = 0
        preds = []
        for k, idx in enumerate(Xtest):
            try:
                p = chgnet.predict_structure(self._structures[idx])
                preds.append(float(p['e']))
            except RuntimeError as exc:
                if "out of memory" in str(exc) and use_gpu:
                    # Free fragmented cache; spawn a fresh CPU-only model for this structure.
                    # Do NOT move the main model off GPU — that move itself would OOM.
                    torch.cuda.empty_cache()
                    try:
                        chgnet_cpu = CHGNet.load()
                        chgnet_cpu = chgnet_cpu.to("cpu")
                        p = chgnet_cpu.predict_structure(self._structures[idx])
                        preds.append(float(p['e']))
                        del chgnet_cpu
                        n_oom += 1
                    except Exception as exc2:
                        print(f"  [CHGNet] CPU fallback also failed at idx {idx}: {exc2}")
                        preds.append(float(np.mean(Ytrain)))
                        n_oom += 1
                else:
                    print(f"  [CHGNet] Warning: prediction failed at idx {idx}: {exc}")
                    preds.append(float(np.mean(Ytrain)))
            except Exception as exc:
                print(f"  [CHGNet] Warning: prediction failed at idx {idx}: {exc}")
                preds.append(float(np.mean(Ytrain)))

            if (k + 1) % 2000 == 0:
                print(f"  [CHGNet] predicted {k+1}/{len(Xtest)} (OOM fallbacks so far: {n_oom})")

        if n_oom:
            print(f"  [CHGNet] Total OOM fallbacks (CPU rescue): {n_oom}/{len(Xtest)}")
        return np.array(preds, dtype=np.float32)
