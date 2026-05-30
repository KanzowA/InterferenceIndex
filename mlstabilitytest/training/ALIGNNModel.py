"""
ALIGNNModel.py  —  ALIGNN wrapper for the stability-prediction pipeline.

Fits the same preprocess / fit_and_predict interface as MatminerModel and
iiLossNN, so it plugs straight into process.py / train_models.py.

Install dependencies (once, on the cluster):
    pip install alignn jarvis-tools --break-system-packages
    # DGL is pulled in automatically by alignn
    # Structures are fetched via plain requests (no mp_api needed)

Structures are fetched from the MP API using the material_id stored in each
hullout entry and cached as pymatgen JSON files under:
    <repo_root>/structures_cache/<material_id>.json

Set your API key:
    export MP_API_KEY="your-key-here"

Usage via train_models.py:
    python mlstabilitytest/train_models.py allMP_current Ef ALIGNN
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

# Eagerly import heavy deps so missing packages surface at import time,
# not silently per-structure inside a try/except loop.
try:
    import dgl
    from jarvis.core.atoms import Atoms as _JAtoms
    from jarvis.core.specie import get_node_attributes as _get_node_attr
    _JARVIS_OK = True
except ImportError as _e:
    _JARVIS_OK = False
    _JARVIS_ERR = str(_e)


# ── graph helpers (inlined from alignn.graphs to avoid dgl.data → torchdata) ──

def _compute_bond_cosines(edges):
    """Bond-angle cosines for line-graph edge features."""
    r1 = -edges.src["r"]
    r2 =  edges.dst["r"]
    cos = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1) + 1e-8
    )
    return {"h": torch.clamp(cos, -1.0, 1.0)}


def _nearest_neighbor_edges(atoms, cutoff=8.0, max_neighbors=12,
                             use_canonize=True):
    """k-NN edge list for a jarvis Atoms object (mirrors alignn.graphs)."""
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(nl) for nl in all_neighbors)
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        r_cut = max(lat.a, lat.b, lat.c) if cutoff < max(lat.a, lat.b, lat.c) \
                else 2 * cutoff
        return _nearest_neighbor_edges(atoms, r_cut, max_neighbors, use_canonize)

    def _canonize(src, dst, si, di):
        if dst < src:
            src, dst, si, di = dst, src, di, si
        if si != (0, 0, 0):
            shift = si
            si = tuple(np.subtract(si, shift))
            di = tuple(np.subtract(di, shift))
        return src, dst, si, di

    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([n[2] for n in neighborlist])
        ids       = np.array([n[1] for n in neighborlist])
        images    = np.array([n[3] for n in neighborlist])
        max_dist  = distances[max_neighbors - 1]
        ids    = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        for dst, image in zip(ids, images):
            if use_canonize:
                s, d, si, di = _canonize(site_idx, dst, (0,0,0), tuple(image))
                edges[(s, d)].add(di)
            else:
                edges[(site_idx, dst)].add(tuple(image))
    return edges, images


def _build_undirected_edgedata(atoms, edges):
    """Undirected edge arrays from _nearest_neighbor_edges output."""
    u, v, r, all_images = [], [], [], []
    for (src_id, dst_id), img_set in edges.items():
        for dst_image in img_set:
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            d = atoms.lattice.cart_coords(dst_coord - atoms.frac_coords[src_id])
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu); v.append(vv); r.append(dd)
                all_images.append(dst_image)
    u = torch.tensor(np.array(u))
    v = torch.tensor(np.array(v))
    r = torch.tensor(np.array(r), dtype=torch.float32)
    all_images = torch.tensor(np.array(all_images), dtype=torch.float32)
    return u, v, r, all_images

# ── repo paths ────────────────────────────────────────────────────────────────
from os.path import dirname, abspath, join

_HERE       = dirname(abspath(__file__))   # mlstabilitytest/training/
_BASE       = dirname(_HERE)               # mlstabilitytest/
_REPO       = dirname(_BASE)              # repo root
_STRUCT_DIR = join(_REPO, "structures_cache")


class ALIGNNModel:
    """
    ALIGNN (Atomistic Line Graph Neural Network) for formation-energy /
    hull-distance prediction from crystal structures.

    Reference:
        Choudhary & DeCost, npj Comput. Mater. 7, 185 (2021).
        https://doi.org/10.1038/s41524-021-00650-1
    """

    def __init__(
        self,
        target,
        *,
        cutoff           = 8.0,    # Å — neighbour cutoff for graph construction
        max_neighbors    = 12,     # max edges per atom
        atom_features    = "cgcnn",# atom featurisation scheme used by ALIGNN
        hidden_features  = 256,    # width of ALIGNN hidden layers
        n_layers         = 4,      # number of ALIGNN + line-graph update blocks
        epochs           = 100,
        batch_size       = 64,
        lr               = 1e-3,
        weight_decay     = 1e-5,
        structures_dir   = _STRUCT_DIR,
    ):
        self.target         = target
        self.cutoff         = float(cutoff)
        self.max_neighbors  = int(max_neighbors)
        self.atom_features  = atom_features
        self.hidden_features= hidden_features
        self.n_layers       = n_layers
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.structures_dir = structures_dir

        # populated during preprocess(); persists for fit_and_predict()
        self._graphs: list | None = None

    # ── structure helpers ─────────────────────────────────────────────────────

    def _load_cached(self, material_id: str):
        """Return cached pymatgen Structure or None."""
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

    def _batch_fetch(self, material_ids: list[str]) -> dict:
        """
        Fetch structures from the MP REST API v2 using plain requests —
        no mp_api / emmet-core imports, so no Python-version dependency hell.
        """
        import requests
        from pymatgen.core import Structure

        api_key  = os.environ.get("MP_API_KEY", "")
        base_url = "https://api.materialsproject.org/materials/core/"
        headers  = {"X-Api-Key": api_key}
        result   = {}
        chunk_size = 50

        for start in range(0, len(material_ids), chunk_size):
            chunk = material_ids[start:start + chunk_size]
            params = {
                "material_ids": ",".join(chunk),
                "_fields":      "material_id,structure",
            }
            resp = requests.get(base_url, headers=headers,
                                params=params, timeout=120)
            if not resp.ok:
                print(f"  Warning: HTTP {resp.status_code} — {resp.text[:200]}")
                resp.raise_for_status()
            data = resp.json().get("data", [])

            for item in data:
                mid       = item["material_id"]
                structure = Structure.from_dict(item["structure"])
                self._save_structure(mid, structure)
                result[mid] = structure

            fetched = start + len(chunk)
            print(f"  [ALIGNN] fetched {fetched}/{len(material_ids)} structures")

        return result

    def _to_graph(self, structure):
        """pymatgen Structure → (atom-graph g, line-graph lg, lattice lat) triple.

        ALIGNN.forward() unpacks its input as (g, lg, lat).

        We build graphs manually rather than via Graph.atom_dgl_multigraph because:
          1. str(site.specie) gives 'Al3+' for oxidation-state decorated structures;
             site.specie.symbol always gives the bare element symbol.
          2. g.line_graph(shared=True) has a DGL 2.x bug that returns a graph with
             a null internal handle for certain multi-element / large-cell structures.
             Using shared=False + manual ndata copy is the safe DGL-2.x-compatible path.
        """
        if not _JARVIS_OK:
            raise ImportError(_JARVIS_ERR)

        # ── element symbols — strip oxidation state if present ────────────────
        elements = [s.specie.symbol for s in structure]

        atoms = _JAtoms(
            lattice_mat = structure.lattice.matrix.tolist(),
            elements    = elements,
            coords      = structure.frac_coords.tolist(),
            cartesian   = False,
        )

        # ── edges (local helpers — no alignn.graphs import needed) ────────────
        edges, _ = _nearest_neighbor_edges(
            atoms, self.cutoff, self.max_neighbors, use_canonize=True)
        u, v, r, images = _build_undirected_edgedata(atoms, edges)

        # ── atom features ─────────────────────────────────────────────────────
        sps_feats = []
        for el in atoms.elements:
            feat = _get_node_attr(el, atom_features=self.atom_features)
            if feat is None:
                raise ValueError(f"No CGCNN features for element '{el}'")
            sps_feats.append(feat)
        node_features = torch.tensor(np.array(sps_feats), dtype=torch.float32)

        # ── atom graph ────────────────────────────────────────────────────────
        g = dgl.graph((u, v))
        g.ndata["atom_features"] = node_features
        g.ndata["V"]             = torch.tensor(
            [atoms.volume] * atoms.num_atoms, dtype=torch.float32)
        g.ndata["frac_coords"]   = torch.tensor(
            atoms.frac_coords, dtype=torch.float32)
        g.edata["r"]             = r.float()
        g.edata["images"]        = images.float()

        # ── line graph (shared=False is the DGL 2.x-safe path) ───────────────
        lg = g.line_graph(shared=False)
        lg.ndata["r"] = g.edata["r"]   # nodes of lg = edges of g
        lg.apply_edges(_compute_bond_cosines)

        lat = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
        return g, lg, lat

    # ── pipeline interface ────────────────────────────────────────────────────

    def preprocess(self, input_data: dict):
        """
        Build ALIGNN graphs for every entry that has a material_id.

        Parameters
        ----------
        input_data : dict
            {formula: {"Ef": float, "Ed": float, "material_id": str, ...}}

        Returns
        -------
        indices : np.ndarray of int   — positions into self._graphs
        targets : np.ndarray of float
        labels  : np.ndarray of str   — formula strings
        """
        # ── 1. collect which structures we need ───────────────────────────────
        entries   = []   # (formula, target_val, material_id)
        needs_fetch = []

        for formula, props in input_data.items():
            t_val = props.get(self.target)
            mid   = props.get("material_id")
            if t_val is None or mid is None:
                continue
            entries.append((formula, float(t_val), mid))
            if self._load_cached(mid) is None:
                needs_fetch.append(mid)

        # ── 2. batch-fetch missing structures ─────────────────────────────────
        if needs_fetch:
            print(f"  [ALIGNN] fetching {len(needs_fetch)} structures from MP…")
            self._batch_fetch(list(set(needs_fetch)))

        # ── 3. build graphs ───────────────────────────────────────────────────
        graphs, targets_list, labels_list = [], [], []
        n = len(entries)

        for k, (formula, t_val, mid) in enumerate(entries):
            if (k + 1) % 1000 == 0:
                print(f"  [ALIGNN] building graphs {k+1}/{n}")
            try:
                structure = self._load_cached(mid)
                if structure is None:
                    raise FileNotFoundError(f"Structure not in cache: {mid}")
                graph_pair = self._to_graph(structure)
            except Exception as exc:
                print(f"  Warning: skipping {formula} ({mid}): {exc}")
                continue

            graphs.append(graph_pair)
            targets_list.append(t_val)
            labels_list.append(formula)

        self._graphs = graphs
        print(f"  [ALIGNN] preprocess done — {len(graphs)}/{n} entries kept")

        indices = np.arange(len(graphs), dtype=np.int64)
        return indices, np.array(targets_list, dtype=np.float32), np.array(labels_list)

    def fit_and_predict(self, Xtrain, Ytrain, Xtest):
        """
        Train ALIGNN on (Xtrain, Ytrain) and return predictions for Xtest.

        Parameters
        ----------
        Xtrain, Xtest : array-like of int — indices into self._graphs
        Ytrain        : array-like of float

        Returns
        -------
        preds : np.ndarray of float, shape (len(Xtest),)
        """
        import dgl
        from alignn.models.alignn import ALIGNN, ALIGNNConfig

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [ALIGNN] train={len(Xtrain)}  test={len(Xtest)}  device={dev}")

        # ── datasets & loaders ────────────────────────────────────────────────
        # ALIGNN.forward() unpacks its input as (g, lg, lat), so every batch
        # must include the lattice matrix tensor as the third graph element.

        def _collate(batch):
            gs, lgs, lats, ys = zip(*batch)
            return (dgl.batch(gs), dgl.batch(lgs), torch.stack(lats),
                    torch.tensor(ys, dtype=torch.float32))

        train_ds = [(self._graphs[i][0], self._graphs[i][1],
                     self._graphs[i][2], float(Ytrain[j]))
                    for j, i in enumerate(Xtrain)]
        test_ds  = [(self._graphs[i][0], self._graphs[i][1],
                     self._graphs[i][2], 0.0)
                    for i in Xtest]

        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True,  collate_fn=_collate, num_workers=0)
        test_dl  = DataLoader(test_ds,  batch_size=self.batch_size,
                              shuffle=False, collate_fn=_collate, num_workers=0)

        # ── model ─────────────────────────────────────────────────────────────
        atom_feat_size = self._graphs[0][0].ndata["atom_features"].shape[1]

        config = ALIGNNConfig(
            name                  = "alignn",
            atom_input_features   = atom_feat_size,
            edge_input_features   = 80,   # RBF-encoded bond lengths
            triplet_input_features= 40,   # RBF-encoded bond angles
            embedding_features    = 64,
            hidden_features       = self.hidden_features,
            output_features       = 1,
            alignn_layers         = self.n_layers,
            gcn_layers            = self.n_layers,
            link                  = "identity",
            classification        = False,
        )

        model = ALIGNN(config).to(dev)
        opt   = torch.optim.AdamW(model.parameters(),
                                  lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=self.epochs, eta_min=self.lr * 0.01)
        loss_fn = torch.nn.MSELoss()

        # ── training loop ─────────────────────────────────────────────────────
        for epoch in range(self.epochs):
            model.train()
            ep_loss, ep_n = 0.0, 0
            for bg, blg, blat, yt in train_dl:
                bg, blg, blat, yt = (bg.to(dev), blg.to(dev),
                                     blat.to(dev), yt.to(dev))
                opt.zero_grad()
                pred = model((bg, blg, blat)).squeeze(-1)
                loss = loss_fn(pred, yt)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(yt)
                ep_n    += len(yt)
            sched.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    epoch {epoch+1:>3}/{self.epochs}  "
                      f"train MSE={ep_loss/ep_n:.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}")

        # ── inference ─────────────────────────────────────────────────────────
        model.eval()
        preds = []
        with torch.no_grad():
            for bg, blg, blat, _ in test_dl:
                bg, blg, blat = bg.to(dev), blg.to(dev), blat.to(dev)
                out = model((bg, blg, blat)).squeeze(-1)
                preds.extend(out.cpu().tolist())

        return np.array(preds, dtype=np.float32)
