"""
GammaLossNN — ElementFraction MLP trained with a normalised (1−R²) + γ²/N loss.

Loss:
    L = (1 − λ) · MSE_Ef / Var(Y)  +  λ · mean_r( γ²_r / N_r )

where
    MSE_Ef / Var(Y)  =  1 − R²                             (normalised Ef error, ∈ [0,1])
    γ²_r / N_r       =  (Σ_k ν_k δ_k)² / (N_r·(Σ_k (ν_k δ_k)² + ε))
                                                            (normalised γ², ∈ [0,1])
    δ_i  = Ef_pred_i − Ef_DFT_i
    N_r  = number of non-padded members in reaction r
    Var(Y) = population variance of training Ef values (fixed per fold)

Both terms live in [0, 1] at initialisation, so λ is a genuine fractional weight
throughout training — not a threshold that tips gradient dominance from one term
to the other as MSE decays.

Suggested λ sweep with normalised loss: 0.1, 0.2, 0.3, 0.5, 0.7

Plugs directly into the existing Bartel-et-al. infrastructure:
    python train_models.py allMP Ef GammaLoss_0.3
    python interference_score.py
"""

import re
import numpy as np
import torch
import torch.nn as nn

try:
    from pymatgen.core import Composition
except ImportError:
    from pymatgen import Composition  # older pymatgen

from matminer.featurizers.composition import ElementFraction
from mlstabilitytest.training.MLModel import MLModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RXN_SIZE = 10   # maximum compounds per decomposition reaction in hullout.json


# ---------------------------------------------------------------------------
# MLP definition
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    """Simple feed-forward net:  input → [Linear → BN → ReLU] × L → Linear → scalar"""

    def __init__(self, input_dim, hidden=(512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)   # (N,)


# ---------------------------------------------------------------------------
# GammaLossNN — implements MLModel interface
# ---------------------------------------------------------------------------
class GammaLossNN(MLModel):
    """
    ElementFraction MLP with γ²-regularised loss.

    Parameters
    ----------
    target : str
        Key in the input dict to use as the regression target ('Ef' or 'Ed').
    lam : float
        Weight of γ² loss term (0 = pure MSE, 1 = pure γ²).
        Recommended sweep: 0.0, 0.05, 0.1, 0.2, 0.5.
    eps : float
        Stabiliser added to the γ² denominator (eV²/atom²).
    hidden : tuple of int
        Hidden layer sizes.
    lr : float
        Adam learning rate.
    epochs : int
        Training epochs (full-batch).
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    model_type = 'gamma_nn'

    def __init__(
        self,
        target='Ef',
        lam=0.25,
        eps=1e-4,
        hidden=(512, 256, 128),
        lr=1e-3,
        epochs=300,
        device=None,
    ):
        self.target  = target
        self.lam     = lam
        self.eps     = eps
        self.hidden  = tuple(hidden)
        self.lr      = lr
        self.epochs  = epochs
        self.device  = (
            device if device
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Set in preprocess(); used by fit() to recover reaction graph
        self._labels = None   # (N_total,) np.ndarray of formula strings
        self._data   = None   # full hullout dict passed to preprocess()
        self._net    = None

    # ------------------------------------------------------------------
    # MLModel interface
    # ------------------------------------------------------------------
    def preprocess(self, X):
        """
        Featurise with ElementFraction.

        The returned feature array has an extra FIRST column containing the
        original row index.  This threads through KFold's numpy slicing so
        that fit() can recover which formulas — and thus which reactions —
        belong to the current training fold without any changes to process.py.

        Parameters
        ----------
        X : dict  {formula: {"Ef": float, "Ed": float, "rxn": str, ...}}

        Returns
        -------
        features_with_idx : np.ndarray  (N, 1+118)   col 0 = row index
        targets           : np.ndarray  (N,)
        labels            : np.ndarray  (N,)  formula strings
        """
        self._data = X   # store for reaction lookup in fit()

        featurizer = ElementFraction()

        labels, targets, rows = [], [], []
        for formula, entry in X.items():
            try:
                comp = Composition(formula)
                vec  = featurizer.featurize(comp)
                rows.append(vec)
                targets.append(float(entry[self.target]))
                labels.append(formula)
            except Exception:
                continue

        self._labels = np.array(labels)
        features     = np.array(rows, dtype=np.float32)   # (N, 118)
        targets_arr  = np.array(targets, dtype=np.float32)

        # Prepend index column
        N       = len(labels)
        idx_col = np.arange(N, dtype=np.float32).reshape(-1, 1)
        features_with_idx = np.hstack([idx_col, features])   # (N, 119)

        return features_with_idx, targets_arr, self._labels

    def fit(self, X, Y):
        """
        Train the network.

        Parameters
        ----------
        X : np.ndarray  (N_train, 119)  — col 0 = original row index
        Y : np.ndarray  (N_train,)      — target values

        Returns
        -------
        self
        """
        dev = torch.device(self.device)

        # ---- Recover indices and features --------------------------------
        train_indices = X[:, 0].astype(int)        # which rows from preprocess()
        X_feat        = X[:, 1:].astype(np.float32)  # (N_train, 118)
        N_train       = len(train_indices)

        # Map formula → position within this training fold
        train_labels      = self._labels[train_indices]
        label_to_pos      = {lbl: i for i, lbl in enumerate(train_labels)}

        # ---- Build padded reaction tensors --------------------------------
        # R_idx   (M, K): compound positions in the training fold (0-padded)
        # R_coeff (M, K): stoichiometric coefficients (0-padded)
        # R_mask  (M, K): True for real entries
        R_idx_list, R_coeff_list, R_mask_list = [], [], []

        for lbl in train_labels:
            entry   = self._data.get(lbl, {})
            rxn_str = entry.get('rxn', '')
            if not rxn_str:
                continue
            pairs = _parse_rxn(rxn_str, label_to_pos)
            if len(pairs) < 2:
                continue   # skip reactions where <2 members are in this fold

            idx_row   = [p[0] for p in pairs]
            coeff_row = [p[1] for p in pairs]
            mask_row  = [True] * len(pairs)

            # Pad to MAX_RXN_SIZE
            pad = MAX_RXN_SIZE - len(pairs)
            idx_row   += [0] * pad
            coeff_row += [0.0] * pad
            mask_row  += [False] * pad

            R_idx_list.append(idx_row[:MAX_RXN_SIZE])
            R_coeff_list.append(coeff_row[:MAX_RXN_SIZE])
            R_mask_list.append(mask_row[:MAX_RXN_SIZE])

        if R_idx_list:
            R_idx   = torch.tensor(R_idx_list,   dtype=torch.long,  device=dev)  # (M, K)
            R_coeff = torch.tensor(R_coeff_list, dtype=torch.float32, device=dev)
            R_mask  = torch.tensor(R_mask_list,  dtype=torch.float32, device=dev)  # 1/0
            has_rxn = True
        else:
            has_rxn = False
            print("  [GammaLossNN] Warning: no reactions found in training fold; "
                  "falling back to pure MSE.")

        # ---- Build network ------------------------------------------------
        input_dim = X_feat.shape[1]
        self._net = _MLP(input_dim, self.hidden).to(dev)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        X_t = torch.tensor(X_feat, dtype=torch.float32, device=dev)
        Y_t = torch.tensor(Y,      dtype=torch.float32, device=dev)

        # ---- Normalisation constants (fixed for this fold) ----------------
        # mse_init : MSE of the untrained network on the training set.
        #            Normalising by this guarantees the Ef term starts at
        #            exactly 1.0 at epoch 0, matching γ²/N ≈ 1.0.
        #            λ is then a genuine fractional weight throughout training.
        # var_y    : used as a floor so mse_init never collapses to zero.
        var_y    = Y_t.var(unbiased=False).clamp(min=1e-6)
        with torch.no_grad():
            mse_init = ((self._net(X_t) - Y_t) ** 2).mean().clamp(min=var_y)

        # ---- Training loop (full-batch) -----------------------------------
        # Full-batch is required: γ² needs to see complete reactions in one
        # forward pass.  68 K compounds × 118 features ≈ 30 MB — fits easily.
        self._net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            pred  = self._net(X_t)          # (N_train,)
            delta = pred - Y_t              # per-compound errors

            # -- Normalised MSE term: MSE / MSE_init ∈ [0, 1]
            #    = 1.0 at epoch 0, → 0 as training improves
            mse_loss = (delta ** 2).mean() / mse_init

            # -- Normalised γ²/N term ∈ [0, 1]
            #    γ²_r / N_r  =  (Σ c)² / (N_r · (Σ c² + ε))
            #    dividing by N_r brings the upper bound to 1 for any
            #    reaction size, making the term truly scale-invariant.
            if has_rxn:
                with torch.no_grad() if self.lam == 0 else torch.enable_grad():
                    c    = R_coeff * delta[R_idx]        # (M, K)
                    c    = c * R_mask                    # zero padding
                    N_r  = R_mask.sum(dim=1)             # (M,) real members per reaction
                    num  = c.sum(dim=1) ** 2             # (M,)
                    den  = (c ** 2).sum(dim=1) + self.eps
                    gamma_loss = (num / (N_r * den)).mean()
            else:
                gamma_loss = torch.zeros(1, device=dev).squeeze()

            loss = (1.0 - self.lam) * mse_loss + self.lam * gamma_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                print(
                    f'  [GammaLossNN] epoch {epoch:>4d} | '
                    f'loss={loss.item():.5f}  '
                    f'(1-R²)={mse_loss.item():.5f}  '
                    f'γ²/N={gamma_loss.item():.5f}'
                )

        self._net.eval()
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : np.ndarray  (N_test, 119)  — col 0 = index (ignored), cols 1.. = features

        Returns
        -------
        list of float
        """
        dev   = torch.device(self.device)
        X_t   = torch.tensor(X[:, 1:], dtype=torch.float32, device=dev)
        with torch.no_grad():
            preds = self._net(X_t).cpu().numpy()
        return [float(v) for v in preds]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _parse_rxn(rxn_str, label_to_pos):
    """
    Parse a reaction string like '0.6667_Ac + 0.3333_Ac1Ag3' into
    [(position, coeff), ...] for compounds present in label_to_pos.

    Coefficients represent the fraction of each end-member in the
    decomposition reaction (all positive; the compound itself is on the
    left-hand side, so its signed contribution is -1, but since we're
    computing errors relative to DFT, we can use unsigned coefficients —
    the sign convention for δ is already handled by the residual).
    """
    pairs = []
    for token in rxn_str.split('+'):
        token = token.strip()
        m = re.match(r'^([\d.]+)_(.+)$', token)
        if m:
            coeff   = float(m.group(1))
            formula = m.group(2).strip()
            if formula in label_to_pos:
                pairs.append((label_to_pos[formula], coeff))
    return pairs
