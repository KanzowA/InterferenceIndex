"""
iiLossNN — ElementFraction MLP trained with a normalised MSE + ξ² loss.

Loss:
    L = (1 − λ) · MSE_Ef / MSE_ref  +  λ · ξ² / ξ²_ref

where MSE_ref and ξ²_ref are periodically refreshed reference values so that
the λ-weighting is reinstated during training, not just at epoch 0.

Architecture:
    ResidualMLP — input → projection → ResBlock × L → output
    LayerNorm (no running stats, works cleanly with full-batch and at eval time).
    Skip connections at every layer so the subtle ξ² gradient signal
    has short paths back to early weights.

Training improvements:
    warmup_frac  — first fraction of epochs trains with pure MSE (λ=0),
                   giving Ef accuracy a head start before ξ² steers.
    renorm_every — recompute MSE_ref and ξ²_ref every N epochs so the
                   λ fractional weighting stays honest as MSE decays.
    grad_clip    — clip gradient norm to stabilise the coupled ξ² updates.

Called by (λ = 0.1):
    python train_models.py allMP Ef iiLoss_0.1

Metrics:
    python interference_score.py
"""

import re
import numpy as np
import torch
import torch.nn as nn

try:
    from pymatgen.core import Composition
except ImportError:
    from pymatgen import Composition

from matminer.featurizers.composition import ElementFraction
from mlstabilitytest.training.MLModel import MLModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RXN_SIZE = 10

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """
    Single residual block: main path (Linear → LN → ReLU → Dropout)
    with a skip projection for dimension changes.

        out = ReLU( LN( W·x ) ) + W_skip·x

    Using pre-activation style (LN before activation) stabilises
    gradient flow through many blocks.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
        )
        # Skip: project if dims differ, else pass-through
        self.skip = (nn.Linear(in_dim, out_dim, bias=False)
                     if in_dim != out_dim else nn.Identity())
        self.act  = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class _ResidualMLP(nn.Module):
    """
    input_dim → Linear(input_dim, hidden[0]) → LN → ReLU
              → ResBlock(hidden[0] → hidden[1])
              → ResBlock(hidden[1] → hidden[2])
              → ...
              → Linear(hidden[-1], 1)
    """

    def __init__(self, input_dim: int,
                 hidden: tuple = (1024, 512, 256, 128),
                 dropout: float = 0.0):
        super().__init__()

        # Input projection (no skip — maps from raw features)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.ReLU(),
        )

        # Residual blocks
        blocks = []
        for in_h, out_h in zip(hidden[:-1], hidden[1:]):
            blocks.append(_ResBlock(in_h, out_h, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        # Output
        self.out = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.out(x).squeeze(1)


# ---------------------------------------------------------------------------
# iiLossNN
# ---------------------------------------------------------------------------

class iiLossNN(MLModel):
    """
    ElementFraction MLP with ξ²-regularised loss (interference index loss).

    Parameters
    ----------
    target : str
        Regression target key ('Ef' or 'Ed').
    lam : float
        Weight of ξ² loss term (0 = pure MSE, 1 = pure ξ²).
    eps : float
        Stabiliser in ξ² denominator (eV²/atom²).
    hidden : tuple of int
        Hidden layer widths. Default: (1024, 512, 256, 128).
    dropout : float
        Dropout probability (0 = no dropout). Default: 0.0.
    lr : float
        Adam initial learning rate.
    epochs : int
        Total training epochs (full-batch).
    warmup_frac : float
        Fraction of epochs to train with λ=0 (pure MSE) before
        ramping to the target λ.  E.g. 0.1 = 50 warmup epochs
        out of 500.  Gives Ef accuracy a head start.
    renorm_every : int
        Recompute MSE_ref and ξ²_ref every N epochs so the
        (1-λ)/λ weighting stays honest as MSE decays during
        training.  Set to 0 to use fixed epoch-0 references
        (original behaviour).
    grad_clip : float
        Max gradient norm for clipping (0 = no clipping).
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    model_type = 'ii_nn'

    def __init__(
        self,
        target       = 'Ef',
        lam          = 0.1,
        eps          = 1e-4,
        hidden       = (1024, 512, 256, 128),
        dropout      = 0.0,
        lr           = 1e-3,
        epochs       = 500,
        warmup_frac  = 0.1,
        renorm_every = 50,
        grad_clip    = 1.0,
        device       = None,
    ):
        self.target       = target
        self.lam          = lam
        self.eps          = eps
        self.hidden       = tuple(hidden)
        self.dropout      = dropout
        self.lr           = lr
        self.epochs       = epochs
        self.warmup_frac  = warmup_frac
        self.renorm_every = renorm_every
        self.grad_clip    = grad_clip
        self.device       = (device if device
                             else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self._labels = None
        self._data   = None
        self._net    = None

    # ------------------------------------------------------------------
    # MLModel interface
    # ------------------------------------------------------------------
    def preprocess(self, X):
        """
        Featurise with ElementFraction.
        Returns features with prepended index column (col 0 = row index)
        so that KFold slicing in process.py doesn't need modification.
        """
        self._data = X

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

        self._labels   = np.array(labels)
        features       = np.array(rows,    dtype=np.float32)
        targets_arr    = np.array(targets, dtype=np.float32)

        idx_col            = np.arange(len(labels), dtype=np.float32).reshape(-1, 1)
        features_with_idx  = np.hstack([idx_col, features])   # (N, 119)

        return features_with_idx, targets_arr, self._labels

    def fit(self, X, Y):
        dev = torch.device(self.device)

        # -- Unpack index column -------------------------------------------
        train_indices = X[:, 0].astype(int)
        X_feat        = X[:, 1:].astype(np.float32)

        train_labels  = self._labels[train_indices]
        label_to_pos  = {lbl: i for i, lbl in enumerate(train_labels)}

        # -- Build reaction tensors ----------------------------------------
        R_idx_list, R_coeff_list, R_mask_list = [], [], []

        for lbl in train_labels:
            entry   = self._data.get(lbl, {})
            rxn_str = entry.get('rxn', '')
            if not rxn_str:
                continue
            pairs = _parse_rxn(rxn_str, label_to_pos)
            if len(pairs) < 2:
                continue

            idx_row   = [p[0] for p in pairs]
            coeff_row = [p[1] for p in pairs]
            mask_row  = [True] * len(pairs)

            pad = MAX_RXN_SIZE - len(pairs)
            idx_row   += [0]   * pad
            coeff_row += [0.0] * pad
            mask_row  += [False] * pad

            R_idx_list.append(idx_row[:MAX_RXN_SIZE])
            R_coeff_list.append(coeff_row[:MAX_RXN_SIZE])
            R_mask_list.append(mask_row[:MAX_RXN_SIZE])

        if R_idx_list:
            R_idx   = torch.tensor(R_idx_list,   dtype=torch.long,    device=dev)
            R_coeff = torch.tensor(R_coeff_list, dtype=torch.float32, device=dev)
            R_mask  = torch.tensor(R_mask_list,  dtype=torch.float32, device=dev)
            has_rxn = True
        else:
            has_rxn = False
            print("  [iiLossNN] Warning: no reactions in fold — pure MSE.")

        # -- Network, optimiser, scheduler ---------------------------------
        input_dim = X_feat.shape[1]
        self._net = _ResidualMLP(input_dim, self.hidden, self.dropout).to(dev)

        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        X_t = torch.tensor(X_feat, dtype=torch.float32, device=dev)
        Y_t = torch.tensor(Y,      dtype=torch.float32, device=dev)

        # -- Warmup schedule -----------------------------------------------
        # Phase 1: first warmup_frac of epochs → λ_eff = 0 (pure MSE)
        # Phase 2: next warmup_frac of epochs  → λ_eff ramps 0 → λ
        # Phase 3: remainder                   → λ_eff = λ
        warmup_end  = int(self.warmup_frac * self.epochs)
        ramp_end    = int(2 * self.warmup_frac * self.epochs)

        def effective_lam(epoch: int) -> float:
            if epoch < warmup_end:
                return 0.0
            if epoch < ramp_end:
                return self.lam * (epoch - warmup_end) / max(ramp_end - warmup_end, 1)
            return self.lam

        # -- Initial reference values --------------------------------------
        mse_ref, xi2_ref = self._compute_refs(
            X_t, Y_t, R_idx, R_coeff, R_mask, has_rxn, dev
        )

        # -- Training loop (full-batch) ------------------------------------
        self._net.train()
        for epoch in range(self.epochs):

            # Refresh normalisation references periodically
            if (self.renorm_every > 0
                    and epoch > 0
                    and epoch % self.renorm_every == 0):
                mse_ref, xi2_ref = self._compute_refs(
                    X_t, Y_t, R_idx, R_coeff, R_mask, has_rxn, dev
                )

            lam_eff = effective_lam(epoch)

            optimizer.zero_grad()

            pred  = self._net(X_t)
            delta = pred - Y_t

            mse_loss = (delta ** 2).mean() / mse_ref

            if has_rxn and lam_eff > 0:
                c       = R_coeff * delta[R_idx] * R_mask
                num     = c.sum(dim=1) ** 2
                den     = (c ** 2).sum(dim=1) + self.eps
                xi_loss = (num / den).mean() / xi2_ref
            else:
                xi_loss = torch.zeros(1, device=dev).squeeze()

            loss = (1.0 - lam_eff) * mse_loss + lam_eff * xi_loss
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self._net.parameters(), self.grad_clip)

            optimizer.step()
            scheduler.step()

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                phase = ('warmup' if epoch < warmup_end
                         else 'ramp' if epoch < ramp_end
                         else 'train')
                msg = (f'  [iiLossNN] ep {epoch:>4d} [{phase}] '
                       f'loss={loss.item():.5f}  '
                       f'MSE/ref={mse_loss.item():.5f}  '
                       f'lam_eff={lam_eff:.3f}')
                if lam_eff > 0 and has_rxn:
                    msg += f'  xi2/ref={xi_loss.item():.5f}'
                print(msg)

        self._net.eval()
        return self

    def predict(self, X):
        dev = torch.device(self.device)
        X_t = torch.tensor(X[:, 1:], dtype=torch.float32, device=dev)
        with torch.no_grad():
            preds = self._net(X_t).cpu().numpy()
        return [float(v) for v in preds]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_refs(self, X_t, Y_t, R_idx, R_coeff, R_mask,
                      has_rxn, dev):
        """
        Compute fresh MSE_ref and ξ²_ref from current model predictions.
        Called at epoch 0 and every renorm_every epochs so that the
        (1-λ)/λ weighting stays calibrated as MSE decays during training.
        """
        var_y = Y_t.var(unbiased=False).clamp(min=1e-6)
        with torch.no_grad():
            pred0  = self._net(X_t)
            delta0 = pred0 - Y_t
            mse_ref = (delta0 ** 2).mean().clamp(min=var_y)

            if has_rxn:
                c0      = R_coeff * delta0[R_idx] * R_mask
                num0    = c0.sum(dim=1) ** 2
                den0    = (c0 ** 2).sum(dim=1) + self.eps
                xi2_ref = (num0 / den0).mean().clamp(min=1e-6)
            else:
                xi2_ref = torch.ones(1, device=dev)

        return mse_ref, xi2_ref


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_rxn(rxn_str, label_to_pos):
    """
    Parse '0.6667_Ac + 0.3333_AcAg' into [(position, coeff), ...]
    for compounds present in label_to_pos.
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
