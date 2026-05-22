"""
EdLossNN — ElementFraction MLP trained with Ef MSE + α · Ed MSE.

Loss:
    L = mean_i(δ_i²)  +  α · mean_r( (Σ_k ν_k δ_k)² )

where
    δ_i              = Ef_pred_i − Ef_DFT_i      (per-compound error)
    Σ_k ν_k δ_k      = Ed_pred_r − Ed_DFT_r      (per-reaction Ed error)

Unlike GammaLossNN's γ² term, the Ed MSE term is NOT scale-normalised —
it directly penalises large reaction prediction errors in eV²/atom².
The Ef term is kept at full weight (no (1-α) factor) so Ef accuracy is
never actively suppressed; α only adds an additional Ed signal.

Usage:
    python train_models.py allMP_single Ef EdLoss_1
    python train_models.py allMP_single Ef EdLoss_5
    python train_models.py allMP_single Ef EdLoss_10
    python train_models.py allMP_single Ef EdLoss_25

    # After finding best alpha, run 5-fold:
    python train_models.py allMP Ef EdLoss_<best>
"""

import numpy as np
import torch

from mlstabilitytest.training.GammaLossNN import GammaLossNN, _MLP, _parse_rxn, MAX_RXN_SIZE


class EdLossNN(GammaLossNN):
    """
    ElementFraction MLP with direct Ed-MSE regularisation.

    Parameters
    ----------
    target : str
        Regression target key ('Ef' or 'Ed').
    alpha : float
        Weight on the Ed MSE term.  α=0 → pure Ef MSE (baseline).
        Suggested sweep: 1, 5, 10, 25.
    hidden : tuple of int
    lr : float
    epochs : int
    device : str or None
    """

    model_type = 'ed_nn'

    def __init__(
        self,
        target='Ef',
        alpha=5.0,
        hidden=(512, 256, 128),
        lr=1e-3,
        epochs=300,
        device=None,
    ):
        # Initialise via GammaLossNN with lam=0 (no γ² term used here)
        super().__init__(
            target=target,
            lam=0.0,
            hidden=hidden,
            lr=lr,
            epochs=epochs,
            device=device,
        )
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Override fit() to use the Ed-MSE loss instead of γ²
    # ------------------------------------------------------------------
    def fit(self, X, Y):
        """
        Parameters
        ----------
        X : np.ndarray  (N_train, 119)  — col 0 = original row index
        Y : np.ndarray  (N_train,)      — Ef target values
        """
        dev = torch.device(self.device)

        # ---- Recover indices and features --------------------------------
        train_indices = X[:, 0].astype(int)
        X_feat        = X[:, 1:].astype(np.float32)

        train_labels  = self._labels[train_indices]
        label_to_pos  = {lbl: i for i, lbl in enumerate(train_labels)}

        # ---- Build padded reaction tensors (same as GammaLossNN) ---------
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
            idx_row   += [0] * pad
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
            print("  [EdLossNN] Warning: no reactions found in training fold; "
                  "falling back to pure Ef MSE.")

        # ---- Build network -----------------------------------------------
        input_dim = X_feat.shape[1]
        self._net = _MLP(input_dim, self.hidden).to(dev)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        X_t      = torch.tensor(X_feat, dtype=torch.float32, device=dev)
        Y_t      = torch.tensor(Y,      dtype=torch.float32, device=dev)
        var_y    = Y_t.var(unbiased=False).clamp(min=1e-6)
        with torch.no_grad():
            mse_init = ((self._net(X_t) - Y_t) ** 2).mean().clamp(min=var_y)

        # ---- Training loop -----------------------------------------------
        self._net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            pred  = self._net(X_t)       # (N_train,)
            delta = pred - Y_t           # per-compound Ef errors

            # -- Normalised Ef MSE: MSE / MSE_init ∈ [0, 1], full weight always
            mse_loss = (delta ** 2).mean() / mse_init

            # -- Ed MSE: mean_r( (Σ_k ν_k δ_k)² )  [eV²/atom², not normalised]
            #    Keeping Ed term in absolute units is intentional — α then has
            #    units of 1/eV² and can be interpreted as "how many eV² of
            #    reaction error you tolerate per unit of (1-R²)".
            if has_rxn:
                with torch.no_grad() if self.alpha == 0 else torch.enable_grad():
                    c        = R_coeff * delta[R_idx]   # (M, K)
                    c        = c * R_mask               # zero padding
                    ed_resid = c.sum(dim=1)             # (M,)  = Ed_DFT - Ed_pred
                    ed_loss  = (ed_resid ** 2).mean()   # scalar, eV²/atom²
            else:
                ed_loss = torch.zeros(1, device=dev).squeeze()

            loss = mse_loss + self.alpha * ed_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                print(
                    f'  [EdLossNN α={self.alpha}] epoch {epoch:>4d} | '
                    f'loss={loss.item():.5f}  '
                    f'(1-R²)={mse_loss.item():.5f}  '
                    f'Ed_mse={ed_loss.item():.5f}'
                )

        self._net.eval()
        return self
