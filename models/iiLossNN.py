"""
iiLossNN - ElementFraction MLP trained with a normalised MSE + xi^2 loss.
"""

import csv
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from matminer.featurizers.composition import ElementFraction

try:
    from pymatgen.core import Composition
except ImportError:
    from pymatgen import Composition

MAX_RXN_SIZE = 10

# -- Formula helpers ------------------------------------------------------

def _num_atoms(formula: str) -> int:
    """Total atom count for a formula string, e.g. 'Ac1Ag3' -> 4."""
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    return sum(int(n) if n else 1 for el, n in tokens if el)


def _is_element(formula: str) -> bool:
    """True if the formula contains only one distinct element symbol."""
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    return len(set(symbols)) == 1


# -- Architecture ---------------------------------------------------------

class _ResBlock(nn.Module):
    """Linear -> LayerNorm -> ReLU -> Dropout, with a skip connection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.skip = (nn.Linear(in_dim, out_dim, bias=False)
                     if in_dim != out_dim else nn.Identity())
        self.act  = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class _ResidualMLP(nn.Module):
    """Input projection, a stack of residual blocks, then a scalar head."""

    def __init__(self, input_dim: int,
                 hidden: tuple = (1024, 512, 256, 128),
                 dropout: float = 0.0):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.ReLU(),
        )

        blocks = []
        for in_h, out_h in zip(hidden[:-1], hidden[1:]):
            blocks.append(_ResBlock(in_h, out_h, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.out = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.out(x).squeeze(1)


# -- iiLossNN -------------------------------------------------------------

class iiLossNN(nn.Module):
    """ElementFraction MLP trained with a normalised MSE + xi^2 loss.

    Parameters
    ----------
    target : str
        Regression target key, 'Hf' or 'Hd'.
    lam : float
        Weight of the xi^2 term; 0 is pure MSE, 1 is pure xi^2.
    eps : float
        Stabiliser in the xi^2 denominator, in eV^2/atom^2.
    hidden : tuple of int
        Hidden layer widths.
    dropout : float
        Dropout probability; 0 disables it.
    lr : float
        Adam initial learning rate.
    epochs : int
        Total training epochs, full-batch.
    warmup_frac : float
        Fraction of epochs held at lam=0 before ramping to the target lam.
    renorm_every : int
        Recompute the loss reference scales every N epochs; 0 fixes them at
        their epoch-0 values.
    grad_clip : float
        Maximum gradient norm; 0 disables clipping.
    device : str or None
        'cuda', 'cpu', or None to auto-detect.
    checkpoint_dir : str or None
        Directory to write per-fold weights to after stage-one training.
    finetune_from : str or None
        Directory to load per-fold stage-one weights from.
    finetune_lr : float
        Learning rate used when fine-tuning.
    finetune_epochs : int
        Epoch count used when fine-tuning.
    use_pcgrad : bool
        Project the xi^2 gradient orthogonal to the MSE gradient each step.
    pcgrad_conditional : bool
        Project only on steps where the two gradients conflict, as in the
        published PCGrad rule. Ignored unless use_pcgrad is set.
    seed : int
        Base seed. Each fold derives its own RNG state from (seed, fold).
    history_path : str or None
        CSV to append the per-epoch training curve to.
    """

    model_type = "ii_nn"

    def __init__(
        self,
        target          = "Hf",
        lam             = 0.1,
        eps             = 1e-4,
        hidden          = (1024, 512, 256, 128),
        dropout         = 0.0,
        lr              = 1e-3,
        epochs          = 500,
        warmup_frac     = 0.1,
        renorm_every    = 50,
        grad_clip       = 1.0,
        device          = None,
        checkpoint_dir  = None,
        finetune_from   = None,
        finetune_lr     = 1e-4,
        finetune_epochs = 200,
        use_pcgrad      = False,
        pcgrad_conditional = False,
        seed            = 0,
        history_path    = None,
    ):
        super().__init__()
        self.target          = target
        self.lam             = lam
        self.eps             = eps
        self.hidden          = tuple(hidden)
        self.dropout         = dropout
        self.lr              = lr
        self.epochs          = epochs
        self.warmup_frac     = warmup_frac
        self.renorm_every    = renorm_every
        self.grad_clip       = grad_clip
        self.device          = (device if device
                                else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_dir  = checkpoint_dir
        self.finetune_from   = finetune_from
        self.finetune_lr     = finetune_lr
        self.finetune_epochs = finetune_epochs
        self.use_pcgrad      = use_pcgrad
        self.pcgrad_conditional = pcgrad_conditional
        self.seed            = seed
        self.history_path    = history_path
        self.history         = []

        self._labels      = None
        self._data        = None
        self._net         = None
        self._fold_counter = 0

    # -- MLModel interface ------------------------------------------------

    def preprocess(self, X):
        """Featurise with ElementFraction.
        Column 0 of the returned features is the row index, so that the KFold
        slicing in process.py carries the label mapping through unchanged.
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

        fold_idx = self._fold_counter
        self._fold_counter += 1

        self._seed_fold(fold_idx)

        train_indices = X[:, 0].astype(int)
        X_feat        = X[:, 1:].astype(np.float32)

        train_labels  = self._labels[train_indices]
        label_to_pos  = {lbl: i for i, lbl in enumerate(train_labels)}

        R_idx_list, R_coeff_list, R_mask_list = [], [], []

        for lbl in train_labels:
            entry   = self._data.get(lbl, {})
            rxn_str = entry.get("rxn", "")
            if not rxn_str:
                continue
            pairs = _parse_rxn(rxn_str, label_to_pos)
            # The reactant enters the reaction with coefficient -1.
            pairs.insert(0, (label_to_pos[lbl], -1.0))
            if len(pairs) < 2:
                # No non-elemental products, so Hd is undefined here.
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
            print("  [iiLossNN] Warning: no reactions in fold - pure MSE.")

        input_dim = X_feat.shape[1]
        self._net = _ResidualMLP(input_dim, self.hidden, self.dropout).to(dev)

        is_finetune = self.finetune_from is not None
        if is_finetune:
            ckpt_path = os.path.join(self.finetune_from,
                                     f"{self.target}_fold{fold_idx}_w{self.hidden[0]}_seed{self.seed}.pt")
            state = torch.load(ckpt_path, map_location=dev)
            self._net.load_state_dict(state)
            print(f"  [iiLossNN] Loaded pretrained weights: {ckpt_path}")
            actual_lr      = self.finetune_lr
            actual_epochs  = self.finetune_epochs
            # Stage two starts converged, so no warmup and fixed references.
            actual_warmup  = 0.0
            actual_renorm  = 0
        else:
            actual_lr      = self.lr
            actual_epochs  = self.epochs
            actual_warmup  = self.warmup_frac
            actual_renorm  = self.renorm_every

        # -- Optimiser and scheduler --------------------------------------
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=actual_lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=actual_epochs, eta_min=actual_lr * 0.01
        )

        X_t = torch.tensor(X_feat, dtype=torch.float32, device=dev)
        Y_t = torch.tensor(Y,      dtype=torch.float32, device=dev)

        # lam is held at 0 for the first warmup_frac of epochs, ramped over
        # the next warmup_frac, then held at its target value.
        warmup_end  = int(actual_warmup * actual_epochs)
        ramp_end    = int(2 * actual_warmup * actual_epochs)

        def effective_lam(epoch: int) -> float:
            if epoch < warmup_end:
                return 0.0
            if epoch < ramp_end:
                return self.lam * (epoch - warmup_end) / max(ramp_end - warmup_end, 1)
            return self.lam

        mse_ref, xi2_ref = self._compute_refs(
            X_t, Y_t,
            R_idx   if has_rxn else None,
            R_coeff if has_rxn else None,
            R_mask  if has_rxn else None,
            has_rxn, dev,
        )

        self._net.train()
        for epoch in range(actual_epochs):

            if (actual_renorm > 0
                    and epoch > 0
                    and epoch % actual_renorm == 0):
                mse_ref, xi2_ref = self._compute_refs(
                    X_t, Y_t,
                    R_idx   if has_rxn else None,
                    R_coeff if has_rxn else None,
                    R_mask  if has_rxn else None,
                    has_rxn, dev,
                )

            lam_eff = effective_lam(epoch)

            if self.use_pcgrad and has_rxn and lam_eff > 0:
                # Two backward passes so the gradients can be projected
                # against each other before the optimiser step.
                optimizer.zero_grad()
                pred      = self._net(X_t)
                delta     = pred - Y_t
                mse_loss  = (delta ** 2).mean() / mse_ref
                mse_loss.backward()
                g_mse = [p.grad.detach().clone() if p.grad is not None
                         else torch.zeros_like(p)
                         for p in self._net.parameters()]

                optimizer.zero_grad()
                pred      = self._net(X_t)
                delta     = pred - Y_t
                c         = R_coeff * delta[R_idx] * R_mask
                num       = c.sum(dim=1) ** 2
                den       = (c ** 2).sum(dim=1) + self.eps
                xi_loss   = (num / den).mean() / xi2_ref
                xi_loss.backward()
                g_xi = [p.grad.detach().clone() if p.grad is not None
                        else torch.zeros_like(p)
                        for p in self._net.parameters()]

                # Remove the part of the xi^2 gradient that opposes the MSE
                # gradient in parameter space. In the conditional form the
                # projection is applied only when the two gradients conflict.
                dot   = sum((a * b).sum() for a, b in zip(g_xi, g_mse))
                norm2 = sum((b * b).sum() for b in g_mse).clamp(min=1e-12)
                if self.pcgrad_conditional and dot >= 0:
                    g_xi_proj = g_xi
                else:
                    g_xi_proj = [a - (dot / norm2) * b
                                 for a, b in zip(g_xi, g_mse)]

                optimizer.zero_grad()
                for p, gm, gp in zip(self._net.parameters(), g_mse, g_xi_proj):
                    p.grad = (1.0 - lam_eff) * gm + lam_eff * gp

                loss = (1.0 - lam_eff) * mse_loss + lam_eff * xi_loss  # logging only

            else:
                optimizer.zero_grad()
                pred      = self._net(X_t)
                delta     = pred - Y_t
                mse_loss  = (delta ** 2).mean() / mse_ref

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

            self.history.append({
                "fold": fold_idx, "seed": self.seed, "epoch": epoch,
                "lam_eff": lam_eff,
                "mse_over_ref": float(mse_loss),
                "penalty_over_ref": float(xi_loss),
                "total": float(loss),
            })

            if epoch % 50 == 0 or epoch == actual_epochs - 1:
                phase = ("warmup" if epoch < warmup_end
                         else "ramp" if epoch < ramp_end
                         else "finetune" if is_finetune
                         else "train")
                msg = (f"  [iiLossNN] ep {epoch:>4d} [{phase}] "
                       f"loss={loss.item():.5f}  "
                       f"MSE/ref={mse_loss.item():.5f}  "
                       f"lam_eff={lam_eff:.3f}")
                if lam_eff > 0 and has_rxn:
                    msg += f"  xi2/ref={xi_loss.item():.5f}"
                print(msg)

        self._net.eval()

        self._flush_history()

        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(self.checkpoint_dir,
                                     f"{self.target}_fold{fold_idx}_w{self.hidden[0]}_seed{self.seed}.pt")
            torch.save(self._net.state_dict(), ckpt_path)
            print(f"  [iiLossNN] Saved checkpoint: {ckpt_path}")

        return self

    def predict(self, X):
        dev = torch.device(self.device)
        X_t = torch.tensor(X[:, 1:], dtype=torch.float32, device=dev)
        with torch.no_grad():
            preds = self._net(X_t).cpu().numpy()
        return [float(v) for v in preds]

    def fit_and_predict(self, Xtrain, Ytrain, Xtest):
        """Fit on the training fold and predict the held-out fold."""
        return self.fit(Xtrain, Ytrain).predict(Xtest)


    # -- Internal helpers -------------------------------------------------

    def _flush_history(self):
        """Append this fold's curve to the CSV, writing a header once."""
        if not self.history_path or not self.history:
            return
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        fields = list(self.history[0])
        new = not os.path.exists(self.history_path)
        with open(self.history_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerows(self.history)
        self.history = []

    def _seed_fold(self, fold_idx):
        """Seed every RNG this fold will draw from.

        SeedSequence hashes the (seed, fold) pair into a statistically
        independent stream, so folds do not share an initialisation and
        distinct base seeds cannot collide.
        """
        entropy = int(np.random.SeedSequence([self.seed, fold_idx])
                      .generate_state(1, dtype=np.uint32)[0])
        random.seed(entropy)
        np.random.seed(entropy)
        torch.manual_seed(entropy)
        torch.cuda.manual_seed_all(entropy)

    def _compute_refs(self, X_t, Y_t, R_idx, R_coeff, R_mask,
                      has_rxn, dev):
        """Loss reference scales taken at the current model state.

        Both terms are divided by these so that lam weights them comparably;
        without it lam would not mean the same thing across the two losses.
        """
        var_y = Y_t.var(unbiased=False).clamp(min=1e-6)
        with torch.no_grad():
            pred0  = self._net(X_t)
            delta0 = pred0 - Y_t
            # In stage two the model starts from the stage-one checkpoint, so
            # its MSE is already far below the target variance and this clamp
            # always binds. The reference is therefore var_y, not the current
            # MSE. That is deliberate: a reference that shrank with the model
            # would make the accuracy term grow without bound as it improved.
            mse_ref = (delta0 ** 2).mean().clamp(min=var_y)

            if has_rxn:
                c0      = R_coeff * delta0[R_idx] * R_mask
                num0    = c0.sum(dim=1) ** 2
                den0    = (c0 ** 2).sum(dim=1) + self.eps
                xi2_ref = (num0 / den0).mean().clamp(min=1e-6)
            else:
                xi2_ref = torch.ones(1, device=dev)

        return mse_ref, xi2_ref


# -- Reaction parsing -----------------------------------------------------

def _parse_rxn(rxn_str, label_to_pos):
    """Parse the product side of a reaction into [(position, weight), ...].

    Elements are skipped: their formation enthalpy is zero by definition, so
    they contribute nothing to the aggregate error.
    """
    pairs = []
    for token in rxn_str.split("+"):
        token = token.strip()
        m = re.match(r"^([\d.]+)_(.+)$", token)
        if m:
            coeff   = float(m.group(1))
            formula = m.group(2).strip()
            if _is_element(formula):
                continue
            if formula in label_to_pos:
                pairs.append((label_to_pos[formula], coeff))
    return pairs