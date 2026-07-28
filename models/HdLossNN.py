"""HdLossNN - ElementFraction MLP trained with a normalised MSE + Hd^2 loss."""

import os

import numpy as np
import torch
import torch.nn as nn

from .iiLossNN import (
    iiLossNN, _parse_rxn, MAX_RXN_SIZE, _ResidualMLP
)


class HdLossNN(iiLossNN):
    """ElementFraction MLP trained with a normalised MSE + Hd^2 loss.

    Parameters
    ----------
    target : str
        Regression target key, 'Hf' or 'Hd'.
    lam : float
        Weight of the Hd^2 term; 0 is pure MSE, 1 is pure Hd^2.
    eps : float
        Stabiliser added to the Hd^2 reference denominator, in eV^2/atom^2.
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
        Recompute the loss reference scales every N epochs; 0 fixes them.
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
        Project the Hd^2 gradient orthogonal to the MSE gradient each step.
    seed : int
        Base seed. Each fold derives its own RNG state from (seed, fold).
    history_path : str or None
        CSV to append the per-epoch training curve to.
    """

    model_type = "ed_nn"

    def __init__(
        self,
        target          = "Hf",
        lam           = 0.1,
        eps             = 1e-4,
        hidden          = (1024, 512, 256, 128),
        dropout         = 0.0,
        lr              = 1e-3,
        epochs          = 500,
        warmup_frac     = 0.1,
        renorm_every    = 50,
        grad_clip       = 1.0,
        device          = None,
        # two-stage fine-tuning
        checkpoint_dir  = None,
        finetune_from   = None,
        finetune_lr     = 1e-4,
        finetune_epochs = 200,
        use_pcgrad      = False,
        seed            = 0,
        history_path    = None,
    ):
        # Initialise via iiLossNN with lam=0; finetune params stored on self
        super().__init__(
            target          = target,
            lam             = 0.0,
            eps             = eps,
            hidden          = hidden,
            dropout         = dropout,
            lr              = lr,
            epochs          = epochs,
            warmup_frac     = warmup_frac,
            renorm_every    = renorm_every,
            grad_clip       = grad_clip,
            device          = device,
            checkpoint_dir  = checkpoint_dir,
            finetune_from   = finetune_from,
            finetune_lr     = finetune_lr,
            finetune_epochs = finetune_epochs,
            use_pcgrad      = use_pcgrad,
            seed            = seed,
            history_path    = history_path,
        )
        self.lam = lam

    # -- Reference computation --------------------------------------------

    def _compute_refs(self, X_t, Y_t, R_idx, R_coeff, R_mask, has_rxn, dev):
        """Loss reference scales taken at the current model state."""
        var_y = Y_t.var(unbiased=False).clamp(min=1e-6)
        with torch.no_grad():
            pred0   = self._net(X_t)
            delta0  = pred0 - Y_t
            mse_ref = (delta0 ** 2).mean().clamp(min=var_y)

            if has_rxn:
                c0       = R_coeff * delta0[R_idx] * R_mask
                hd_resid = c0.sum(dim=1)
                hd2_ref  = (hd_resid ** 2).mean().clamp(min=1e-6)
            else:
                hd2_ref = torch.ones(1, device=dev).squeeze()

        return mse_ref, hd2_ref

    # -- Training loop ----------------------------------------------------

    def fit(self, X, Y):
        dev = torch.device(self.device)

        fold_idx = self._fold_counter
        self._fold_counter += 1

        # Must precede network construction, which consumes the RNG.
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
            # Prepend reactant with coefficient -1
            pairs.insert(0, (label_to_pos[lbl], -1.0))
            if len(pairs) < 2:
                # Only reactant, no non-elemental products -> Hd undefined
                continue

            idx_row   = [p[0] for p in pairs]
            coeff_row = [p[1] for p in pairs]
            mask_row  = [True] * len(pairs)

            pad = MAX_RXN_SIZE - len(pairs)
            idx_row   += [0]     * pad
            coeff_row += [0.0]   * pad
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
            print("  [HdLossNN] Warning: no reactions in fold - pure MSE.")

        # Network
        input_dim = X_feat.shape[1]
        self._net = _ResidualMLP(input_dim, self.hidden, self.dropout).to(dev)

        # Stage-2: load pretrained weights if fine-tuning
        if self.finetune_from is not None:
            ckpt_path = os.path.join(
                self.finetune_from, f"{self.target}_fold{fold_idx}_w{self.hidden[0]}_seed{self.seed}.pt")
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=dev)
                self._net.load_state_dict(state)
                print(f"  [HdLossNN] loaded checkpoint: {ckpt_path}")
            else:
                print(f"  [HdLossNN] WARNING: checkpoint not found: {ckpt_path}")
            actual_lr      = self.finetune_lr
            actual_epochs  = self.finetune_epochs
            actual_warmup  = 0.0
            actual_renorm  = 0
        else:
            actual_lr      = self.lr
            actual_epochs  = self.epochs
            actual_warmup  = self.warmup_frac
            actual_renorm  = self.renorm_every

        # Optimiser & Scheduler
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=actual_lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=actual_epochs, eta_min=actual_lr * 0.01
        )

        X_t = torch.tensor(X_feat, dtype=torch.float32, device=dev)
        Y_t = torch.tensor(Y,      dtype=torch.float32, device=dev)

        # Warmup schedule
        warmup_end = int(actual_warmup * actual_epochs)
        ramp_end   = int(2 * actual_warmup * actual_epochs)

        def effective_lam(epoch: int) -> float:
            if epoch < warmup_end:
                return 0.0
            if epoch < ramp_end:
                return self.lam * (epoch - warmup_end) / max(ramp_end - warmup_end, 1)
            return self.lam

        # initial reference values (calibrated at checkpoint, not random init)
        mse_ref, hd2_ref = self._compute_refs(
            X_t, Y_t, R_idx if has_rxn else None,
            R_coeff if has_rxn else None,
            R_mask  if has_rxn else None,
            has_rxn, dev
        )

        # Training loop (full-batch)
        self._net.train()
        for epoch in range(actual_epochs):

            # Refresh refs periodically
            if (actual_renorm > 0
                    and epoch > 0
                    and epoch % actual_renorm == 0):
                mse_ref, hd2_ref = self._compute_refs(
                    X_t, Y_t, R_idx if has_rxn else None,
                    R_coeff if has_rxn else None,
                    R_mask  if has_rxn else None,
                    has_rxn, dev
                )

            lam_eff = effective_lam(epoch)

            # PCGrad path
            if self.use_pcgrad and has_rxn and lam_eff > 0:

                # MSE gradient
                optimizer.zero_grad()
                pred  = self._net(X_t)
                delta = pred - Y_t
                mse_loss = (delta ** 2).mean() / mse_ref
                mse_loss.backward()
                g_mse = [p.grad.detach().clone()
                         if p.grad is not None else torch.zeros_like(p)
                         for p in self._net.parameters()]

                optimizer.zero_grad()
                pred  = self._net(X_t)
                delta = pred - Y_t
                c        = R_coeff * delta[R_idx] * R_mask
                hd_resid = c.sum(dim=1)
                ed_loss  = (hd_resid ** 2).mean() / hd2_ref
                ed_loss.backward()
                g_ed = [p.grad.detach().clone()
                        if p.grad is not None else torch.zeros_like(p)
                        for p in self._net.parameters()]

                # Remove the part of the Hd^2 gradient that opposes the MSE
                # gradient in parameter space.
                dot   = sum((a * b).sum() for a, b in zip(g_ed,  g_mse))
                norm2 = sum((b * b).sum() for b in g_mse).clamp(min=1e-12)
                g_ed_proj = [a - (dot / norm2) * b
                             for a, b in zip(g_ed, g_mse)]

                optimizer.zero_grad()
                for p, gm, gp in zip(self._net.parameters(), g_mse, g_ed_proj):
                    p.grad = (1.0 - lam_eff) * gm + lam_eff * gp

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self._net.parameters(), self.grad_clip)
                optimizer.step()

                loss     = mse_loss  # logging only
                ed_log   = ed_loss

            else:
                optimizer.zero_grad()
                pred  = self._net(X_t)
                delta = pred - Y_t

                mse_loss = (delta ** 2).mean() / mse_ref

                if has_rxn and lam_eff > 0:
                    c        = R_coeff * delta[R_idx] * R_mask
                    hd_resid = c.sum(dim=1)
                    ed_loss  = (hd_resid ** 2).mean() / hd2_ref
                else:
                    ed_loss = torch.zeros(1, device=dev).squeeze()

                loss = (1.0 - lam_eff) * mse_loss + lam_eff * ed_loss
                loss.backward()

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self._net.parameters(), self.grad_clip)

                optimizer.step()
                ed_log = ed_loss

            scheduler.step()

            self.history.append({
                "fold": fold_idx, "seed": self.seed, "epoch": epoch,
                "lam_eff": lam_eff,
                "mse_over_ref": float(mse_loss),
                "penalty_over_ref": float(ed_log),
                "total": float(loss),
            })

            if epoch % 50 == 0 or epoch == actual_epochs - 1:
                phase = ("warmup" if epoch < warmup_end
                         else "ramp" if epoch < ramp_end
                         else "train")
                mode  = "pcgrad" if self.use_pcgrad else "std"
                msg   = (f"  [HdLossNN/{mode}] ep {epoch:>4d} [{phase}] "
                         f"loss={loss.item():.5f}  "
                         f"MSE/ref={mse_loss.item():.5f}  "
                         f"lam_eff={lam_eff:.3f}")
                if lam_eff > 0 and has_rxn:
                    msg += f"  Hd^2/ref={ed_log.item():.5f}"
                print(msg)

        self._net.eval()
        self._flush_history()

        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                self.checkpoint_dir, f"{self.target}_fold{fold_idx}_w{self.hidden[0]}_seed{self.seed}.pt")
            torch.save(self._net.state_dict(), ckpt_path)
            print(f"  [HdLossNN] saved checkpoint: {ckpt_path}")

        return self
