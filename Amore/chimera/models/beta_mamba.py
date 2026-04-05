"""Learnable output gate for Mamba blocks — constant and shared-β depth variants.
Also provides PSoftMambaWrapper for Level 1 Predictive Coding (P_soft).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaGatedMamba(nn.Module):
    """Wraps a Mamba block with a learnable output gate.

    Two modes controlled by shared_beta:

      shared_beta=None  →  gate = sigmoid(gamma_i)
          Pure per-layer constant. No depth dependence (D_constant variant).

      shared_beta=nn.Parameter  →  gate = sigmoid(shared_beta * depth_i + gamma_i)
          Depth-aware gate. ONE shared β is owned by the parent HybridChimeraModel
          and referenced here without re-registering, so gradients accumulate once
          rather than 18× (D_proper variant).

    Output: alpha * mamba_out  (the transformer layer adds the residual outside)

    Why shared_beta is not stored as self.shared_beta
    --------------------------------------------------
    PyTorch's Module.__setattr__ intercepts any nn.Parameter assignment and calls
    register_parameter, even for names with underscores. Using object.__setattr__
    bypasses this so the parameter is registered exactly once, on the parent model.
    """

    def __init__(
        self,
        mamba_block: nn.Module,
        mamba_idx: int,
        total_mamba: int,
        shared_beta: nn.Parameter | None = None,
    ):
        """
        Args:
            mamba_block:  A RealMambaBlock (or FallbackMamba) instance.
            mamba_idx:    Index of this Mamba block among all Mamba layers (0-based).
            total_mamba:  Total number of Mamba layers in the model.
            shared_beta:  Single nn.Parameter shared across all BetaGatedMamba
                          instances and registered on the parent model.
                          Pass None for the constant-gate (D_constant) variant.
        """
        super().__init__()
        self.mamba_block = mamba_block
        self.depth = mamba_idx / max(total_mamba - 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # Bypass Module.__setattr__ — ownership belongs to the parent model.
        object.__setattr__(self, "_shared_beta", shared_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] — already normed by the transformer layer.
        Returns:
            [batch, seq_len, d_model] — gated Mamba output (pre-residual).
        """
        out = self.mamba_block(x)
        if self._shared_beta is not None:
            alpha = torch.sigmoid(self._shared_beta * self.depth + self.gamma)
        else:
            alpha = torch.sigmoid(self.gamma)
        return alpha * out

    def init_from_attention(self, **kwargs) -> None:
        """Delegate weight initialisation to the inner Mamba block if supported."""
        if hasattr(self.mamba_block, "init_from_attention"):
            self.mamba_block.init_from_attention(**kwargs)


class PSoftMambaWrapper(nn.Module):
    """Level 1 Predictive Coding (P_soft) wrapper around a BetaGatedMamba block.

    Changes the Mamba state update from raw-input-driven to error-driven:

        Standard:  h_t = A * h_{t-1}  +  B @ x_t
        P_soft:    h_t = A * h_{t-1}  +  B @ (x_t - pred_t)
        where      pred_t = W_pred @ x_{t-1}   (causal, approximates C @ h_{t-1})

    Output = inner(error) + pred  — correction plus prediction.

    The state no longer encodes "compressed past" — it encodes "prediction error
    over the domain model", which is a fundamentally different representation
    that CANNOT be recovered by fine-tuning a standard Mamba checkpoint.

    Gradient flow (intentionally decoupled):
    - pred_proj trains via loss_pc (self-supervised prediction objective) only.
    - inner (BetaGatedMamba → Mamba) trains via distillation loss on the error signal.
    - Detaching pred in both the error and output paths keeps the two objectives
      from interfering: Mamba cannot collapse error to zero by gaming pred_proj.

    loss_accum: shared mutable list owned by the student model. Each forward call
    appends one scalar loss term. The training loop clears it before each step.
    """

    def __init__(self, inner: nn.Module, d_model: int, loss_accum: list):
        super().__init__()
        self.inner       = inner          # BetaGatedMamba instance
        self.pred_proj   = nn.Linear(d_model, d_model, bias=False)
        self._loss_accum = loss_accum     # reference to student.psoft_loss_accum
        # Identity init: pred_t ≈ x_{t-1} on day zero — stable training start.
        nn.init.eye_(self.pred_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model] — already normed by the transformer layer.
        Returns:
            [B, L, d_model] — error-corrected output.
        """
        # Causal prediction: pred[t] = W @ x[t-1], pred[0] = 0
        pred = torch.zeros_like(x)
        if x.shape[1] > 1:
            pred[:, 1:] = self.pred_proj(x[:, :-1])

        # Error = residual the model failed to predict
        error = x - pred.detach()

        # Mamba integrates error into its state (not raw input)
        out = self.inner(error)

        # Prediction loss: pred[t] should anticipate x[t] — self-supervised signal.
        # Only accumulated during training; detach x so gradients go to pred_proj only.
        if self.training and x.shape[1] > 1:
            self._loss_accum.append(F.mse_loss(pred[:, 1:], x[:, 1:].detach()))

        # Output = Mamba correction + detached prediction
        return out + pred.detach()

    # --- Delegate attribute access needed by train_variant gate reporting ---

    @property
    def gamma(self):
        return self.inner.gamma

    @property
    def depth(self):
        return self.inner.depth

    def init_from_attention(self, **kwargs) -> None:
        if hasattr(self.inner, "init_from_attention"):
            self.inner.init_from_attention(**kwargs)
