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


class BetaGated2AMamba(nn.Module):
    """Round 2A: BetaGatedMamba + per-token and per-group input gates.

    Measures whether the model wants to selectively suppress Mamba input
    on a per-token (g_time) and per-feature-group (g_block) basis.

    Gate design
    -----------
    g_time  [B, L, 1]        per-token scalar  — global write suppression
    g_block [B, L, n_blocks] per-group gate    — structured feature sparsity
    combined = g_time * expand(g_block) applied to input x before Mamba

    n_blocks = d_state // 16.  For d_state=64: n_blocks=4, block_size = d_model//4.

    Sparsity regularisation (prevents gates from staying at 0.5):
        λ_t * g_time.mean() + λ_b * g_block.mean()   (λ_t=0.005, λ_b=0.01)

    Go/no-go signal: ≥30% tokens silenced + clear block structure + depth-dependent
    usage → proceed to multi-head SSM routing.  Everything on or everything off →
    state capacity not a learnable prior at this scale.

    sparsity_accum and stats_accum are mutable lists owned by the parent
    HybridChimeraModel.  The training loop reads sparsity_accum each step;
    stats_accum is read and printed at each val step.
    """

    _LAMBDA_T = 0.005   # temporal gate sparsity weight
    _LAMBDA_B = 0.01    # block gate sparsity weight

    def __init__(
        self,
        mamba_block: nn.Module,
        mamba_idx: int,
        total_mamba: int,
        d_model: int,
        d_state: int,
        sparsity_accum: list,
        stats_accum: list,
        shared_beta: "nn.Parameter | None" = None,
    ):
        super().__init__()
        self.mamba_block = mamba_block
        self.depth = mamba_idx / max(total_mamba - 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        object.__setattr__(self, "_shared_beta", shared_beta)

        n_blocks   = max(1, d_state // 16)
        block_size = d_model // n_blocks
        self._n_blocks   = n_blocks
        self._d_model    = d_model
        self._block_size = block_size

        self.W_t = nn.Linear(d_model, 1,        bias=True)
        self.W_b = nn.Linear(d_model, n_blocks, bias=True)
        # Init open: sigmoid(+2) ≈ 0.88 — starts near the ungated baseline.
        nn.init.zeros_(self.W_t.weight)
        nn.init.constant_(self.W_t.bias, 2.0)
        nn.init.zeros_(self.W_b.weight)
        nn.init.constant_(self.W_b.bias, 2.0)

        self._sparsity_accum = sparsity_accum
        self._stats_accum    = stats_accum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g_time  = torch.sigmoid(self.W_t(x))          # [B, L, 1]
        g_block = torch.sigmoid(self.W_b(x))          # [B, L, n_blocks]

        # Expand g_block: each of n_blocks values covers block_size input features
        g_exp = g_block.repeat_interleave(self._block_size, dim=-1)
        g_exp = g_exp[..., :self._d_model]             # trim if d_model % n_blocks != 0

        out = self.mamba_block(g_time * g_exp * x)

        if self._shared_beta is not None:
            alpha = torch.sigmoid(self._shared_beta * self.depth + self.gamma)
        else:
            alpha = torch.sigmoid(self.gamma)

        if self.training:
            self._sparsity_accum.append(
                self._LAMBDA_T * g_time.mean() + self._LAMBDA_B * g_block.mean()
            )
            self._stats_accum.append({
                "g_time_mean":    g_time.detach().mean().item(),
                "g_time_silence": (g_time.detach() < 0.1).float().mean().item(),
            })

        return alpha * out

    def init_from_attention(self, **kwargs) -> None:
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
