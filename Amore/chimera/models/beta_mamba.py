"""Learnable output gate for Mamba blocks — constant and shared-β depth variants."""

import torch
import torch.nn as nn


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
