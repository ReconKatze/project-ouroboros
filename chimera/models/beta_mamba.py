"""Depth-aware learnable output gate for Mamba blocks (Step 4, Variant D)."""

import torch
import torch.nn as nn


class BetaGatedMamba(nn.Module):
    """Wraps a Mamba block with a depth-aware learnable output gate.

    Gate:   alpha = sigmoid(beta * depth + gamma)
    Output: alpha * mamba_out   (the transformer layer adds residual outside)

    depth = mamba_layer_idx / (total_mamba_layers - 1)  in [0.0, 1.0]

    Initialized: beta=0, gamma=0 → alpha=0.5 at all depths.
    After training, depth-varying alpha indicates the model has learned
    different memory retention rates at different network depths.

    Used only in Variant D of the Step 4 d_state gradient experiment.
    """

    def __init__(self, mamba_block: nn.Module, mamba_idx: int, total_mamba: int):
        """
        Args:
            mamba_block: A RealMambaBlock (or FallbackMamba) instance.
            mamba_idx: Index of this Mamba block among all Mamba layers (0-based).
            total_mamba: Total number of Mamba layers in the model.
        """
        super().__init__()
        self.mamba_block = mamba_block
        self.depth = mamba_idx / max(total_mamba - 1, 1)
        self.beta  = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model] — already normed by the transformer layer.

        Returns:
            [batch, seq_len, d_model] — gated Mamba output (pre-residual).
        """
        out   = self.mamba_block(x)
        alpha = torch.sigmoid(self.beta * self.depth + self.gamma)
        return alpha * out

    def init_from_attention(self, **kwargs):
        """Delegate weight initialisation to the inner Mamba block if supported."""
        if hasattr(self.mamba_block, "init_from_attention"):
            self.mamba_block.init_from_attention(**kwargs)
