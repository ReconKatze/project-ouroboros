"""FallbackMamba: CPU-compatible Mamba substitute for testing conversion wiring.

This is NOT a real SSM. It's a gated linear unit with norm + residual that
matches the interface of a Mamba block. It has no temporal mixing (no recurrence).
Real Mamba kernels (mamba-ssm) require CUDA and are used in Steps 2+.

The purpose is to verify that:
1. Weight extraction from attention layers works
2. The hybrid model's forward pass completes without errors
3. Shapes and norms are correct throughout
"""

import torch
import torch.nn as nn


class FallbackMamba(nn.Module):
    """CPU-compatible gated linear unit that replaces an attention layer.

    Weight mapping from attention (v1.2 corrected):
        Q -> out_proj_c (C = output read projection)
        K -> in_proj_b  (B = gate/control signal)
        V -> in_proj_x  (X = input content)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        # V weights -> input content
        self.in_proj_x = nn.Linear(d_model, d_model, bias=False)
        # K weights -> gate/control
        self.in_proj_b = nn.Linear(d_model, d_model, bias=False)
        # Q weights -> output read
        self.out_proj_c = nn.Linear(d_model, d_model, bias=False)

    def init_from_attention(self, q_weight, k_weight, v_weight,
                            norm_weight=None, norm_bias=None):
        """Initialize FallbackMamba weights from extracted attention weights.

        Args:
            q_weight: Q projection weight [hidden, hidden] -> out_proj_c
            k_weight: K projection weight [hidden, hidden] -> in_proj_b
            v_weight: V projection weight [hidden, hidden] -> in_proj_x
            norm_weight: LayerNorm weight (optional)
            norm_bias: LayerNorm bias (optional)
        """
        with torch.no_grad():
            self.out_proj_c.weight.copy_(q_weight)
            self.in_proj_b.weight.copy_(k_weight)
            self.in_proj_x.weight.copy_(v_weight)

            if norm_weight is not None:
                self.norm.weight.copy_(norm_weight)
            if norm_bias is not None:
                self.norm.bias.copy_(norm_bias)

    def forward(self, x):
        """Forward pass with residual connection.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model] (same shape as input)
        """
        h = self.norm(x)
        gate = torch.sigmoid(self.in_proj_b(h))
        mixed = self.in_proj_x(h) * gate
        return x + self.out_proj_c(mixed)
