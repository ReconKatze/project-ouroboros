"""Real Mamba block using mamba-ssm (CUDA required).

Used on Colab A100 (Steps 2-4). Falls back to FallbackMamba on CPU.
Uses Mamba-3 (mamba-ssm >= 2.3.1): complex-valued states via RoPE,
no d_state ceiling, MIMO mode available via is_mimo=True.
"""

import torch
import torch.nn as nn


class RealMambaBlock(nn.Module):
    """Wraps mamba-ssm's Mamba3 layer for use in the hybrid model.

    Requires CUDA + mamba-ssm >= 2.3.1. Install with:
        pip install mamba-ssm --no-build-isolation

    Mamba-3 uses RoPE-based complex state updates instead of d_conv.
    No hard d_state ceiling (Mamba-1 was capped at 256 by CUDA kernel).
    Mamba3.forward() signature: forward(u, seq_idx, cu_seqlens, inference_params)
    — calling self.mamba(x) passes x as u with all optional args defaulting.
    """

    def __init__(self, d_model: int, d_state: int = 64, headdim: int = 64):
        super().__init__()
        from mamba_ssm import Mamba3
        # Mamba3 constraint: headdim_angles = (d_state * rope_fraction) // 2 <= headdim // 2
        # Clamp rope_fraction so headdim_angles stays at headdim//2 for any d_state.
        # d_state <= headdim: rope_fraction stays at 0.5 (default, full oscillatory coverage).
        # d_state >  headdim: rope_fraction scales down, keeping absolute oscillatory count fixed.
        rope_fraction = min(0.5, float(headdim) / d_state)
        self.mamba = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim,
                            rope_fraction=rope_fraction)

    def forward(self, x):
        """Forward pass through Mamba SSM.

        Args:
            x: [batch, seq_len, d_model] — already normed by the layer

        Returns:
            [batch, seq_len, d_model] — pre-residual output
        """
        return self.mamba(x)


def create_mamba_block(d_model: int, d_state: int = 64, device: str = "cpu") -> nn.Module:
    """Factory: returns RealMambaBlock on CUDA or FallbackMamba on CPU.

    Args:
        d_model: Model hidden dimension.
        d_state: SSM state dimension.
        device: Target device string ('cpu', 'cuda', 'cuda:0', etc.).

    Returns:
        RealMambaBlock if CUDA available and device is not CPU,
        else FallbackMamba.
    """
    use_cuda = device != "cpu" and torch.cuda.is_available()

    if use_cuda:
        try:
            return RealMambaBlock(d_model, d_state)
        except ImportError:
            print("WARNING: mamba-ssm not installed. Falling back to FallbackMamba.")
            print("  Install with: pip install mamba-ssm --no-build-isolation")

    from chimera.models.fallback_mamba import FallbackMamba
    return FallbackMamba(d_model)
