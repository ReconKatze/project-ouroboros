"""Real Mamba block using mamba-ssm (CUDA required).

Used on Colab T4 (Steps 2-4). Falls back to FallbackMamba on CPU.
"""

import torch
import torch.nn as nn


class RealMambaBlock(nn.Module):
    """Wraps mamba-ssm's Mamba layer for use in the hybrid model.

    Requires CUDA + mamba-ssm package. Install with:
        pip install mamba-ssm --no-build-isolation

    The mamba-ssm Mamba class handles its own input projection and
    output projection internally, so this wrapper is minimal.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=d_model, d_state=d_state)

    def forward(self, x):
        """Forward pass through Mamba SSM.

        Args:
            x: [batch, seq_len, d_model] — already normed by the layer

        Returns:
            [batch, seq_len, d_model] — pre-residual output
        """
        return self.mamba(x)


def create_mamba_block(d_model: int, d_state: int = 16, device: str = "cpu") -> nn.Module:
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
