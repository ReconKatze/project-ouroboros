import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath("/run/media/deck/Katze2/ProjectOuroboros/Amore/V2"))

from life_eq_v2 import LifeEquationConfig, LifeEquationModel, StateStore
from life_eq_v2.spec_check import validate_locked_conventions


def test_locked_conventions_hold():
    result = validate_locked_conventions(LifeEquationConfig())
    assert result.passed, result.messages


# Mamba3Block requires CUDA (mamba-ssm git source, A100 target).
# These tests run on Colab; skip on CPU dev machines.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba3Block requires CUDA (mamba-ssm)"
)


@requires_cuda
def test_forward_phase_order_and_shapes():
    device = "cuda"
    # Use smallest viable config to keep VRAM low in the test
    config = LifeEquationConfig(
        vocab_size=512,
        d_model=128,
        n_layers_total=4,
        n_mamba_layers=2,
        attention_anchors=(0, 3),
        n_heads=4,
        n_id_heads=2,
        d_state=16,
        device=device,
    )
    model = LifeEquationModel(config).to(device)
    x = torch.randint(0, 512, (2, 4), device=device)
    out = model(x, step=10)
    assert out.phase_trace == list(model.config.locked_phase_order)
    assert out.logits.shape == (2, 512)
    # Z_cog: real [B, n_mamba_layers, d_model] (changed from complex in Mamba-3 redesign)
    assert out.state.Z_cog.shape == (2, model.config.n_mamba_layers, model.config.d_model)
    assert out.state.Z_cog.dtype == torch.float32


@requires_cuda
def test_persistence_round_trip():
    device = "cuda"
    config = LifeEquationConfig(vocab_size=512, device=device)
    model = LifeEquationModel(config).to(device)
    store = StateStore(config)
    state = model.init_state(batch_size=1)
    state_id = store.save_state(state, metadata={"tags": ["test"], "files": ["V2"]})
    loaded = store.load_state(state_id)
    assert loaded.Z_cog.shape == state.Z_cog.shape
    assert len(loaded.manifest) == len(state.manifest)


def test_state_shapes_cpu():
    """Verify state tensor shapes are correct without running the full model."""
    from life_eq_v2.state import zero_state
    config = LifeEquationConfig(vocab_size=512, device="cpu")
    state = zero_state(config, batch_size=2)
    # Z_cog: real [B, n_mamba_layers, d_model]
    assert state.Z_cog.shape == (2, config.n_mamba_layers, config.d_model)
    assert state.Z_cog.dtype == torch.float32
    # Z_id / I_0: real [B, n_id_heads, d_model]
    assert state.Z_id.shape == (2, config.n_id_heads, config.d_model)
    assert state.I_0.shape == (2, config.n_id_heads, config.d_model)
    # pool_complex_state: [B, 2]
    from life_eq_v2.state import pool_complex_state
    pool = pool_complex_state(state.Z_cog)
    assert pool.shape == (2, 2)
