import os
import sys

import torch

sys.path.insert(0, os.path.abspath("/run/media/deck/Katze2/ProjectOuroboros/Amore/V2"))

from life_eq_v2 import LifeEquationConfig, LifeEquationModel, StateStore
from life_eq_v2.spec_check import validate_locked_conventions


def test_locked_conventions_hold():
    result = validate_locked_conventions(LifeEquationConfig())
    assert result.passed, result.messages


def test_forward_phase_order_and_shapes():
    model = LifeEquationModel(LifeEquationConfig(vocab_size=512, device="cpu"))
    x = torch.randint(0, 512, (2, 4))
    out = model(x, step=10)
    assert out.phase_trace == list(model.config.locked_phase_order)
    assert out.logits.shape == (2, 512)
    assert out.state.Z_cog.shape == (2, model.config.n_mamba_layers, model.config.n_heads, model.config.d_state)


def test_persistence_round_trip():
    config = LifeEquationConfig(vocab_size=512, device="cpu")
    model = LifeEquationModel(config)
    store = StateStore(config)
    state = model.init_state(batch_size=1)
    state_id = store.save_state(state, metadata={"tags": ["test"], "files": ["V2"]})
    loaded = store.load_state(state_id)
    assert loaded.Z_cog.shape == state.Z_cog.shape
    assert len(loaded.manifest) == len(state.manifest)
