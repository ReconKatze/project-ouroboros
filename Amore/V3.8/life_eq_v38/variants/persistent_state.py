from __future__ import annotations

from typing import Optional

from ..config import LifeEquationConfig
from ..factory import build_model

VARIANT_NAME = "persistent_state"


def build(config: Optional[LifeEquationConfig] = None):
    return build_model(VARIANT_NAME, config)
