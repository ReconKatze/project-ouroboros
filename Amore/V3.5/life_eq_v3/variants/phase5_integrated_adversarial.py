from __future__ import annotations

from typing import Optional

from ..config import LifeEquationConfig
from ..factory import build_model

VARIANT_NAME = "phase5_integrated_adversarial"


def build(config: Optional[LifeEquationConfig] = None):
    return build_model(VARIANT_NAME, config)
