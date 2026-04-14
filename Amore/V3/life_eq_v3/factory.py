from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .config import LifeEquationConfig
from .model import LifeEquationModel
from .variant_profiles import get_variant_profile


def build_config(variant_name: str, config: Optional[LifeEquationConfig] = None) -> LifeEquationConfig:
    base = LifeEquationConfig() if config is None else config
    return replace(base, variant_profile=get_variant_profile(variant_name))


def build_model(variant_name: str, config: Optional[LifeEquationConfig] = None) -> LifeEquationModel:
    return LifeEquationModel(build_config(variant_name, config))
