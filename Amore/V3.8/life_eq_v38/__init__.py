from .config import LifeEquationConfig, VariantProfile
from .factory import build_config, build_model
from .variant_profiles import VARIANT_PROFILES, get_variant_profile

__all__ = [
    "LifeEquationConfig",
    "VariantProfile",
    "VARIANT_PROFILES",
    "build_config",
    "build_model",
    "get_variant_profile",
]

try:
    from .forensics import ForensicConfig, ForensicEventManager
    from .identity_snapshot import (
        ANCHOR_CATEGORIES,
        ANCHOR_CORPUS,
        apply_identity_snapshot,
        build_anchor_corpus,
        snapshot_identity,
    )
    from .model import LifeEquationModel
    from .modules import SelfDynamicsModel
    from .persistence import StateStore
    from .state import FullState, ManifestEntry, zero_state

    __all__.extend(
        [
            "ANCHOR_CATEGORIES",
            "ANCHOR_CORPUS",
            "apply_identity_snapshot",
            "build_anchor_corpus",
            "ForensicConfig",
            "ForensicEventManager",
            "FullState",
            "LifeEquationModel",
            "ManifestEntry",
            "SelfDynamicsModel",
            "snapshot_identity",
            "StateStore",
            "zero_state",
        ]
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
