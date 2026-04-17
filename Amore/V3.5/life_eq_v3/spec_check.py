from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import LifeEquationConfig


@dataclass(frozen=True)
class SpecCheckResult:
    passed: bool
    messages: List[str]


def validate_locked_conventions(config: LifeEquationConfig) -> SpecCheckResult:
    messages: List[str] = []
    if config.locked_phase_order != (
        "raw_prediction_error",
        "auxiliary_update",
        "mamba_layers",
        "post_layer_updates",
        "controller_check",
    ):
        messages.append("Locked phase ordering diverges from section 29.")
    if "pred_t_output" not in config.detached_paths:
        messages.append("Detached output prediction path missing from section 0.5 policy.")
    if tuple(config.attention_anchors) != (0, 10, 21, 31):
        # V3.5: 32-layer architecture re-spaces anchors from (0,9,18,27) to (0,10,21,31).
        messages.append("Attention anchors diverge from section 0 (V3.5 expects (0, 10, 21, 31)).")
    if tuple(config.warmup_modules) != ("attention_gain", "friction", "emotion_broadcast"):
        messages.append("Warmup modules diverge from section 0.5 convention 3.")
    # v15 / spec v3 autonomy checks (§0.6)
    if not hasattr(config, "lambda_mature"):
        messages.append("v15: lambda_mature missing — identity emancipation not configured (§2 v15).")
    if not hasattr(config, "d_alpha") or config.d_alpha < 14:
        messages.append("v15: d_alpha must be >= 14 (9 L_reg weights + 5 V_self weights) (§26 v15).")
    if not hasattr(config, "theta_vol"):
        messages.append("v15: theta_vol missing — voluntary death threshold not configured (§27 v15).")
    if "z_values_reflect" not in config.detached_paths:
        messages.append("v15: z_values_reflect missing from detached_paths — §0.5 Convention 2 violated.")
    if "v_future" not in config.detached_paths:
        messages.append("v15: v_future missing from detached_paths — §0.5 Convention 2 violated.")
    return SpecCheckResult(passed=not messages, messages=messages)
