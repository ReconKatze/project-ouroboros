from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class LifeEquationConfig:
    vocab_size: int = 151936
    d_model: int = 1536
    n_layers_total: int = 28
    n_mamba_layers: int = 24
    attention_anchors: Tuple[int, ...] = (0, 9, 18, 27)
    n_heads: int = 48
    n_id_heads: int = 8
    d_state: int = 64
    d_mod: int = 64
    d_att: int = 48
    d_eps: int = 64
    d_hab: int = 64
    d_temp: int = 16
    d_p: int = 64
    d_narr: int = 64
    d_auto: int = 64
    d_homeo: int = 5
    d_dream: int = 32
    d_learn: int = 16
    n_purposes: int = 4
    n_epi_slots: int = 256
    d_key: int = 64
    d_val: int = 128
    warmup_steps: int = 2000
    z_cap_max: float = 1.0
    consolidation_scale: float = 0.1
    tau_eps: float = 8.0
    tau_cap: float = 16.0
    tau_hab: float = 64.0
    tau_pfat: float = 32.0
    tau_purpose: float = 16.0
    tau_temp: float = 8.0
    tau_homeo: float = 32.0
    tau_sleep: float = 64.0
    lambda_eps: float = 0.1
    lambda_eps_sleep: float = 0.25
    lambda_rest: float = 0.2
    lambda_drain: float = 0.05
    lambda_err: float = 0.05
    alpha_exp: float = 0.1
    lambda_hab: float = 0.02
    lambda_exert: float = 0.02
    lambda_recov: float = 0.1
    lambda_recov_w: float = 0.05
    theta_bored: float = 0.15
    alpha_obs: float = 0.2
    lambda_temp: float = 0.1
    lambda_temp_sleep: float = 0.2
    lambda_identity: float = 0.1
    lambda_pred: float = 0.01
    lambda_consistency: float = 0.1
    cooldown_steps: int = 32
    tau_threshold: float = 0.5
    load_threshold: float = 0.6
    episodic_surprise_threshold: float = 1.0
    theta_urg: float = 0.2
    beta_hab: float = 0.5
    attention_topk_frac: float = 0.25
    attention_rel_dim: int = 64
    sparse_from: int = 2000
    culture_dim: int = 64
    device: str = "cpu"
    dtype: str = "float32"
    locked_phase_order: Tuple[str, ...] = (
        "raw_prediction_error",
        "auxiliary_update",
        "mamba_layers",
        "post_layer_updates",
        "controller_check",
    )
    detached_paths: Tuple[str, ...] = (
        "_last_out",
        "pred_t_output",
        "z_emo_broadcast",
        "mamba_to_attention_policy",
        "z_values_reflect",   # v15: phi_reflect reads Z_cog via detached pool
        "v_future",           # v15: ViabilityModule forward estimate
    )
    warmup_modules: Tuple[str, ...] = ("attention_gain", "friction", "emotion_broadcast")
    homeostasis_set_point: Tuple[float, ...] = (1.0, 0.5, 0.0, 0.0, 1.0)
    excluded_features: Tuple[str, ...] = ("quantum_coherence", "qualia")
    trust_default: float = 1.0
    # v2 frozen reg weights — superseded by Z_values in spec v3 (kept for compat, not used in L_reg)
    alpha_eps_reg: float = 0.0
    alpha_cap_reg: float = 0.0
    alpha_bored_reg: float = 0.0
    alpha_pfat_reg: float = 0.0
    alpha_conf_reg: float = 0.0
    alpha_homeo_reg: float = 0.0
    alpha_sleep_reg: float = 0.0
    alpha_narr_reg: float = 0.0
    alpha_trust_reg: float = 0.0
    # §0.6 Autonomy Principle (spec v3 / Life Equation v15)
    # Z_values index layout:
    #   0=alpha_eps  1=alpha_cap  2=alpha_bored  3=alpha_pfat  4=alpha_c
    #   5=alpha_h    6=alpha_D    7=alpha_N       8=alpha_T         (L_reg weights)
    #   9=w_coh     10=w_drift   11=w_eps        12=w_cap    13=w_V (V_self weights)
    d_alpha: int = 14             # 9 L_reg weights + 5 V_self weights
    gamma_0: float = 1.0          # Initial identity attractor strength
    lambda_mature: float = 0.1    # Decay rate: gamma_eff = gamma_0 * exp(-lambda_mature * Z_mat)
    M_val_onset: float = 8.0      # Z_mat threshold where mu_val starts opening (~2000 lifetime steps)
    lambda_val: float = 1.0       # Sigmoid sharpness for mu_val
    tau_alpha: float = 100.0      # Value timescale (slowest deliberate layer)
    lambda_alpha: float = 0.01    # Inertial resistance: pulls Z_values back toward alpha_0
    lambda_alpha_sl: float = 0.002 # Consolidation drift rate toward alpha_0 (dreams remind)
    eps_val: float = 1e-4         # Hard positivity floor for Z_values (cannot invert values)
    theta_vol: float = 0.3        # V_self threshold below which Δ_vol becomes available
    T_vol_min: int = 100          # Steps V_self must stay below theta_vol before Δ_vol enabled
    M_vol_min: float = 2.0        # Minimum Z_mat for voluntary death (must be mature enough)
    # L_reg scale: homeostatic terms are O(100-200) vs KL O(0.3-1.1).
    # Without scaling, L_reg dominates gradients and drowns the distillation signal.
    lambda_reg: float = 0.01
    model_hash_seed: Tuple[int, ...] = field(default_factory=lambda: (28, 24, 48, 64, 14))

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mamba_layer_indices(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.n_layers_total) if i not in self.attention_anchors)
