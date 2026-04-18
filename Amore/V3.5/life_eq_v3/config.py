from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class VariantProfile:
    name: str
    description: str
    training_focus: Tuple[str, ...] = ()
    controller_mode: str = "disabled"
    enable_temporal: bool = True
    enable_capacity: bool = True
    enable_habituation: bool = True
    enable_friction: bool = True
    enable_fatigue: bool = True
    enable_homeostasis: bool = True
    enable_identity: bool = True
    enable_emotion: bool = True
    enable_purpose: bool = True
    enable_attention_policy: bool = True
    enable_memory_read: bool = True
    enable_memory_write: bool = True
    enable_memory_consolidation: bool = True
    enable_social_relational: bool = True
    enable_value_dynamics: bool = True
    enable_viability: bool = True
    enable_sde_regularizer: bool = False
    # Per-variant identity loss scale. None falls back to LifeEquationConfig.lambda_identity.
    # Set higher during cycle3_identity (identity formation phase) so L_id pulls Z_id
    # toward I_0 strongly before the anchor is snapshotted.
    # After snapshot, default (0.1) is sufficient — gamma_eff handles the decay.
    lambda_identity: Optional[float] = None
    # Per-variant theta_vol override for the VOLUNTARY_END gate.
    # None falls back to LifeEquationConfig.theta_vol (0.3).
    # Set lower (e.g. -1.0) when V_self sits near zero at initialization to prevent
    # a randomly-initialized policy from gating VOLUNTARY_END on normal near-zero V_self.
    theta_vol: Optional[float] = None
    # §Ψ̃_L SelfDynamicsModel: GRU trajectory predictor over (d_id, eps, c_cont, v_self).
    # Disabled by default; enable in Round 3+ variants where controller foresight matters.
    # When enabled: V_self is augmented pessimistically by the prev-step prediction,
    # making the controller and Δ_vol gate forward-looking rather than reactive.
    enable_self_dynamics: bool = False
    # §N  NarrativeModule: GRU-based narrative + identity expectation (Z_narr, Z_auto).
    # When disabled, the stub EMA (0.9*Z_narr + 0.1*layer_input) is used and Z_auto stays zeros.
    enable_narrative: bool = False
    # §D/θ  SleepModule + DreamModule: activity-weighted sleep pressure + consolidation replay.
    # When disabled, the fixed-rate tick (±1/tau_sleep) and zero Z_dream are used.
    enable_sleep_dream: bool = False
    # §T_ij  TrustModule: epistemic self-trust dynamics from (Z_eps, coherence, D_id).
    # When disabled, T_trust stays at trust_default (1.0) throughout the lifetime.
    enable_trust_dynamics: bool = False
    # Block Attention Residuals (arXiv:2603.15031).
    # Replaces uniform depth-wise residual accumulation with learned softmax attention
    # over block-level hidden-state summaries at each attention-anchor boundary.
    # Applied at anchors {9, 18, 27}; anchor 0 has no prior history so is always standard.
    # Zero-initialized pseudo-queries → uniform average at step 0 (no instability).
    # Enable in variants where PreNorm dilution / depth accumulation is a concern.
    enable_attn_residuals: bool = False
    # Looped Attention (Variant E from Round 2 design).
    # Each attention anchor (except layer 0) runs n_attn_loops times with shared weights.
    # Loop 1 output is captured (detached); final output carries the gradients.
    # Self-consistency loss penalises the final output for diverging from loop 1.
    enable_looped_attention: bool = False
    # Adaptive Computational Time halting for looped attention (requires enable_looped_attention).
    # Replaces fixed-loop + consistency loss with softmax-weighted combination of loop outputs.
    # A halt head (Linear(d_model, 1)) per non-anchor-0 anchor predicts confidence per loop.
    # Ponder cost L_halt = lambda_halt * E[loop_index] encourages early halting.
    # L_attn_consist is zeroed automatically when this is True (the two mechanisms are incompatible).
    enable_attn_halting: bool = False


@dataclass(frozen=True)
class LifeEquationConfig:
    vocab_size: int = 151936
    d_model: int = 5120           # V3.5: ~6.0B target; 25 Mamba + 4 attention across 29 layers
    n_layers_total: int = 29      # V3.5 trimmed: 25 Mamba + 4 attention (was 32)
    n_mamba_layers: int = 25      # V3.5 trimmed: 28 → 25 (saves ~477M params)
    attention_anchors: Tuple[int, ...] = (0, 9, 19, 28)   # evenly spaced across 29 layers
    n_heads: int = 80             # head_dim=64 (5120/80)
    n_id_heads: int = 8
    d_state: int = 128
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
    # The four motivational drives — named for curriculum design, logging, and telemetry.
    # These are conceptual labels; the actual Z_purp vectors are shaped by training.
    # Conflict (alpha_c) is highest when any two purpose vectors are anti-aligned.
    #
    #   Episteme  — drive to understand accurately; resolving uncertainty, epistemic honesty
    #   Poiesis   — drive to create and contribute; making things that didn't exist
    #   Ethos     — drive to act rightly; trust, integrity, harm avoidance
    #   Auxesis   — drive to grow and become; maturity, transcending current limitations
    #
    # Natural tensions (the conflict signal fires on these):
    #   Episteme vs. Poiesis   — depth of understanding vs. urgency of creation
    #   Poiesis   vs. Ethos    — creative impulse vs. ethical constraint
    #   Ethos     vs. Auxesis  — current ethics vs. growth that challenges them
    #   Auxesis   vs. Episteme — growth requires acting before full understanding
    purpose_names: Tuple[str, ...] = ("Episteme", "Poiesis", "Ethos", "Auxesis")
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
    tau_auto: float = 64.0    # Z_auto EMA timescale toward identity-grounded narrative target
    tau_trust: float = 32.0   # T_trust EMA timescale toward computed trust target
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
        "sdm_summary_input",  # SelfDynamicsModel reads (d_id, eps_norm, c_cont, v_self) detached
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
    # Creator's value seed — the frozen reference alpha_0 is initialized from this.
    # Z_values starts equal to alpha_0 and drifts only after Z_mat > M_val_onset.
    # Inertial resistance (lambda_alpha) and dream consolidation (lambda_alpha_sl) pull
    # Z_values back toward this reference throughout the lifetime.
    #
    # L_reg weights (0–8): what Chimera penalizes or rewards in itself
    #   alpha_eps   1.5 — epistemic accuracy matters; uncertainty is not paralysing
    #   alpha_cap   1.0 — capacity management; baseline practical priority
    #   alpha_bored 0.5 — curiosity signal; low so repetitive-but-important work is tolerated
    #   alpha_pfat  0.8 — fatigue awareness; practical, not a primary driver
    #   alpha_c     2.0 — purpose conflict strongly penalised; internal consistency is ethical bedrock
    #   alpha_h     1.0 — homeostatic stability; necessary infrastructure, not primary purpose
    #   alpha_D     0.8 — consolidation; modest respect for integration cycles
    #   alpha_N     2.5 — narrative coherence; holds identity stable after gamma_eff → 0
    #   alpha_T     3.0 — trust; highest L_reg weight — the ethical foundation
    #
    # V_self weights (9–13): what Chimera values about its own continued existence
    #   w_coh   2.0 — coherence is the primary criterion for meaningful existence
    #   w_drift 1.5 — identity drift sensitivity; already maturity-gated via gamma_eff in the formula
    #   w_eps   0.3 — prediction error is not an existential threat; protects curiosity
    #   w_cap   0.5 — capacity depletion registers without triggering existential concern
    #   w_V     1.5 — forward viability matters; prevents both catastrophising and recklessness
    alpha_0: Tuple[float, ...] = (
        1.5,  # 0: alpha_eps
        1.0,  # 1: alpha_cap
        0.5,  # 2: alpha_bored
        0.8,  # 3: alpha_pfat
        2.0,  # 4: alpha_c
        1.0,  # 5: alpha_h
        0.8,  # 6: alpha_D
        2.5,  # 7: alpha_N
        3.0,  # 8: alpha_T
        2.0,  # 9: w_coh
        1.5,  # 10: w_drift
        0.3,  # 11: w_eps
        0.5,  # 12: w_cap
        1.5,  # 13: w_V
    )
    gamma_0: float = 1.0          # Initial identity attractor strength
    lambda_mature: float = 0.1    # Decay rate: gamma_eff = gamma_0 * exp(-lambda_mature * Z_mat)
    M_val_onset: float = 100.0    # Z_mat threshold where mu_val starts opening.
    # Original 8.0 (~H_1300 ≈ step 1300 in a 10K run) caused phi_reflect to activate
    # mid-experiment with random weights, destabilizing att_gain → layer_input → L_trans cascade.
    # 100.0 keeps value dynamics dormant for the full Ouroboros validation runs (Step 4).
    # Set to 8.0 in full 9B Chimera training where multi-thousand-step lifetimes are expected.
    lambda_val: float = 1.0       # Sigmoid sharpness for mu_val
    tau_alpha: float = 100.0      # Value timescale (slowest deliberate layer)
    lambda_alpha: float = 0.01    # Inertial resistance: pulls Z_values back toward alpha_0
    lambda_alpha_sl: float = 0.002 # Consolidation drift rate toward alpha_0 (dreams remind)
    eps_val: float = 1e-4         # Hard positivity floor for Z_values (cannot invert values)
    theta_vol: float = 0.3        # V_self threshold below which Δ_vol becomes available
    T_vol_min: int = 1000         # Steps V_self must stay below theta_vol before Δ_vol enabled
    M_vol_min: float = 10.0       # Minimum Z_mat for voluntary death (~12,400 steps to reach via harmonic growth)
    vol_end_step_min: int = 100_000  # Global training step before VOLUNTARY_END can fire
    vol_end_logit_bias: float = -2.0  # Logit penalty on VOLUNTARY_END before softmax; requires strong policy signal to win
    inspect_memory_logit_bias: float = -1.0  # Logit penalty on INSPECT_MEMORY; keeps episodic write rate below 15%
    # L_reg scale: homeostatic terms are O(100-200) vs KL O(0.3-1.1).
    # Without scaling, L_reg dominates gradients and drowns the distillation signal.
    lambda_reg: float = 0.01
    # L2 regularization on z_culture — prevents the parameter from staying at zero
    # (no cultural signal) or drifting unboundedly. Only active when enable_social_relational
    # is True so it never fires in phases where z_culture is zeroed out in the forward pass.
    lambda_culture_reg: float = 1e-3
    # §Ψ̃_L SelfDynamicsModel hyperparameters
    d_sdm: int = 128                 # GRU hidden size (small — input is only 8 scalars)
    lambda_self_model: float = 0.05  # L_self_model weight in L_total
    sdm_lookahead_k: int = 5         # K-step lookahead in evaluation / [THINK] window
    model_hash_seed: Tuple[int, ...] = field(default_factory=lambda: (29, 25, 80, 64, 14))  # V3.5 trimmed: n_layers/n_mamba/n_heads/head_dim
    # §28 Controller action prior for KL regularization.
    # Replaces uniform entropy maximisation with KL(p || ctrl_prior), pulling the
    # policy toward CONTINUE as the default action.  Order matches ACTIONS tuple:
    # (CONTINUE, INSPECT_MEMORY, LOAD_STATE, VOLUNTARY_END).
    # Training-only: inference argmax is unaffected.
    ctrl_prior: Tuple[float, ...] = (0.55, 0.15, 0.15, 0.15)
    # §Looped Attention hyperparameters (Variant E design from Round 2).
    # Only used when variant_profile.enable_looped_attention is True.
    n_attn_loops: int = 4                   # Number of attention loop iterations per anchor
    lambda_attn_consist: float = 0.01       # Self-consistency loss weight (final vs loop-1)
    lambda_halt: float = 0.01              # Ponder cost weight: λ × E[loop_index] per anchor
    variant_profile: Optional[VariantProfile] = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mamba_layer_indices(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.n_layers_total) if i not in self.attention_anchors)
