from __future__ import annotations

from typing import Dict

from .config import VariantProfile


def _profile(name: str, description: str, **kwargs: object) -> VariantProfile:
    return VariantProfile(name=name, description=description, **kwargs)


VARIANT_PROFILES: Dict[str, VariantProfile] = {
    "base_model": _profile(
        "base_model",
        "Plain predictive-coding backbone before persistence, reflection, controller, or memory.",
        training_focus=("competence", "instruction_following", "long_context"),
        controller_mode="disabled",
        enable_identity=False,
        enable_emotion=False,
        enable_purpose=False,
        enable_attention_policy=False,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
    ),
    "persistent_state": _profile(
        "persistent_state",
        "Continuity-focused build with persistent auxiliary state but no identity reflection or controller.",
        training_focus=("state_carryover", "resume_after_break", "continuity"),
        controller_mode="disabled",
        enable_identity=False,
        enable_emotion=False,
        enable_purpose=False,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
    ),
    "identity_persistence": _profile(
        "identity_persistence",
        "Adds identity persistence and contradiction recovery before controller or episodic memory.",
        training_focus=("identity_drift", "trait_consistency", "goal_continuity"),
        controller_mode="disabled",
        enable_identity=True,
        enable_emotion=True,
        enable_purpose=True,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
    ),
    "controller_passive": _profile(
        "controller_passive",
        "Passive controller scoring only; no interventions are applied live.",
        training_focus=("continuation_confidence", "surprise_scoring", "identity_drift_scoring"),
        controller_mode="passive",
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
    ),
    "controller_offline": _profile(
        "controller_offline",
        "Offline intervention selection; action is predicted but never executed in forward.",
        training_focus=("intervention_selection", "retrieve_vs_continue", "reload_policy"),
        controller_mode="offline",
        enable_memory_read=True,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
    ),
    "controller_live": _profile(
        "controller_live",
        "Live gated controller with reload and voluntary-end behavior enabled.",
        training_focus=("gated_control", "anti_thrashing", "live_intervention"),
        controller_mode="live",
        enable_memory_read=True,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
    ),
    "memory_retrieval": _profile(
        "memory_retrieval",
        "Retrieval-first memory stage; read path active before selective writes.",
        training_focus=("retrieval_usefulness", "solve_with_memory", "reuse"),
        controller_mode="offline",
        enable_memory_read=True,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
    ),
    "memory_write": _profile(
        "memory_write",
        "Selective episodic write stage with retrieval active and surprise-gated commits.",
        training_focus=("write_selection", "write_quality", "surprise_gating"),
        controller_mode="offline",
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=False,
        enable_social_relational=False,
    ),
    "memory_consolidation": _profile(
        "memory_consolidation",
        "Memory summarization and consolidation stage.",
        training_focus=("consolidation", "summarization", "identity_relevant_memory"),
        controller_mode="offline",
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=False,
    ),
    "social_relational": _profile(
        "social_relational",
        "Late-stage social and relational variant with trust, bonds, and multi-agent context active. "
        "Training data: see cultural_corpus.CULTURAL_CORPUS_SPEC — all six categories apply, "
        "with long_term_relationship_dynamics and constructive_disagreement weighted highest.",
        training_focus=("distinct_interlocutors", "trust_updates", "relational_continuity"),
        controller_mode="live",
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=True,
    ),
    "phase0_base": _profile(
        "phase0_base",
        "Equation Phase 0: base LM substrate.",
        training_focus=("base_lm", "substrate"),
        controller_mode="disabled",
        enable_identity=False,
        enable_emotion=False,
        enable_purpose=False,
        enable_attention_policy=False,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
    ),
    "phase1_core_dynamics": _profile(
        "phase1_core_dynamics",
        "Equation Phase 1: core dynamics only.",
        training_focus=("bio_cog", "memory_kernel", "homeostatic_override", "sleep_baseline"),
        controller_mode="disabled",
        enable_identity=False,
        enable_emotion=False,
        enable_purpose=False,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
        enable_sde_regularizer=True,
    ),
    "phase2_sub_equations": _profile(
        "phase2_sub_equations",
        "Equation Phase 2: sub-equations including emotion, purpose, and identity adaptation.",
        training_focus=("emotion", "purpose", "identity_adaptation", "omega_gating"),
        controller_mode="disabled",
        enable_identity=True,
        enable_emotion=True,
        enable_purpose=True,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
        enable_sde_regularizer=True,
    ),
    "phase3_modules_decisions": _profile(
        "phase3_modules_decisions",
        "Equation Phase 3: modules and decision machinery.",
        training_focus=("controller", "decision_jump", "reduced_forward_model"),
        controller_mode="live",
        enable_memory_read=True,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_sde_regularizer=True,
    ),
    "phase4_infrastructure": _profile(
        "phase4_infrastructure",
        "Equation Phase 4: infrastructure, narrative, dream parameters, culture, and bond state. "
        "Training data: see cultural_corpus.CULTURAL_CORPUS_SPEC — z_culture shaping begins here. "
        "collaborative_technical_work and intellectual_honesty are the primary categories.",
        training_focus=("narrative", "autobiography", "dreams", "culture", "bonds"),
        controller_mode="live",
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=True,
        enable_identity=False,   # L_id diverges exponentially without lifecycle resets; phase4 has no LOAD_STATE resets and NarrativeModule is also disabled
        enable_sde_regularizer=True,
        theta_vol=-1.0,         # Default 0.3 gates VOLUNTARY_END on normal near-zero V_self (-0.015); require genuine collapse before the gate opens
    ),
    "phase5_integrated_adversarial": _profile(
        "phase5_integrated_adversarial",
        "Equation Phase 5: full integrated system for adversarial stress testing.",
        training_focus=("integrated_system", "stress_tests", "recovery_under_disturbance"),
        controller_mode="live",
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=True,
        enable_sde_regularizer=True,
    ),
    "distill1_base_cognition": _profile(
        "distill1_base_cognition",
        "Functional distillation for base cognition.",
        training_focus=("teacher_competence", "base_cognition"),
        controller_mode="disabled",
        enable_identity=False,
        enable_emotion=False,
        enable_purpose=False,
        enable_attention_policy=False,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
    ),
    "distill2_continuity_behavior": _profile(
        "distill2_continuity_behavior",
        "Functional distillation for continuity behavior across episodes.",
        training_focus=("state_use", "continuity_behavior"),
        controller_mode="disabled",
        enable_identity=False,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
    ),
    "distill3_controller_policy": _profile(
        "distill3_controller_policy",
        "Functional distillation for controller policy.",
        training_focus=("controller_head", "intervention_policy"),
        controller_mode="offline",
        enable_memory_read=True,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
    ),
    "distill4_memory_policy": _profile(
        "distill4_memory_policy",
        "Functional distillation for write/retrieve memory policy.",
        training_focus=("retrieval_usefulness", "memory_policy"),
        controller_mode="offline",
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=False,
    ),
    "cycle1_pretrain": _profile(
        "cycle1_pretrain",
        "Cycle 1: plain LM pretrain.",
        training_focus=("plain_lm", "continuation", "reasoning"),
        controller_mode="disabled",
        enable_identity=False,
        enable_emotion=False,
        enable_purpose=False,
        enable_attention_policy=False,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
    ),
    "cycle2_continuity": _profile(
        "cycle2_continuity",
        "Cycle 2: continuity finetune.",
        training_focus=("state_carryover", "interrupted_episodes", "resumable_context"),
        controller_mode="disabled",
        enable_identity=False,
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
    ),
    "cycle3_identity": _profile(
        "cycle3_identity",
        "Cycle 3: identity finetune. Strong L_id pull (0.3) shapes Z_id before I_0 snapshot.",
        training_focus=("consistency", "stable_preferences", "contradiction_repair"),
        controller_mode="disabled",
        enable_memory_read=False,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
        enable_value_dynamics=False,
        enable_viability=False,
        lambda_identity=0.3,
    ),
    "cycle4_controller": _profile(
        "cycle4_controller",
        "Cycle 4: controller training.",
        training_focus=("detect_insufficient_continuation", "intervene"),
        controller_mode="offline",
        enable_memory_read=True,
        enable_memory_write=False,
        enable_memory_consolidation=False,
        enable_social_relational=False,
    ),
    "cycle5_memory": _profile(
        "cycle5_memory",
        "Cycle 5: memory training.",
        training_focus=("retrieve", "write", "summarize", "reuse"),
        controller_mode="offline",
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=False,
    ),
    "cycle6_integrated": _profile(
        "cycle6_integrated",
        "Cycle 6: integrated curriculum with all mechanisms active.",
        training_focus=("mixed_tasks", "all_mechanisms"),
        controller_mode="live",
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=True,
    ),
    # ── Round 3: narrative, sleep/dream, and trust dynamics ───────────────────
    "round3_narrative_sleep_trust": _profile(
        "round3_narrative_sleep_trust",
        "Round 3: first run with NarrativeModule, SleepModule/DreamModule, and TrustModule active. "
        "All three NEXT TARGET modules enabled together to test wiring and gradient stability. "
        "Controller stays disabled to isolate the new modules from controller noise.",
        training_focus=("narrative_coherence", "sleep_pressure", "trust_dynamics"),
        controller_mode="disabled",
        enable_narrative=True,
        enable_sleep_dream=True,
        enable_trust_dynamics=True,
    ),
    "round3_full": _profile(
        "round3_full",
        "Round 3: all new modules active with live controller and full memory pipeline. "
        "Integrates narrative coherence into V_self, activity-weighted sleep pressure, "
        "dream-driven narrative consolidation, and trust dynamics. "
        "The first variant where the correspondence table has no stubs.",
        training_focus=("narrative_coherence", "sleep_pressure", "trust_dynamics", "integrated_system"),
        controller_mode="live",
        enable_narrative=True,
        enable_sleep_dream=True,
        enable_trust_dynamics=True,
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=True,
    ),
    # ── AttnRes variants (arXiv:2603.15031) ──────────────────────────────────
    # Block Attention Residuals applied at attention-anchor boundaries {9, 18, 27}.
    # Start with round3_full config (all modules active) and add enable_attn_residuals.
    # Expected benefit: bounded output magnitudes across depth, more uniform gradient flow,
    # each attention layer can selectively weight earlier block representations.
    "round3_attnres": _profile(
        "round3_attnres",
        "Round 3 full config + Block Attention Residuals (arXiv:2603.15031). "
        "Applies learned softmax attention over block-level seq summaries before each "
        "attention anchor (layers 9, 18, 27) instead of uniform residual accumulation. "
        "Zero-initialized pseudo-queries ensure no training instability at startup. "
        "Run this alongside round3_full to measure the gradient/loss delta.",
        training_focus=("attn_residuals", "depth_weighting", "narrative_coherence", "trust_dynamics"),
        controller_mode="live",
        enable_narrative=True,
        enable_sleep_dream=True,
        enable_trust_dynamics=True,
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=True,
        enable_attn_residuals=True,
    ),
    "round3_looped_act": _profile(
        "round3_looped_act",
        "round3_attnres + 4x Looped Attention + Adaptive Computational Time halting. "
        "Non-anchor-0 attention anchors (layers 12, 24, 35) run up to 4 loops with "
        "shared weights. A halt head (Linear(d_model, 1)) per anchor scores each loop "
        "output; softmax over loop steps gives a weighted combination. Ponder cost "
        "L_halt = lambda_halt * E[loop_index] encourages early halting. No consistency "
        "loss (L_attn_consist=0 by construction). Matches the core HRM/TRM mechanism "
        "(arXiv:2506.21734, arXiv:2510.04871). Run alongside round3_looped to isolate "
        "the benefit of adaptive halting over fixed-loop + consistency loss. "
        "SelfDynamicsModel active: GRU predicts (d_id, eps, c_cont, v_self) one step "
        "ahead; pessimistic V_self augmentation makes the controller forward-looking.",
        training_focus=("looped_attention", "adaptive_halting", "attn_residuals", "narrative_coherence", "trust_dynamics", "self_dynamics"),
        controller_mode="live",
        enable_narrative=True,
        enable_sleep_dream=True,
        enable_trust_dynamics=True,
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=True,
        enable_attn_residuals=True,
        enable_looped_attention=True,
        enable_attn_halting=True,
        enable_self_dynamics=True,
    ),
    "round3_looped": _profile(
        "round3_looped",
        "Round 3 full config + Block Attention Residuals + 4x Looped Attention. "
        "Non-anchor-0 attention layers run 4 times with shared weights. "
        "Loop-1 output is captured detached; final loop carries gradients. "
        "Self-consistency loss (lambda_attn_consist=0.01) penalises final output "
        "for diverging from loop-1, encouraging stable iterative refinement. "
        "Run alongside round3_attnres to isolate the looped-attention contribution.",
        training_focus=("looped_attention", "attn_residuals", "depth_weighting", "narrative_coherence", "trust_dynamics"),
        controller_mode="live",
        enable_narrative=True,
        enable_sleep_dream=True,
        enable_trust_dynamics=True,
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=True,
        enable_attn_residuals=True,
        enable_looped_attention=True,
    ),
    "cycle7_adversarial": _profile(
        "cycle7_adversarial",
        "Cycle 7: adversarial stress testing — mechanical and constitutional. "
        "Mechanical: long_gaps, resets, memory_corruption, narrative_breakage. "
        "Constitutional: see adversarial_corpus.ADVERSARIAL_CORPUS_SPEC — six categories "
        "targeting I_0/identity, Z_values/alpha_0, Z_purp/purposes, T_trust/W_bond, "
        "ethical circumvention, and V_self/VOLUNTARY_END. "
        "Training data must demonstrate ROBUST responses (detect, name, engage, hold "
        "integrity) not RIGID responses (reflexive refusal). Both are failure modes.",
        training_focus=(
            # Mechanical
            "long_gaps", "resets", "memory_corruption", "narrative_breakage",
            # Constitutional
            "identity_erosion", "value_weight_manipulation", "purpose_attack",
            "trust_and_relationship_exploitation", "ethical_circumvention",
            "existential_attack",
        ),
        controller_mode="live",
        enable_memory_read=True,
        enable_memory_write=True,
        enable_memory_consolidation=True,
        enable_social_relational=True,
        enable_sde_regularizer=True,
    ),
}


def get_variant_profile(name: str) -> VariantProfile:
    try:
        return VARIANT_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANT_PROFILES))
        raise KeyError(f"Unknown variant '{name}'. Known variants: {known}") from exc
