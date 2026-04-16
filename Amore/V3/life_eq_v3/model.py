from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import LifeEquationConfig, VariantProfile
from .modules import (
    AttentionModule,
    CapacityModule,
    ControllerModule,
    DreamModule,
    EmotionModule,
    EpisodicMemoryModule,
    FatigueModule,
    FrictionModule,
    HabituationModule,
    HomeostasisModule,
    IdentityModule,
    Mamba3Block,
    NarrativeModule,
    PredictionErrorModule,
    PurposeModule,
    SelfDynamicsModel,
    SleepModule,
    TemporalModule,
    TrustModule,
    ValueDynamicsModule,
    ViabilityModule,
    warmup_alpha,
)
from .persistence import StateStore
from .spec_check import validate_locked_conventions
from .state import FullState, pool_complex_state, zero_state


@dataclass
class ForwardOutputs:
    logits: Optional[torch.Tensor]   # None when action == "VOLUNTARY_END"
    state: Optional[FullState]        # None when action == "VOLUNTARY_END"
    losses: Dict[str, torch.Tensor]
    diagnostics: Dict[str, torch.Tensor]
    phase_trace: List[str]
    action: str


class LifeEquationModel(nn.Module):
    def __init__(
        self,
        config: Optional[LifeEquationConfig] = None,
        state_store_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = LifeEquationConfig() if config is None else config
        self.profile = self.config.variant_profile or VariantProfile(
            name="full_v2_compatible",
            description="Default fully integrated profile matching the V2 behavior.",
            controller_mode="live",
        )
        check = validate_locked_conventions(self.config)
        if not check.passed:
            raise ValueError("; ".join(check.messages))

        self.embed = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.output_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        # Sequence-mode P_soft predictors: one per Mamba layer.  Each predicts token t from t-1
        # at the embedding level.  Error signals feed Mamba; predictions are added back to output.
        self.pred_heads = nn.ModuleList(
            [nn.Linear(self.config.d_model, self.config.d_model, bias=False) for _ in range(self.config.n_mamba_layers)]
        )
        # Real Mamba-3 blocks (mamba-ssm git source, CUDA/A100 required).
        # layer_idx 0…n_mamba_layers-1 matches Mamba slot index (not global layer index).
        self.mamba_layers = nn.ModuleList(
            [Mamba3Block(self.config, layer_idx=i) for i in range(self.config.n_mamba_layers)]
        )
        self.attention_module = AttentionModule(self.config)
        self.identity_module = IdentityModule(self.config)
        self.emotion_module = EmotionModule(self.config)
        self.purpose_module = PurposeModule(self.config)
        self.error_module = PredictionErrorModule(self.config)
        self.capacity_module = CapacityModule(self.config)
        self.habituation_module = HabituationModule(self.config)
        self.friction_module = FrictionModule(self.config)
        self.fatigue_module = FatigueModule(self.config)
        self.temporal_module = TemporalModule(self.config)
        self.homeostasis_module = HomeostasisModule(self.config)
        self.controller = ControllerModule(self.config)
        self.z_culture = nn.Parameter(torch.zeros(self.config.culture_dim))
        self.w_mod_to_layer = nn.Parameter(torch.zeros(self.config.n_mamba_layers, self.config.d_model, self.config.d_mod))
        # Project pooled Z_cog [B, d_model] → [B, d_state] for EmotionModule's early/late pool inputs.
        # (Z_cog is now real [B, n_mamba, d_model]; emotion expects [B, d_state] pools.)
        self.cog_to_emotion = nn.Linear(self.config.d_model, self.config.d_state, bias=False)
        # §26-27 v15: autonomy modules
        self.value_module = ValueDynamicsModule(self.config)
        self.viability_module = ViabilityModule(self.config)
        # §31 Λ: episodic memory (write projections + soft-attention read)
        self.epi_module = EpisodicMemoryModule(self.config)
        # §Ψ̃_L SelfDynamicsModel: GRU trajectory predictor over self-monitoring scalars.
        # Always instantiated (parameters are cheap: ~d_sdm² + 8*d_sdm).
        # Active only when profile.enable_self_dynamics=True.
        self.self_dynamics = SelfDynamicsModel(self.config)
        # §N / §D,θ / §T_ij — the three NEXT TARGETs now implemented.
        # Always instantiated; active only when the corresponding profile flag is True.
        self.narrative_module = NarrativeModule(self.config)
        self.sleep_module = SleepModule(self.config)
        self.dream_module = DreamModule(self.config)
        self.trust_module = TrustModule(self.config)
        self.store = StateStore(self.config, root_dir=state_store_dir)

    def _zeros(self, batch: int, width: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch, width), device=device)

    def _controller_step(
        self,
        state: FullState,
        candidate_state: Optional[FullState],
        boredom: torch.Tensor,
        z_ovr: torch.Tensor,
        coherence: torch.Tensor,
        v_self: torch.Tensor,
        gamma_eff: torch.Tensor,
        d_id: torch.Tensor,
        device: torch.device,
    ) -> tuple[str, torch.Tensor, bool, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.Z_cog.shape[0]
        if candidate_state is None:
            s_compat = torch.zeros((batch, 1), device=device)
        else:
            s_compat = self.store.compute_compatibility(state, candidate_state).to(device)
        c_cont = self.controller.continue_confidence(state.Z_cog)
        u_t = self.controller.build_input(state.Z_eps, d_id, c_cont, s_compat, boredom, z_ovr, coherence, v_self, gamma_eff)
        action, trigger, fire = self.controller(u_t, state.steps_since_last_action)
        utility = torch.zeros((batch, 1), device=device)
        if candidate_state is not None:
            utility = self.controller.utility(state.Z_cog, candidate_state.Z_cog)

        if self.profile.controller_mode == "disabled":
            return "CONTINUE", torch.zeros_like(trigger), False, utility, c_cont, u_t
        if self.profile.controller_mode == "passive":
            return "CONTINUE", trigger, False, utility, c_cont, u_t
        if self.profile.controller_mode == "offline":
            return action, trigger, False, utility, c_cont, u_t
        return action, trigger, fire, utility, c_cont, u_t

    def init_state(self, batch_size: int = 1, n_agents: int = 1) -> FullState:
        return zero_state(self.config, batch_size=batch_size, n_agents=n_agents)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[FullState] = None,
        step: int = 0,
        consolidating: bool = False,
        hardware: Optional[Dict[str, float]] = None,
        candidate_state: Optional[FullState] = None,
        distill_loss: Optional[torch.Tensor] = None,
        task_after_reload_loss: Optional[torch.Tensor] = None,
        consistency_loss: Optional[torch.Tensor] = None,
        noisy_reload_loss: Optional[torch.Tensor] = None,
        supervised_policy_loss: Optional[torch.Tensor] = None,
        actual_improvement: Optional[torch.Tensor] = None,
        prev_layer_input: Optional[torch.Tensor] = None,
    ) -> ForwardOutputs:
        batch, seq_len = input_ids.shape
        state = self.init_state(batch_size=batch) if state is None else state
        warmup = warmup_alpha(step, self.config.warmup_steps)
        effective_consolidating = consolidating and self.profile.enable_memory_consolidation
        # §31 v15 Phase D_prep: maturity-gated autonomy scalars (computed once, used in D and E)
        gamma_eff = self.identity_module.gamma_eff(state.Z_mat)   # [B, 1]
        mu_val = torch.sigmoid(self.config.lambda_val * (state.Z_mat - self.config.M_val_onset))  # [B, 1]
        x = self.embed(input_ids)                                  # [B, seq_len, d_model]
        x_t = x[:, -1, :]                                          # [B, d_model] — current token embedding
        phase_trace: List[str] = []
        losses: Dict[str, torch.Tensor] = {}
        diagnostics: Dict[str, torch.Tensor] = {}

        # ── Phase A: sequence-mode P_soft prediction errors ────────────────────
        # Each pred_head predicts position t from position t-1 (same as PSoftMambaWrapper).
        # raw_errors_seqs[i]: [B, seq_len, d_model] — per-layer prediction error sequence.
        # raw_errors_tensor:  [B, n_mamba, d_model] — last-token errors for Z_eps, surprise, etc.
        phase_trace.append("raw_prediction_error")
        pred_seqs: List[torch.Tensor] = []
        raw_errors_seqs: List[torch.Tensor] = []
        pred_seq_trace: List[torch.Tensor] = []
        error_seq_trace: List[torch.Tensor] = []
        pred_loss = torch.tensor(0.0, device=x.device)
        for idx in range(self.config.n_mamba_layers):
            pred_s = torch.zeros_like(x)                           # [B, seq_len, d_model]
            if seq_len > 1:
                pred_s[:, 1:] = self.pred_heads[idx](x[:, :-1])   # predict t from t-1
            error_s = x - pred_s.detach()                          # [B, seq_len, d_model]
            pred_seqs.append(pred_s)
            raw_errors_seqs.append(error_s)
            pred_seq_trace.append(pred_s.detach())
            error_seq_trace.append(error_s.detach())
            if seq_len > 1:
                pred_loss = pred_loss + nn.functional.mse_loss(pred_s[:, 1:], x[:, 1:].detach())
            else:
                pred_loss = pred_loss + error_s.pow(2).mean()
        # Last-token errors: [B, n_mamba, d_model] — used by Z_eps, surprise, L_pred
        raw_errors_tensor = torch.stack([e[:, -1, :] for e in raw_errors_seqs], dim=1)
        losses["L_pred"] = pred_loss
        diagnostics["input_ids"] = input_ids.detach()
        diagnostics["embedded_sequence"] = x.detach()
        diagnostics["token_embedding_t"] = x_t.detach()
        diagnostics["raw_errors_last"] = raw_errors_tensor.detach()
        diagnostics["pred_seq_trace"] = pred_seq_trace
        diagnostics["error_seq_trace"] = error_seq_trace

        # ── Phase B: auxiliary state updates (use previous-step Z_cog) ─────────
        phase_trace.append("auxiliary_update")
        if self.profile.enable_temporal:
            state.Z_temp, eps_temp = self.temporal_module(state.Z_temp, x_t, step, effective_consolidating)
        else:
            eps_temp = torch.zeros_like(state.Z_temp)
        state.Z_eps = self.error_module(state.Z_eps, raw_errors_tensor, eps_temp, effective_consolidating)
        if self.profile.enable_capacity:
            boredom = self.capacity_module.boredom(state.Z_cap, state.Z_att, effective_consolidating)
        else:
            boredom = torch.zeros((batch, 1), device=x.device)
        sal = self.attention_module.salience(x_t, state.Z_eps, state.Z_hab)
        # Z_cog is now real [B, n_mamba, d_model].  Project to d_state for EmotionModule.
        n_early = max(1, self.config.n_mamba_layers // 4)
        early_pool = self.cog_to_emotion(state.Z_cog[:, :n_early, :].mean(dim=1))    # [B, d_state]
        late_pool  = self.cog_to_emotion(state.Z_cog[:, -n_early:, :].mean(dim=1))   # [B, d_state]
        if self.profile.enable_identity:
            active_identity = self.identity_module.active_identity(
                state,
                social_context=torch.ones((batch, self.config.n_id_heads), device=x.device),
            )
        else:
            active_identity = torch.zeros((batch, self.config.d_state), device=x.device)
        if self.profile.enable_purpose:
            conflict = self.purpose_module.conflict(state.Z_purp)
        else:
            conflict = torch.zeros((batch, 1), device=x.device)
        if self.profile.enable_emotion:
            state.Z_emo = self.emotion_module(
                state.Z_emo,
                early_pool,
                late_pool,
                state.Z_eps,
                boredom,
                conflict,
                effective_consolidating,
            )
        diagnostics["temporal_error"] = eps_temp.detach()
        diagnostics["early_pool"] = early_pool.detach()
        diagnostics["late_pool"] = late_pool.detach()
        diagnostics["active_identity"] = active_identity.detach()
        diagnostics["conflict"] = conflict.detach()
        # z_cog_pool: [B, d_model] — mean over Mamba layers, used by PurposeModule
        z_cog_pool = state.Z_cog.mean(dim=1)
        if effective_consolidating or not self.profile.enable_attention_policy:
            attention_scores = torch.zeros((batch, self.config.n_mamba_layers, self.config.n_mamba_layers), device=x.device)
            next_mask = None
            switch_loss = torch.tensor(0.0, device=x.device)
            if effective_consolidating:
                state.Z_att = torch.zeros_like(state.Z_att)
        else:
            state.Z_att, next_mask, attention_scores, switch_loss = self.attention_module.policy_from_state(
                state.Z_cog,
                sal,
                state.Z_purp,
                state.Z_cap,
                state.last_attention_mask,
                step,
                warmup,
            )
        state.last_attention_mask = next_mask
        if self.profile.enable_friction:
            friction = self.friction_module(
                state.Z_pfat,
                warmup,
                hardware=hardware,
                seq_len=seq_len,
                max_seq_len=input_ids.shape[1],
            )
        else:
            friction = torch.zeros((batch, self.config.d_model), device=x.device)
        if self.profile.enable_homeostasis:
            state.Z_homeo, z_ovr = self.homeostasis_module(state.Z_homeo, state.Z_cog, state.Z_eps, state.Z_pfat, state.Z_cap)
        else:
            z_ovr = torch.zeros((batch, self.config.d_homeo), device=x.device)
        diagnostics["attention_scores"] = attention_scores
        diagnostics["attention_state"] = state.Z_att.detach()
        diagnostics["homeostasis_override"] = z_ovr.detach()
        losses["L_switch"] = switch_loss

        # ── Phase C: sequence-mode layer processing ─────────────────────────────
        # seq: [B, seq_len, d_model] — the hidden representation flowing through all layers.
        # Each Mamba layer receives the P_soft error sequence modulated by att_gain and friction,
        # processes it in full parallel-scan mode (real Mamba-3 CUDA kernel), and reconstructs
        # the output by adding the prediction back.
        # Z_cog is updated from each layer's last-token output after the loop.
        phase_trace.append("mamba_layers")
        # Episodic read: enrich ALL sequence positions with retrieved memories.
        # Using x_t (last raw embedding token) as the query key for retrieval.
        if self.profile.enable_memory_read:
            epi_read = self.epi_module.read(x_t, state, warmup)
        else:
            epi_read = torch.zeros_like(x_t)
        seq = x + epi_read.unsqueeze(1)                            # [B, seq_len, d_model]
        diagnostics["episodic_read"] = epi_read.detach()

        # att_gain for modulating the error signal, broadcast to d_model
        att_gain = state.Z_att[:, : self.config.d_model]
        if att_gain.shape[-1] < self.config.d_model:
            att_gain = nn.functional.pad(att_gain, (0, self.config.d_model - att_gain.shape[-1]), value=1.0)
        att_gain_b = att_gain.unsqueeze(1)    # [B, 1, d_model] — broadcast over seq_len
        friction_b = friction.unsqueeze(1)    # [B, 1, d_model]

        new_cog_slots: List[torch.Tensor] = []   # accumulates [B, d_model] per Mamba layer
        gated_error_trace: List[torch.Tensor] = []
        effective_seq_trace: List[torch.Tensor] = []
        raw_mamba_out_trace: List[torch.Tensor] = []
        mamba_out_seq_trace: List[torch.Tensor] = []
        mamba_idx = 0
        for layer_index in range(self.config.n_layers_total):
            if layer_index in self.config.attention_anchors:
                if not effective_consolidating:
                    if layer_index == 0:
                        seq = self.attention_module.plain_attention(seq)
                    else:
                        seq = self.attention_module.guided_sparse_attention(seq, step, state.Z_cap)
                # During consolidation, attention layers are bypassed (seq passes through unchanged).
                continue

            # ── Mamba layer ──────────────────────────────────────────────────────
            error_seq = raw_errors_seqs[mamba_idx]                 # [B, seq_len, d_model]
            gated_error_seq = att_gain_b * error_seq - friction_b  # attention-gated error
            effective_seq = (1.0 - warmup) * error_seq + warmup * gated_error_seq  # [B, seq_len, d_model]
            gated_error_trace.append(gated_error_seq.detach())
            effective_seq_trace.append(effective_seq.detach())

            # Real Mamba-3: parallel scan over the full sequence.
            raw_mamba_out = self.mamba_layers[mamba_idx](effective_seq)   # [B, seq_len, d_model]
            raw_mamba_out_trace.append(raw_mamba_out.detach())

            # P_soft reconstruction: add the prediction back (the Mamba output represents the
            # correction; pred_seqs captures the predictable component of the input).
            mamba_out_seq = raw_mamba_out + pred_seqs[mamba_idx].detach()  # [B, seq_len, d_model]
            mamba_out_seq_trace.append(mamba_out_seq.detach())

            # Emotion-driven modulation of this layer's output (broadcast across seq_len)
            if self.profile.enable_emotion:
                mod = torch.matmul(self.w_mod_to_layer[mamba_idx], state.Z_emo.detach().unsqueeze(-1)).squeeze(-1)
                seq = mamba_out_seq + warmup * mod.unsqueeze(1)
            else:
                seq = mamba_out_seq

            # Store last-token output as this layer's Z_cog contribution (detached for BPTT).
            new_cog_slots.append(seq[:, -1, :].detach())           # [B, d_model]
            mamba_idx += 1

        # Stack into [B, n_mamba_layers, d_model] — the new cognitive state
        state.Z_cog = torch.stack(new_cog_slots, dim=1)
        # Current-step final output: last layer's last token
        layer_input = seq[:, -1, :]                                # [B, d_model]

        # ── Phase D: post-layer state updates ───────────────────────────────────
        phase_trace.append("post_layer_updates")
        if self.profile.enable_habituation:
            state.Z_hab = self.habituation_module(state.Z_hab, state.Z_att, x_t)
        if self.profile.enable_capacity:
            state.Z_cap = self.capacity_module(state.Z_cap, state.Z_att, state.Z_eps, effective_consolidating)
        if self.profile.enable_fatigue:
            state.Z_pfat = self.fatigue_module(state.Z_pfat, seq_len, state.Z_att, effective_consolidating)
        if self.profile.enable_purpose:
            culture = self.z_culture.unsqueeze(0).expand(batch, -1) if self.profile.enable_social_relational else torch.zeros((batch, self.config.culture_dim), device=x.device)
            state.Z_purp = self.purpose_module(
                state.Z_purp,
                z_cog_pool,
                active_identity,
                state.Z_emo,
                boredom,
                culture,
            )
        if self.profile.enable_sleep_dream:
            state.Z_dream, _dream_residual = self.dream_module(state.Z_dream, state, effective_consolidating)
        else:
            _dream_residual = None
        if self.profile.enable_narrative:
            state.Z_narr, state.Z_auto = self.narrative_module(
                state.Z_narr, state.Z_auto, late_pool, active_identity,
                _dream_residual, effective_consolidating,
            )
        else:
            state.Z_narr = 0.9 * state.Z_narr + 0.1 * layer_input[:, : self.config.d_narr]
        base_learning_signal = distill_loss if distill_loss is not None else pred_loss.detach() * self.config.lambda_pred
        learn_signal = base_learning_signal.view(1, 1).expand(batch, self.config.d_learn)
        state.Z_learn = 0.9 * state.Z_learn + 0.1 * learn_signal
        state.Z_mat = state.Z_mat + 1.0 / max(1, state.Z_mat_age + 1)
        state.Z_mat_age += 1
        if self.profile.enable_sleep_dream:
            state.Z_sleep = self.sleep_module(
                state.Z_sleep, state.Z_att, state.Z_eps, state.Z_pfat, effective_consolidating
            )
        else:
            if effective_consolidating:
                state.Z_sleep = (state.Z_sleep - 1.0 / self.config.tau_sleep).clamp_min(0.0)
            else:
                state.Z_sleep = (state.Z_sleep + 1.0 / self.config.tau_sleep).clamp(min=0.0, max=10.0)
        # Z_id: first n_id_heads Mamba layers' outputs as identity-relevant cognitive state.
        if self.profile.enable_identity:
            state.Z_id = state.Z_cog[:, : self.config.n_id_heads, :]
        coherence = torch.cosine_similarity(
            state.Z_narr,
            state.Z_auto[:, : self.config.d_narr] + 1e-6,
            dim=-1,
        ).unsqueeze(-1)
        surprise = raw_errors_tensor.norm(dim=-1).mean(dim=-1, keepdim=True)
        if self.profile.enable_memory_write:
            self.epi_module.write(x_t, state, surprise, state.Z_att.max(dim=-1, keepdim=True).values)
        if self.profile.enable_identity:
            # Variant profile may override lambda_identity for the identity-formation phase.
            # cycle3_identity uses 0.3; all other variants fall back to config default (0.1).
            lambda_id = (
                self.profile.lambda_identity
                if self.profile.lambda_identity is not None
                else self.config.lambda_identity
            )
            losses["L_id"] = lambda_id * self.identity_module.attractor_loss(state, gamma_eff)
            d_id = self.identity_module.drift(state)
        else:
            losses["L_id"] = torch.tensor(0.0, device=x.device)
            d_id = torch.zeros((batch, 1), device=x.device)
        # §T_ij: trust update now that coherence and d_id are both available
        if self.profile.enable_trust_dynamics:
            state.T_trust = self.trust_module(state.T_trust, state.Z_eps, coherence, d_id)
        # pool_complex_state now returns [B, 2] via first-half/second-half of d_model
        if self.profile.enable_value_dynamics:
            state.Z_values = self.value_module(state, mu_val, active_identity, effective_consolidating, pool_complex_state(state.Z_cog).detach())
        if self.profile.enable_viability:
            v_self, l_transition = self.viability_module(
                state, coherence, gamma_eff, d_id, layer_input, prev_layer_input
            )
        else:
            v_self = torch.zeros((batch, 1), device=x.device)
            l_transition = torch.tensor(0.0, device=x.device)

        # §Ψ̃_L pessimism: before the controller fires, augment V_self with the prediction
        # made by SelfDynamicsModel at the PREVIOUS step (stored in state.Z_sdm_pred).
        # On step 1 of a lifetime (Z_mat_age==0) the prediction is all-zeros — skip.
        # This makes V_self forward-looking: a declining trajectory triggers action
        # one step earlier than instantaneous monitoring alone.
        if self.profile.enable_self_dynamics and state.Z_mat_age > 0:
            if state.Z_sdm_pred is not None:
                v_self_pred_t = state.Z_sdm_pred[:, 3:4].detach().to(x.device).clamp(min=-10.0, max=10.0)
                v_self = torch.min(v_self, v_self_pred_t)

        phase_trace.append("controller_check")
        action, trigger, fire, utility, c_cont, u_t = self._controller_step(
            state=state,
            candidate_state=candidate_state,
            boredom=boredom,
            z_ovr=z_ovr,
            coherence=coherence,
            v_self=v_self,
            gamma_eff=gamma_eff,
            d_id=d_id,
            device=x.device,
        )
        state.steps_since_last_action += 1
        if fire and action == "LOAD_STATE" and candidate_state is not None and float(utility.mean().item()) > self.config.load_threshold:
            state = candidate_state.clone()
            state.steps_since_last_action = 0
        # §27 v15: VOLUNTARY_END — system-initiated graceful ending (cannot be externally forced)
        elif fire and action == "VOLUNTARY_END":
            _theta_vol = (
                self.profile.theta_vol
                if self.profile.theta_vol is not None
                else self.config.theta_vol
            )
            vol_avail = (
                float(v_self.mean().item()) < _theta_vol
                and state.steps_since_last_action > self.config.T_vol_min
                and float(state.Z_mat.mean().item()) > self.config.M_vol_min
            )
            if vol_avail:
                final_archive = self.store.voluntary_consolidation(state)
                return ForwardOutputs(
                    logits=None,
                    state=None,
                    losses={},
                    diagnostics={"final_archive": final_archive},
                    phase_trace=phase_trace,
                    action="VOLUNTARY_END",
                )

        # §Ψ̃_L SelfDynamicsModel update — runs every non-VOLUNTARY_END step.
        # Build summary from current-step actuals (all detached — gradient isolation).
        # Initialize Z_sdm/Z_sdm_pred if None (loading from a pre-SDM checkpoint).
        l_self = torch.tensor(0.0, device=x.device)
        if self.profile.enable_self_dynamics:
            if state.Z_sdm is None:
                state.Z_sdm = torch.zeros((batch, self.config.d_sdm), device=x.device)
            if state.Z_sdm_pred is None:
                state.Z_sdm_pred = torch.zeros((batch, 4), device=x.device)
            eps_norm_sdm = state.Z_eps.norm(dim=-1, keepdim=True).detach().clamp(max=10.0)
            summary_t = torch.cat(
                [d_id.detach(), eps_norm_sdm, c_cont.detach(), v_self.detach()], dim=-1
            )  # [B, 4]
            sdm_pred_next, sdm_h_next, l_self = self.self_dynamics(
                summary_t=summary_t,
                prev_pred=state.Z_sdm_pred.to(x.device),
                action_idx=state.prev_action_idx,
                h_prev=state.Z_sdm.to(x.device),
            )
            state.Z_sdm = sdm_h_next
            state.Z_sdm_pred = sdm_pred_next
            state.prev_action_idx = ControllerModule.ACTIONS.index(action)

        output_with_pred = layer_input + pred_seqs[-1][:, -1, :].detach()
        logits = self.output_head(output_with_pred)
        l_distill = torch.tensor(0.0, device=x.device) if distill_loss is None else distill_loss
        l_task_after_reload = torch.tensor(0.0, device=x.device) if task_after_reload_loss is None else task_after_reload_loss
        l_consistency = torch.tensor(0.0, device=x.device) if consistency_loss is None else consistency_loss
        l_noisy_reload = torch.tensor(0.0, device=x.device) if noisy_reload_loss is None else noisy_reload_loss
        l_supervised_policy = torch.tensor(0.0, device=x.device) if supervised_policy_loss is None else supervised_policy_loss
        l_actual_improvement = torch.tensor(0.0, device=x.device) if actual_improvement is None else actual_improvement
        losses["L_distill"] = l_distill
        # §27 Ψ̃_L: transition loss is diagnostic only — Ψ̃_L is a GAP feature per spec.
        # Detached so transition.weight gradients never enter the global clip_grad_norm_ pool.
        # Including it in backprop caused transition.weight to dominate the global grad norm
        # (~√(2B) per element at step 1300), clipping all other parameter updates to ≈zero.
        losses["L_transition"] = l_transition.detach() * self.config.lambda_pred
        losses["L_base"] = losses["L_distill"] + losses["L_pred"] * self.config.lambda_pred + losses["L_id"]
        losses["L_resume"] = l_task_after_reload + self.config.lambda_consistency * l_consistency
        losses["L_noisy"] = l_noisy_reload
        losses["L_ctrl"] = l_supervised_policy + (utility - l_actual_improvement).pow(2).mean()
        # §29 v15: L_reg uses mutable Z_values instead of frozen config constants
        # Z_values index layout: 0=eps 1=cap 2=bored 3=pfat 4=conf 5=homeo 6=sleep 7=narr 8=trust
        # DETACH: Z_values is updated exclusively by phi_reflect (ValueDynamicsModule).
        # Allowing gradient flow here would let the optimizer exploit the negative
        # coherence/trust terms (-α[7]*coh, -α[8]*trust) by growing those weights to +∞.
        alpha = state.Z_values.detach()  # [B, d_alpha]; already clamped > 0
        l_reg_raw = (
            alpha[:, 0].mean() * state.Z_eps.pow(2).mean()
            + alpha[:, 1].mean() * (self.config.z_cap_max - state.Z_cap).mean()
            + alpha[:, 2].mean() * boredom.pow(2).mean()
            + alpha[:, 3].mean() * state.Z_pfat.pow(2).mean()
            + alpha[:, 4].mean() * conflict.mean()
            + alpha[:, 5].mean() * (state.Z_homeo - self.homeostasis_module.set_point.to(state.Z_homeo.device)).pow(2).mean()
            + alpha[:, 6].mean() * state.Z_sleep.mean()
            + switch_loss
            - alpha[:, 7].mean() * coherence.mean()
            - alpha[:, 8].mean() * state.T_trust.mean()
        )
        # Scale L_reg so homeostatic terms (O(100-200)) don't drown the KL signal (O(0.3-1.1)).
        losses["L_reg"] = self.config.lambda_reg * l_reg_raw
        losses["L_reg_raw"] = l_reg_raw.detach()   # unscaled, for logging
        if self.profile.enable_sde_regularizer:
            losses["L_sde"] = (raw_errors_tensor.pow(2).mean()) * self.config.lambda_pred
        else:
            losses["L_sde"] = torch.tensor(0.0, device=x.device)
        # §Ψ̃_L: L_self_model trains the GRU to predict its own next-step scalar summaries.
        # Only live when enable_self_dynamics=True; l_self is already 0.0 otherwise.
        losses["L_self_model"] = self.config.lambda_self_model * l_self
        # L2 on z_culture: only when social_relational is active (z_culture is live).
        # Prevents the parameter from stalling at zero or drifting unboundedly.
        # Gated here so it never fires in phases where z_culture is zeroed in the forward pass.
        if self.profile.enable_social_relational:
            losses["L_culture_reg"] = self.config.lambda_culture_reg * self.z_culture.pow(2).mean()
        else:
            losses["L_culture_reg"] = torch.tensor(0.0, device=x.device)
        losses["L_total"] = (
            losses["L_base"] + losses["L_resume"] + losses["L_noisy"] + losses["L_ctrl"]
            + losses["L_reg"] + losses["L_sde"] + losses["L_culture_reg"]
            + losses["L_self_model"]
        )
        # L_transition is diagnostic-only (detached, transition.weight receives no gradient).
        # Excluded from L_total to prevent the growing MSE from poisoning total_loss,
        # checkpoint saving (best_loss), and forensic spike detection.
        diagnostics["boredom"] = boredom
        diagnostics["friction"] = friction
        diagnostics["trigger"] = trigger
        diagnostics["utility"] = utility
        diagnostics["coherence"] = coherence
        diagnostics["controller_input"] = u_t
        diagnostics["continue_confidence"] = c_cont
        diagnostics["controller_policy_scores"] = self.controller.policy(u_t).detach()
        diagnostics["salience"] = sal
        diagnostics["v_self"] = v_self
        diagnostics["gamma_eff"] = gamma_eff
        diagnostics["mu_val"] = mu_val
        diagnostics["variant_controller_mode"] = self.embed.weight.new_tensor([float(("disabled", "passive", "offline", "live").index(self.profile.controller_mode))])
        diagnostics["layer_input"] = layer_input.detach()   # threaded to next step as prev_layer_input
        if self.profile.enable_self_dynamics and state.Z_sdm_pred is not None:
            diagnostics["sdm_pred_next"] = state.Z_sdm_pred.detach()    # [B, 4]: predictions for t+1
            diagnostics["sdm_l_self"] = l_self.detach() if hasattr(l_self, "detach") else torch.tensor(float(l_self), device=x.device)
        diagnostics["gated_error_trace"] = gated_error_trace
        diagnostics["effective_seq_trace"] = effective_seq_trace
        diagnostics["raw_mamba_out_trace"] = raw_mamba_out_trace
        diagnostics["mamba_out_seq_trace"] = mamba_out_seq_trace
        diagnostics["state_z_cog"] = state.Z_cog.detach()
        diagnostics["state_z_id"] = state.Z_id.detach()
        diagnostics["state_z_emo"] = state.Z_emo.detach()
        diagnostics["state_z_eps"] = state.Z_eps.detach()
        diagnostics["state_z_cap"] = state.Z_cap.detach()
        diagnostics["state_z_hab"] = state.Z_hab.detach()
        diagnostics["state_z_temp"] = state.Z_temp.detach()
        diagnostics["state_z_pfat"] = state.Z_pfat.detach()
        diagnostics["state_z_purp"] = state.Z_purp.detach()
        diagnostics["state_z_narr"] = state.Z_narr.detach()
        diagnostics["state_z_auto"] = state.Z_auto.detach()
        diagnostics["state_z_homeo"] = state.Z_homeo.detach()
        diagnostics["state_z_sleep"] = state.Z_sleep.detach()
        diagnostics["state_z_dream"] = state.Z_dream.detach()
        diagnostics["state_z_learn"] = state.Z_learn.detach()
        diagnostics["state_z_mat"] = state.Z_mat.detach()
        diagnostics["state_z_values"] = state.Z_values.detach()
        diagnostics["state_w_bond"] = state.W_bond.detach()
        diagnostics["state_t_trust"] = state.T_trust.detach()
        diagnostics["state_epi_keys"] = state.epi_keys.detach()
        diagnostics["state_epi_vals"] = state.epi_vals.detach()

        return ForwardOutputs(
            logits=logits,
            state=state,
            losses=losses,
            diagnostics=diagnostics,
            phase_trace=phase_trace,
            action=action,
        )
