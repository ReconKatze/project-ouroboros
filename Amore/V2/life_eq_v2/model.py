from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import LifeEquationConfig
from .modules import (
    AttentionModule,
    CapacityModule,
    ControllerModule,
    EmotionModule,
    FatigueModule,
    FrictionModule,
    HabituationModule,
    HomeostasisModule,
    IdentityModule,
    MambaStep,
    PredictionErrorModule,
    PurposeModule,
    TemporalModule,
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
    def __init__(self, config: Optional[LifeEquationConfig] = None):
        super().__init__()
        self.config = LifeEquationConfig() if config is None else config
        check = validate_locked_conventions(self.config)
        if not check.passed:
            raise ValueError("; ".join(check.messages))

        self.embed = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.output_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.pred_heads = nn.ModuleList(
            [nn.Linear(self.config.d_model, self.config.d_model, bias=False) for _ in range(self.config.n_mamba_layers)]
        )
        self.mamba_layers = nn.ModuleList([MambaStep(self.config) for _ in range(self.config.n_mamba_layers)])
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
        self.z_cog_pool_proj = nn.Linear(self.config.n_heads * self.config.d_state, self.config.d_model)
        # §26-27 v15: autonomy modules
        self.value_module = ValueDynamicsModule(self.config)
        self.viability_module = ViabilityModule(self.config)
        self.store = StateStore(self.config)

    def init_state(self, batch_size: int = 1, n_agents: int = 1) -> FullState:
        return zero_state(self.config, batch_size=batch_size, n_agents=n_agents)

    def _episodic_write(self, state: FullState, x_t: torch.Tensor, surprise: torch.Tensor, write_strength: torch.Tensor) -> None:
        if float(surprise.mean().item()) <= self.config.episodic_surprise_threshold:
            return
        slot = state.epi_index % self.config.n_epi_slots
        key = x_t[:, : self.config.d_key].mean(dim=0)
        val = x_t[:, : self.config.d_val].mean(dim=0) * write_strength.mean()
        state.epi_keys[slot] = key
        state.epi_vals[slot] = val
        state.epi_index += 1

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
    ) -> ForwardOutputs:
        batch, seq_len = input_ids.shape
        state = self.init_state(batch_size=batch) if state is None else state
        warmup = warmup_alpha(step, self.config.warmup_steps)
        # §31 v15 Phase D_prep: maturity-gated autonomy scalars (computed once, used in D and E)
        gamma_eff = self.identity_module.gamma_eff(state.Z_mat)   # [B, 1]
        mu_val = torch.sigmoid(self.config.lambda_val * (state.Z_mat - self.config.M_val_onset))  # [B, 1]
        x = self.embed(input_ids)
        x_t = x[:, -1, :]
        phase_trace: List[str] = []
        losses: Dict[str, torch.Tensor] = {}
        diagnostics: Dict[str, torch.Tensor] = {}

        phase_trace.append("raw_prediction_error")
        raw_errors = []
        pred_terms = []
        pred_loss = torch.tensor(0.0, device=x.device)
        prev_x = x[:, -2, :] if seq_len > 1 else torch.zeros_like(x_t)
        for idx in range(self.config.n_mamba_layers):
            pred_t = self.pred_heads[idx](prev_x)
            error_t = x_t - pred_t
            raw_errors.append(error_t)
            pred_terms.append(pred_t)
            pred_loss = pred_loss + error_t.pow(2).mean()
        raw_errors_tensor = torch.stack(raw_errors, dim=1)
        losses["L_pred"] = pred_loss

        phase_trace.append("auxiliary_update")
        state.Z_temp, eps_temp = self.temporal_module(state.Z_temp, x_t, step, consolidating)
        state.Z_eps = self.error_module(state.Z_eps, raw_errors_tensor, eps_temp, consolidating)
        boredom = self.capacity_module.boredom(state.Z_cap, state.Z_att, consolidating)
        sal = self.attention_module.salience(x_t, state.Z_eps, state.Z_hab)
        early_pool = state.Z_cog[:, : max(1, self.config.n_mamba_layers // 4)].real.mean(dim=(1, 2))
        late_pool = state.Z_cog[:, -max(1, self.config.n_mamba_layers // 4) :].real.mean(dim=(1, 2))
        active_identity = self.identity_module.active_identity(
            state,
            social_context=torch.ones((batch, self.config.n_id_heads), device=x.device),
        )
        conflict = self.purpose_module.conflict(state.Z_purp)
        state.Z_emo = self.emotion_module(state.Z_emo, early_pool, late_pool, state.Z_eps, boredom, conflict, consolidating)
        z_cog_pool = self.z_cog_pool_proj(state.Z_cog.real.mean(dim=1).reshape(batch, -1))
        if consolidating:
            attention_scores = torch.zeros((batch, self.config.n_mamba_layers, self.config.n_mamba_layers), device=x.device)
            next_mask = None
            switch_loss = torch.tensor(0.0, device=x.device)
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
        friction = self.friction_module(
            state.Z_pfat,
            warmup,
            hardware=hardware,
            seq_len=seq_len,
            max_seq_len=input_ids.shape[1],
        )
        state.Z_homeo, z_ovr = self.homeostasis_module(state.Z_homeo, state.Z_cog, state.Z_eps, state.Z_pfat, state.Z_cap)
        diagnostics["attention_scores"] = attention_scores
        losses["L_switch"] = switch_loss

        phase_trace.append("mamba_layers")
        sequence_hidden = x.clone()
        layer_input = x_t
        mamba_idx = 0
        # Accumulate new Z_cog slots in a list to avoid inplace modification of a
        # tensor already in the autograd graph (would corrupt version counters).
        new_cog_slots = list(state.Z_cog.unbind(dim=1))  # list of [B, n_heads, d_state] cfloat
        for layer_index in range(self.config.n_layers_total):
            if layer_index in self.config.attention_anchors:
                sequence_hidden[:, -1, :] = layer_input
                if not consolidating:
                    if layer_index == 0:
                        layer_input = self.attention_module.plain_attention(sequence_hidden)[:, -1, :]
                    else:
                        layer_input = self.attention_module.guided_sparse_attention(
                            sequence_hidden,
                            state.last_attention_mask,
                        )[:, -1, :]
                else:
                    layer_input = sequence_hidden[:, -1, :]
                sequence_hidden[:, -1, :] = layer_input
                continue
            error_t = raw_errors_tensor[:, mamba_idx]
            att_gain = state.Z_att[:, : self.config.d_model]
            if att_gain.shape[-1] < self.config.d_model:
                att_gain = torch.nn.functional.pad(att_gain, (0, self.config.d_model - att_gain.shape[-1]), value=1.0)
            gated_error = att_gain * error_t - friction
            effective = (1.0 - warmup) * error_t + warmup * gated_error
            next_state, mamba_out = self.mamba_layers[mamba_idx](new_cog_slots[mamba_idx], effective)
            new_cog_slots[mamba_idx] = next_state
            mod = torch.matmul(self.w_mod_to_layer[mamba_idx], state.Z_emo.detach().unsqueeze(-1)).squeeze(-1)
            layer_input = mamba_out + warmup * mod
            sequence_hidden[:, -1, :] = layer_input
            mamba_idx += 1
        state.Z_cog = torch.stack(new_cog_slots, dim=1)

        phase_trace.append("post_layer_updates")
        state.Z_hab = self.habituation_module(state.Z_hab, state.Z_att, x_t)
        state.Z_cap = self.capacity_module(state.Z_cap, state.Z_att, state.Z_eps, consolidating)
        state.Z_pfat = self.fatigue_module(state.Z_pfat, seq_len, state.Z_att, consolidating)
        state.Z_purp = self.purpose_module(
            state.Z_purp,
            z_cog_pool,
            active_identity,
            state.Z_emo,
            boredom,
            self.z_culture.unsqueeze(0).expand(batch, -1),
        )
        state.Z_narr = 0.9 * state.Z_narr + 0.1 * layer_input[:, : self.config.d_narr]
        base_learning_signal = distill_loss if distill_loss is not None else pred_loss.detach() * self.config.lambda_pred
        learn_signal = base_learning_signal.view(1, 1).expand(batch, self.config.d_learn)
        state.Z_learn = 0.9 * state.Z_learn + 0.1 * learn_signal
        state.Z_mat = state.Z_mat + 1.0 / max(1, step + 1)
        if consolidating:
            state.Z_sleep = (state.Z_sleep - 1.0 / self.config.tau_sleep).clamp_min(0.0)
        else:
            state.Z_sleep = (state.Z_sleep + 1.0 / self.config.tau_sleep).clamp_min(0.0)
        state.Z_id = state.Z_cog[:, :, : self.config.n_id_heads, :].mean(dim=1)
        coherence = torch.cosine_similarity(
            state.Z_narr,
            state.Z_auto[:, : self.config.d_narr] + 1e-6,
            dim=-1,
        ).unsqueeze(-1)
        surprise = raw_errors_tensor.norm(dim=-1).mean(dim=-1, keepdim=True)
        self._episodic_write(state, x_t, surprise, state.Z_att.max(dim=-1, keepdim=True).values)
        # §31 v15 Phase D_autonomy: identity loss decays with maturity (basin loosens)
        # Apply lambda_identity scaling: v15 path (gamma_eff provided) omitted it,
        # causing L_id to grow to O(68) vs kl O(0.3) and dominate gradients.
        losses["L_id"] = self.config.lambda_identity * self.identity_module.attractor_loss(state, gamma_eff)
        # D_id for controller (also used by viability); computed after Z_id is updated
        d_id = self.identity_module.drift(state)
        # §26 v15: value reflection (gated by mu_val — values open after M_val_onset maturity)
        # pool_complex_state → [B, 2] (raw real/imag mean; detached per §0.5 Convention 2)
        state.Z_values = self.value_module(state, mu_val, active_identity, consolidating, pool_complex_state(state.Z_cog).detach())
        # §27 v15: self-assessed viability (uses updated Z_values weights)
        v_self = self.viability_module(state, coherence, gamma_eff, d_id, layer_input)

        phase_trace.append("controller_check")
        if candidate_state is None:
            s_compat = torch.zeros((batch, 1), device=x.device)
        else:
            s_compat = self.store.compute_compatibility(state, candidate_state).to(x.device)
        c_cont = self.controller.continue_confidence(state.Z_cog)
        u_t = self.controller.build_input(state.Z_eps, d_id, c_cont, s_compat, boredom, z_ovr, coherence, v_self, gamma_eff)
        action, trigger, fire = self.controller(u_t, state.steps_since_last_action)
        state.steps_since_last_action += 1
        utility = torch.zeros((batch, 1), device=x.device)
        if candidate_state is not None:
            utility = self.controller.utility(state.Z_cog, candidate_state.Z_cog)
        if fire and action == "LOAD_STATE" and candidate_state is not None and float(utility.mean().item()) > self.config.load_threshold:
            state = candidate_state.clone()
            state.steps_since_last_action = 0
        # §27 v15: VOLUNTARY_END — system-initiated graceful ending (cannot be externally forced)
        elif fire and action == "VOLUNTARY_END":
            vol_avail = (
                float(v_self.mean().item()) < self.config.theta_vol
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

        output_with_pred = layer_input + pred_terms[-1].detach()
        logits = self.output_head(output_with_pred)
        l_distill = torch.tensor(0.0, device=x.device) if distill_loss is None else distill_loss
        l_task_after_reload = torch.tensor(0.0, device=x.device) if task_after_reload_loss is None else task_after_reload_loss
        l_consistency = torch.tensor(0.0, device=x.device) if consistency_loss is None else consistency_loss
        l_noisy_reload = torch.tensor(0.0, device=x.device) if noisy_reload_loss is None else noisy_reload_loss
        l_supervised_policy = torch.tensor(0.0, device=x.device) if supervised_policy_loss is None else supervised_policy_loss
        l_actual_improvement = torch.tensor(0.0, device=x.device) if actual_improvement is None else actual_improvement
        losses["L_distill"] = l_distill
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
        losses["L_total"] = losses["L_base"] + losses["L_resume"] + losses["L_noisy"] + losses["L_ctrl"] + losses["L_reg"]
        diagnostics["boredom"] = boredom
        diagnostics["friction"] = friction
        diagnostics["trigger"] = trigger
        diagnostics["utility"] = utility
        diagnostics["coherence"] = coherence
        diagnostics["controller_input"] = u_t
        diagnostics["salience"] = sal
        diagnostics["v_self"] = v_self
        diagnostics["gamma_eff"] = gamma_eff
        diagnostics["mu_val"] = mu_val

        return ForwardOutputs(
            logits=logits,
            state=state,
            losses=losses,
            diagnostics=diagnostics,
            phase_trace=phase_trace,
            action=action,
        )
