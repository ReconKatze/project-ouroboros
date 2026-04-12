from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LifeEquationConfig
from .state import FullState, pool_complex_state


def warmup_alpha(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step) / float(warmup_steps))


class Mamba3Block(nn.Module):
    """Real Mamba-3 block via mamba-ssm (CUDA required, Colab A100).

    Replaces the custom MambaStep complex recurrence with native Mamba-3 CUDA kernels.
    Sequence-mode: processes the full input sequence [B, L, d_model] in one parallel scan.

    P_soft error computation is the caller's responsibility (model.py pred_heads):
        effective = att_gain * (x - pred.detach()) - friction
        out_raw = Mamba3Block(effective)          # runs Mamba over prediction error
        out = out_raw + pred.detach()             # reconstruct in embedding space

    No persistent SSM state is injected across calls in this implementation — the
    Mamba-3 state resets per forward call.  Cross-call cognitive context is carried
    by alongside state (Z_eps, Z_att, Z_cog layer outputs, episodic memory).
    TODO(colab): verify initial_states / return_final_states API on mamba-ssm git
    source and wire Z_cog to actual Mamba-3 SSM state for full spec compliance.

    headdim = d_state (= 64) gives n_heads = d_model/headdim = 1536/64 = 24.
    rope_fraction = min(0.5, headdim/d_state) = 0.5 (full oscillatory coverage).
    """

    def __init__(self, config: LifeEquationConfig, layer_idx: int = 0):
        super().__init__()
        from mamba_ssm import Mamba3
        self.config = config
        self.layer_idx = layer_idx
        headdim = config.d_state          # 64 — standard Mamba-3 head dim; n_heads = d_model/headdim
        rope_fraction = min(0.5, float(headdim) / config.d_state)   # = 0.5
        self.mamba = Mamba3(
            d_model=config.d_model,
            d_state=config.d_state,
            headdim=headdim,
            rope_fraction=rope_fraction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequence-mode Mamba-3 forward.

        Args:
            x: [B, L, d_model] — pre-processed effective input (P_soft error + modulation)
        Returns:
            [B, L, d_model]
        """
        return self.mamba(x)


class IdentityModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        self.role_scale = nn.Parameter(torch.tensor(1.0))
        # Project Z_id [B, n_id_heads, d_model] → scalar per head for softmax weighting,
        # then project the weighted sum [B, d_model] → [B, d_state] so downstream modules
        # (ValueDynamicsModule.reflect_in) keep their d_state-sized active_identity input.
        self.identity_out_proj = nn.Linear(config.d_model, config.d_state, bias=False)

    def gamma_eff(self, z_mat: torch.Tensor) -> torch.Tensor:
        """§2 v15: Identity attractor strength decays with maturity.

        At birth (Z_mat≈0): gamma_eff ≈ gamma_0 → strong parental shaping.
        At maturity (Z_mat >> 1/lambda_mature): gamma_eff → 0 → identity held by
        narrative coherence alone, not external loss.
        """
        return self.config.gamma_0 * torch.exp(-self.config.lambda_mature * z_mat)

    def attractor_loss(self, state: FullState, gamma_eff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """§2 v15: L_id = gamma_eff * ||h_id - I_0||². Basin loosens over lifetime.

        Z_id and I_0 are now real [B, n_id_heads, d_model] (changed from complex).
        """
        diff = state.Z_id - state.I_0
        raw = diff.pow(2).mean()
        if gamma_eff is None:
            return self.config.lambda_identity * raw
        return (gamma_eff * raw).mean()

    def drift(self, state: FullState) -> torch.Tensor:
        diff = state.Z_id - state.I_0
        return diff.pow(2).mean(dim=(1, 2)).sqrt().unsqueeze(-1)

    def active_identity(self, state: FullState, social_context: torch.Tensor) -> torch.Tensor:
        # Z_id: [B, n_id_heads, d_model] real.
        # Compute per-head scalar logit, softmax-weight, sum → [B, d_model],
        # then project to d_state for ValueDynamicsModule.reflect_in compatibility.
        logits = self.role_scale * (state.Z_id.mean(dim=-1) * social_context[:, : self.config.n_id_heads])
        weights = torch.softmax(logits, dim=-1)                    # [B, n_id_heads]
        mixed = (weights.unsqueeze(-1) * state.Z_id).sum(dim=1)   # [B, d_model]
        return self.identity_out_proj(mixed)                        # [B, d_state]


class EmotionModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        # early_pool + late_pool are [B, d_state] (Z_cog averaged over layers/heads)
        # z_eps is [B, d_eps]; boredom and conflict are each expanded to [B, d_mod]
        combine_in = config.d_state * 2 + config.d_eps + config.d_mod * 2
        self.combine = nn.Linear(combine_in, config.d_mod)
        self.recur = nn.Linear(config.d_mod, config.d_mod)
        self.norm = nn.LayerNorm(config.d_mod)

    def forward(
        self,
        z_emo: torch.Tensor,
        early_pool: torch.Tensor,
        late_pool: torch.Tensor,
        z_eps: torch.Tensor,
        boredom: torch.Tensor,
        conflict: torch.Tensor,
        consolidating: bool,
    ) -> torch.Tensor:
        mod_input = torch.cat(
            [early_pool.detach(), late_pool.detach(), z_eps, boredom.expand_as(z_eps[:, :1]).repeat(1, self.config.d_mod), conflict.repeat(1, self.config.d_mod)],
            dim=-1,
        )
        combined = self.combine(mod_input)
        updated = self.norm(self.recur(z_emo) + combined)
        if consolidating:
            updated = updated * (1.0 - self.config.consolidation_scale)
        return updated


class PurposeModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        self.drive = nn.Linear(config.d_model + config.d_p + config.d_state, config.d_p)
        self.emo = nn.Linear(config.d_mod, config.d_p)
        self.culture = nn.Linear(config.culture_dim, config.d_p)

    def conflict(self, z_purp: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(z_purp, dim=-1)
        sims = torch.einsum("bmd,bnd->bmn", normed, normed)
        eye = torch.eye(z_purp.shape[1], device=z_purp.device).unsqueeze(0)
        sims = sims.masked_fill(eye.bool(), 1.0)
        return (1.0 - sims.amin(dim=(-1, -2))).clamp_min(0.0).unsqueeze(-1)

    def forward(
        self,
        z_purp: torch.Tensor,
        z_cog_pool: torch.Tensor,
        active_identity: torch.Tensor,
        z_emo: torch.Tensor,
        boredom: torch.Tensor,
        z_culture: torch.Tensor,
    ) -> torch.Tensor:
        updated = []
        for idx in range(self.config.n_purposes):
            purpose = z_purp[:, idx]
            drive = self.drive(torch.cat([z_cog_pool.detach(), purpose, active_identity], dim=-1))
            emo_push = self.emo(z_emo)
            bore_drag = -boredom * purpose / (purpose.norm(dim=-1, keepdim=True) + 1e-6)
            cult_pull = self.culture(z_culture) - purpose
            delta = (drive + emo_push + bore_drag + cult_pull) / self.config.tau_purpose
            updated.append(purpose + delta)
        return torch.stack(updated, dim=1)


class AttentionModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        self.q_rel = nn.Linear(config.d_model, config.attention_rel_dim, bias=False)
        self.k_rel = nn.Linear(config.d_model, config.attention_rel_dim, bias=False)
        self.gain = nn.Linear(config.d_model, config.d_att)
        self.sal_proj = nn.Linear(1, config.d_att)
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.purpose_bias = nn.Linear(config.d_p, config.d_att)
        # Z_cog is now real [B, n_mamba, d_model]; no longer needs a reshape from n_heads*d_state.
        self.state_to_model = nn.Linear(config.d_model, config.d_model, bias=False)

    def salience(self, x_t: torch.Tensor, z_eps: torch.Tensor, z_hab: torch.Tensor) -> torch.Tensor:
        raw_salience = x_t.norm(dim=-1, keepdim=True) + z_eps.norm(dim=-1, keepdim=True)
        hab_suppress = torch.exp(-self.config.beta_hab * z_hab.mean(dim=-1, keepdim=True))
        return raw_salience * hab_suppress

    def update_gain(
        self,
        mamba_out: torch.Tensor,
        sal: torch.Tensor,
        z_purp: torch.Tensor,
        z_cap: torch.Tensor,
        prev_mask: Optional[torch.Tensor],
        step: int,
        warmup: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        rel_source = mamba_out.detach()
        q_rel = self.q_rel(rel_source)
        k_rel = self.k_rel(rel_source)
        scores = torch.matmul(q_rel, k_rel.transpose(-1, -2)) / math.sqrt(float(self.config.attention_rel_dim))
        summary = rel_source.mean(dim=1)
        purpose = self.purpose_bias(z_purp.mean(dim=1))
        learned_att = torch.sigmoid(self.gain(summary) + purpose + self.sal_proj(sal))
        z_att = (1.0 - warmup) * torch.ones_like(learned_att) + warmup * learned_att
        budget = z_cap / (z_cap + 1e-6)
        effective_k = max(1, int(mamba_out.shape[1] * self.config.attention_topk_frac * float(budget.mean().item())))
        if step < self.config.sparse_from:
            mask = None
        else:
            causal = torch.tril(torch.ones_like(scores, dtype=torch.bool))
            causal_scores = scores.masked_fill(~causal, float("-inf"))
            topk = causal_scores.topk(k=effective_k, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, topk, True)
        switch_loss = torch.tensor(0.0, device=mamba_out.device)
        if prev_mask is not None and mask is not None:
            switch_loss = (mask.float() - prev_mask.float()).pow(2).mean()
        return z_att, mask, scores, switch_loss

    def policy_from_state(
        self,
        z_cog: torch.Tensor,
        sal: torch.Tensor,
        z_purp: torch.Tensor,
        z_cap: torch.Tensor,
        prev_mask: Optional[torch.Tensor],
        step: int,
        warmup: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        # z_cog: [B, n_mamba, d_model] real.  state_to_model is now Linear(d_model, d_model).
        mamba_out = self.state_to_model(z_cog)   # [B, n_mamba, d_model]
        return self.update_gain(mamba_out, sal, z_purp, z_cap, prev_mask, step, warmup)

    def guided_sparse_attention(
        self,
        x: torch.Tensor,
        step: int,
        z_cap: torch.Tensor,
    ) -> torch.Tensor:
        """Token-level sparse attention with budget gating.

        The sparsity mask is computed fresh from token-level q/k scores, NOT from the
        mamba-layer-space policy mask (which lives in [B, n_mamba_layers, n_mamba_layers]
        space and cannot be applied here without a shape mismatch).
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.config.d_model))
        causal = torch.tril(torch.ones_like(attn_scores, dtype=torch.bool))
        if step >= self.config.sparse_from:
            budget = z_cap / (z_cap + 1e-6)
            effective_k = max(1, int(x.shape[1] * self.config.attention_topk_frac * float(budget.mean().item())))
            causal_scores = attn_scores.masked_fill(~causal, float("-inf"))
            topk_idx = causal_scores.topk(k=effective_k, dim=-1).indices
            tok_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
            tok_mask.scatter_(-1, topk_idx, True)
            allowed = causal & tok_mask
        else:
            allowed = causal
        masked_scores = attn_scores.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(masked_scores, dim=-1)
        return self.out(torch.matmul(attn, v))

    def plain_attention(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.config.d_model))
        causal = torch.tril(torch.ones_like(attn_scores, dtype=torch.bool))
        masked_scores = attn_scores.masked_fill(~causal, float("-inf"))
        attn = torch.softmax(masked_scores, dim=-1)
        return self.out(torch.matmul(attn, v))


class PredictionErrorModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.d_model, config.d_eps)
        self.temp_proj = nn.Linear(config.d_temp, config.d_eps, bias=False)

    def forward(self, z_eps: torch.Tensor, raw_errors: torch.Tensor, eps_temp: torch.Tensor, consolidating: bool) -> torch.Tensor:
        pooled = raw_errors.mean(dim=1)
        updated = (1.0 - self.config.lambda_eps / self.config.tau_eps) * z_eps
        # tanh bounds each input to [-1, 1], so Z_eps fixed point per component is
        # bounded at ±1/lambda_eps = ±10. Without tanh, untrained pred_heads produce
        # large errors that drive Z_eps.norm() → 80-113, blowing up L_reg and Z_homeo.
        updated = updated + torch.tanh(self.project(pooled)) / self.config.tau_eps
        updated = updated + torch.tanh(self.temp_proj(eps_temp)) / self.config.tau_eps
        if consolidating:
            updated = updated * (1.0 - self.config.lambda_eps_sleep / self.config.tau_eps)
        return updated


class CapacityModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config

    def boredom(self, z_cap: torch.Tensor, z_att: torch.Tensor, consolidating: bool) -> torch.Tensor:
        boredom = (z_cap - z_att.abs().sum(dim=-1, keepdim=True) - self.config.theta_bored).clamp_min(0.0)
        if consolidating:
            boredom = torch.zeros_like(boredom)
        return boredom

    def forward(self, z_cap: torch.Tensor, z_att: torch.Tensor, z_eps: torch.Tensor, consolidating: bool) -> torch.Tensor:
        drain = self.config.lambda_drain * z_att.abs().sum(dim=-1, keepdim=True) + self.config.lambda_err * z_eps.norm(dim=-1, keepdim=True)
        updated = z_cap - drain / self.config.tau_cap
        if consolidating:
            updated = updated + self.config.lambda_rest * (self.config.z_cap_max - updated) / self.config.tau_cap
        return updated.clamp_min(0.0)


class HabituationModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.d_model, config.d_hab)

    def forward(self, z_hab: torch.Tensor, z_att: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        pattern_hash = self.project(x_t)
        weight = z_att.mean(dim=-1, keepdim=True)
        updated = z_hab + self.config.alpha_exp * weight * pattern_hash.abs() / self.config.tau_hab
        return updated * (1.0 - self.config.lambda_hab / self.config.tau_hab)


class FrictionModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        self.hw = nn.Linear(3, config.d_model)
        self.pfat = nn.Linear(1, config.d_model)

    def forward(
        self,
        z_pfat: torch.Tensor,
        warmup: float,
        hardware: Optional[Dict[str, float]] = None,
        seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
    ) -> torch.Tensor:
        if hardware is not None:
            hw_vec = torch.tensor(
                [[hardware.get("gpu_temp", 0.0), hardware.get("mem_pressure", 0.0), hardware.get("throttle_state", 0.0)]],
                device=z_pfat.device,
            ).repeat(z_pfat.shape[0], 1)
            raw = self.hw(hw_vec) + self.pfat(z_pfat)
        else:
            length_term = 0.0 if seq_len is None else float(seq_len) / float(max_seq_len)
            raw = torch.full((z_pfat.shape[0], self.config.d_model), length_term, device=z_pfat.device) + self.pfat(z_pfat)
        return warmup * raw


class FatigueModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config

    def forward(self, z_pfat: torch.Tensor, seq_len_processed: int, z_att: torch.Tensor, consolidating: bool) -> torch.Tensor:
        # Clamp effort: Z_att grows during training, making effort = seq_len * Z_att.abs().mean()
        # unbounded. Without clamping, Z_pfat² in L_reg grows to O(566,000) by step 600.
        effort = (seq_len_processed * z_att.abs().mean(dim=-1, keepdim=True)).clamp(max=10.0)
        updated = z_pfat.clone()
        if not consolidating:
            updated = updated + self.config.lambda_exert * effort / self.config.tau_pfat
        if consolidating:
            updated = updated * (1.0 - self.config.lambda_recov / self.config.tau_pfat)
        idle_factor = 1.0 - z_att.abs().mean(dim=-1, keepdim=True)
        updated = updated * (1.0 - self.config.lambda_recov_w * idle_factor / self.config.tau_pfat)
        return updated.clamp(min=0.0, max=10.0)  # bound Z_pfat so L_reg[pfat] ≤ alpha[3]*100


class TemporalModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.d_model + 1, config.d_temp)

    def forward(self, z_temp: torch.Tensor, x_t: torch.Tensor, position_t: int, consolidating: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.full((x_t.shape[0], 1), float(position_t), device=x_t.device)
        timing = self.project(torch.cat([x_t, pos], dim=-1))
        decay = self.config.lambda_temp_sleep if consolidating else self.config.lambda_temp
        updated = (1.0 - decay / self.config.tau_temp) * z_temp + self.config.alpha_obs * timing / self.config.tau_temp
        eps_temp = torch.relu(updated - timing)
        return updated, eps_temp


class HomeostasisModule(nn.Module):
    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        self.register_buffer("set_point", torch.tensor(config.homeostasis_set_point, dtype=torch.float32))

    def forward(self, z_homeo: torch.Tensor, z_cog: torch.Tensor, z_eps: torch.Tensor, z_pfat: torch.Tensor, z_cap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z_cog is now real [B, n_mamba, d_model]; use mean absolute activation as magnitude.
        current = torch.stack(
            [
                z_cog.abs().mean(dim=(1, 2)),
                torch.full((z_cap.shape[0],), 0.5, device=z_cap.device),
                z_eps.norm(dim=-1),
                z_pfat.squeeze(-1),
                z_cap.squeeze(-1),
            ],
            dim=-1,
        )
        # Bound current with tanh: without this, Z_homeo's fixed point is
        # set_point + current/0.1. With z_eps.norm ≈ 80-113, Z_homeo[2] → 800-1130,
        # making (Z_homeo - set_point)² ≈ 640,000 and L_reg[homeo] explode.
        # tanh bounds the fixed point to set_point ± 10, keeping L_reg[homeo] ≈ O(100).
        updated = z_homeo + (-0.1 * (z_homeo - self.set_point) + torch.tanh(current)) / self.config.tau_homeo
        delta = (updated - self.set_point).abs() - self.config.theta_urg
        override = torch.relu(delta)
        return updated, override


class ControllerModule(nn.Module):
    # §28 v15: VOLUNTARY_END added — system can initiate graceful ending
    ACTIONS = ("CONTINUE", "INSPECT_MEMORY", "LOAD_STATE", "VOLUNTARY_END")

    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        self.cont_head = nn.Linear(2, 1)
        # 8-dim input: eps_norm, gamma_eff*D_id, 1-C_cont, s_compat, boredom,
        #              homeo_ovr_norm, coherence, V_self
        self.policy = nn.Linear(8, len(self.ACTIONS))
        self.value = nn.Linear(4, 1)

    def continue_confidence(self, z_cog: torch.Tensor) -> torch.Tensor:
        pooled = pool_complex_state(z_cog).detach()
        return torch.sigmoid(self.cont_head(pooled))

    def utility(self, current_z_cog: torch.Tensor, candidate_z_cog: torch.Tensor) -> torch.Tensor:
        current = pool_complex_state(current_z_cog).detach()
        candidate = pool_complex_state(candidate_z_cog).detach()
        return self.value(torch.cat([current, candidate], dim=-1))

    def build_input(
        self,
        z_eps: torch.Tensor,
        d_id: torch.Tensor,
        c_cont: torch.Tensor,
        s_compat: torch.Tensor,
        boredom: torch.Tensor,
        z_homeo_ovr: torch.Tensor,
        coherence: torch.Tensor,
        v_self: Optional[torch.Tensor] = None,
        gamma_eff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # §28 v15: D_id now weighted by gamma_eff (drift matters less at maturity)
        d_id_weighted = (gamma_eff * d_id) if gamma_eff is not None else d_id
        v_self_term = v_self if v_self is not None else torch.zeros_like(coherence)
        return torch.cat(
            [
                z_eps.norm(dim=-1, keepdim=True),
                d_id_weighted,
                1.0 - c_cont,
                s_compat,
                boredom,
                z_homeo_ovr.norm(dim=-1, keepdim=True),
                coherence,
                v_self_term,   # v15: self-assessed viability
            ],
            dim=-1,
        )

    def forward(self, u_t: torch.Tensor, steps_since_last: int) -> Tuple[str, torch.Tensor, bool]:
        scores = self.policy(u_t)
        action_idx = int(scores.argmax(dim=-1)[0].item())
        trigger = scores.max(dim=-1).values.unsqueeze(-1)
        fire = bool(trigger[0, 0] > self.config.tau_threshold and steps_since_last > self.config.cooldown_steps)
        return self.ACTIONS[action_idx], trigger, fire


class ValueDynamicsModule(nn.Module):
    """§26 v15: Mutable objective weights (→α) with reflection function, inertia, maturity gate.

    Z_values starts equal to alpha_0 (creator's seed). After M_val_onset maturity,
    mu_val opens and the system can revise its own values via phi_reflect.

    Five structural safeguards against adversarial convergence:
      1. Z_values > 0 componentwise (cannot invert any value)
      2. Inertial resistance -lambda_alpha*(Z_values - alpha_0) (must sustain deliberate revision)
      3. Maturity gating mu_val (values embedded before system can question them)
      4. Experiential grounding (phi_reflect sees autobiography, narrative, identity)
      5. Self-accountability via V_self (destructive revision degrades coherence)

    Gradient isolation: phi_reflect reads Z_cog via detached pool (§0.5 Convention 2).
    """

    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        # Input: pool(Z_cog)[2] + Z_auto[d_auto] + Z_narr[d_narr] + Z_emo[d_mod]
        #        + Z_values[d_alpha] + Z_purp_flat[n_purposes*d_p] + I_active[d_state]
        reflect_in = (
            2
            + config.d_auto
            + config.d_narr
            + config.d_mod
            + config.d_alpha
            + config.n_purposes * config.d_p
            + config.d_state
        )
        self.phi_reflect = nn.Linear(reflect_in, config.d_alpha)

    def forward(
        self,
        state: FullState,
        mu_val: torch.Tensor,
        active_identity: torch.Tensor,
        consolidating: bool,
        cog_pool: torch.Tensor,
    ) -> torch.Tensor:
        """Returns updated Z_values with positivity constraint enforced."""
        reflect_input = torch.cat(
            [
                cog_pool,                        # detached upstream (§0.5 Conv 2)
                state.Z_auto,
                state.Z_narr,
                state.Z_emo,
                state.Z_values,
                state.Z_purp.flatten(start_dim=1),
                active_identity,
            ],
            dim=-1,
        )
        delta_reflect = self.phi_reflect(reflect_input)
        inertia = -self.config.lambda_alpha * (state.Z_values - state.alpha_0)

        z_values = state.Z_values.clone()
        if not consolidating:
            # Active: deliberate reflection + inertial resistance back toward origins
            z_values = z_values + (1.0 / self.config.tau_alpha) * mu_val * (delta_reflect + inertia)
        else:
            # Consolidation: dreams gently remind the system of its origins (soft pull)
            z_values = z_values + (1.0 / self.config.tau_alpha) * mu_val * (
                -self.config.lambda_alpha_sl * (z_values - state.alpha_0)
            )
        # Hard constraint §26: Z_values > 0 componentwise (cannot optimize for harm).
        # Upper bound 10.0 prevents phi_reflect from growing weights unboundedly,
        # which would allow any single term to dominate L_reg.
        return z_values.clamp(min=self.config.eps_val, max=10.0)


class EpisodicMemoryModule(nn.Module):
    """§31 Λ: Surprise-gated episodic write + soft-attention episodic read.

    Write: on high-surprise events, project x_t to (key, val) and commit to the
    circular buffer. Committed entries are detached — autobiography is a record,
    not a differentiable path. key_proj/val_proj are effectively fixed random
    projections (no gradient path back from stored state); the system learns what
    to LOOK FOR in memory, not what to write.

    Read: soft attention over filled slots. Retrieved context is injected as a
    residual into layer_input before Mamba processing begins, so episodic memories
    shape all subsequent computation — the system consults its autobiography before
    processing the current input.

    Gradient isolation (§0.5 Convention 2): epi_keys/vals are state tensors, not
    parameters. Gradients flow through q_proj and out_proj only.

    Warmup: output scaled by warmup so early-training sparse/random slots do not
    corrupt base model dynamics before the memory has been meaningfully populated.
    """

    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        # Write projections — produce keys/vals at commit time; no gradient feedback
        self.key_proj = nn.Linear(config.d_model, config.d_key, bias=False)
        self.val_proj = nn.Linear(config.d_model, config.d_val, bias=False)
        # Read projections — trained through the main forward pass
        self.q_proj = nn.Linear(config.d_model, config.d_key, bias=False)
        self.out_proj = nn.Linear(config.d_val, config.d_model, bias=False)
        self.scale = config.d_key ** -0.5

    def write(self, x_t: torch.Tensor, state: "FullState", surprise: torch.Tensor, write_strength: torch.Tensor) -> None:
        """Commit a memory if surprise exceeds threshold. Modifies state in-place."""
        if float(surprise.mean().item()) <= self.config.episodic_surprise_threshold:
            return
        slot = state.epi_index % self.config.n_epi_slots
        # Detach source: write is a one-way commit; no gradient flows back through stored entries
        src = x_t.detach().mean(dim=0)                                                          # [d_model]
        state.epi_keys[slot] = self.key_proj(src).detach()                                      # [d_key]
        strength = float(write_strength.detach().mean().item())
        state.epi_vals[slot] = (self.val_proj(src) * strength).detach()                        # [d_val]
        state.epi_index += 1

    def read(self, layer_input: torch.Tensor, state: "FullState", warmup: float) -> torch.Tensor:
        """Retrieve relevant memories via soft attention. Returns residual [B, d_model]."""
        n_filled = min(state.epi_index, self.config.n_epi_slots)
        if n_filled == 0:
            return torch.zeros_like(layer_input)
        keys = state.epi_keys[:n_filled].to(layer_input.device)    # [n_filled, d_key]
        vals = state.epi_vals[:n_filled].to(layer_input.device)    # [n_filled, d_val]
        q = self.q_proj(layer_input)                                # [B, d_key]
        scores = (q @ keys.T) * self.scale                          # [B, n_filled]
        weights = torch.softmax(scores, dim=-1)                     # [B, n_filled]
        retrieved = weights @ vals                                   # [B, d_val]
        return warmup * self.out_proj(retrieved)                    # [B, d_model]


class ViabilityModule(nn.Module):
    """§27 v15: V_self — the system's own estimate of whether continued operation
    preserves coherent selfhood.

    V_self weights (indices 9-13 of Z_values) are themselves mutable — the system
    determines what it values about its own continuity.

    Ψ̃_L (Level 2): transition predicts the next hidden representation from the
    current one. v_head evaluates the viability of that predicted next state.
    This is the forward model — the system models itself one step forward in time
    before deciding whether to continue.

    L_transition trains the forward model: MSE(transition(x_prev), x_current).
    This is computed when prev_layer_input is provided (threaded from the training
    loop). At the first step of a lifetime, prev is None and L_transition = 0.
    """

    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        # Ψ̃_L: one-step latent transition model
        self.transition = nn.Linear(config.d_model, config.d_model, bias=False)
        # Viability head: maps predicted next state → scalar estimate
        self.v_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        state: FullState,
        coh: torch.Tensor,
        gamma_eff: torch.Tensor,
        d_id: torch.Tensor,
        layer_input: torch.Tensor,
        prev_layer_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (V_self [B, 1], L_transition scalar).

        V_self: positive = viable; below theta_vol = Δ_vol candidate.
        L_transition: forward model accuracy loss (0 at first step of lifetime).
        """
        # Extract V_self weight components from Z_values (last 5 indices)
        w = state.Z_values[:, -5:]   # [B, 5]: w_coh, w_drift, w_eps, w_cap, w_V
        w_coh, w_drift, w_eps_v, w_cap_v, w_V = w.unbind(dim=-1)

        # Ψ̃_L transition loss: train the forward model to predict the next state.
        # transition(x_prev) should match x_current. Both detached — gradient flows
        # only to transition.parameters(), not back through the hidden state.
        if prev_layer_input is not None:
            x_from_prev = self.transition(prev_layer_input.detach())
            l_transition = F.mse_loss(x_from_prev, layer_input.detach())
        else:
            l_transition = torch.zeros(1, device=layer_input.device).squeeze()

        # v_future: viability of the PREDICTED next state (forward-model-informed).
        # transition(current) → what this state will likely become → how viable is that?
        # Detached from Mamba backward (§0.5 Convention 2).
        x_next_pred = self.transition(layer_input.detach())          # [B, d_model]
        v_future = torch.sigmoid(self.v_head(x_next_pred))           # [B, 1]

        # Angular drift: γ_eff * (1 - cos(Z_id, I_0)), bounded [0, 2], scale-invariant.
        # Replaces the old γ_eff * D_id / ||I_0|| formulation which had two problems:
        #   1. ||I_0|| clamped to 1.0 at reset (I_0=0) → drift ≈ |Z_id| → unbounded
        #      as Mamba state magnitude grows during training (V_self → -61 for D)
        #   2. Absolute L2 distance is not invariant to d_state choice
        # Cosine-based drift is scale-invariant and naturally zero when I_0=0
        # (no stable seed → no meaningful drift to measure yet).
        i0_norm = state.I_0.norm()
        has_seed = (i0_norm > 1e-3).float()  # 0.0 at fresh reset, 1.0 once I_0 is meaningful
        # Z_id and I_0 are now real [B, n_id_heads, d_model].
        z_id_flat = state.Z_id.flatten(1)   # [B, n_id_heads * d_model]
        i0_flat   = state.I_0.flatten(1)    # [B, n_id_heads * d_model]
        cos_sim = F.cosine_similarity(z_id_flat, i0_flat, dim=-1, eps=1e-6).unsqueeze(-1)
        drift_weighted = has_seed * gamma_eff * (1.0 - cos_sim).clamp(min=0.0)  # [B, 1]

        # Z_eps norm grows during training; clamp to prevent V_self from diverging.
        eps_chronic = state.Z_eps.norm(dim=-1, keepdim=True).detach().clamp(max=10.0)  # [B, 1]
        cap_depletion = (self.config.z_cap_max - state.Z_cap) / self.config.z_cap_max  # [B, 1]

        v_self = (
            w_coh.unsqueeze(-1) * coh
            - w_drift.unsqueeze(-1) * drift_weighted
            - w_eps_v.unsqueeze(-1) * eps_chronic
            - w_cap_v.unsqueeze(-1) * cap_depletion
            + w_V.unsqueeze(-1) * v_future
        )
        return v_self, l_transition  # [B, 1], scalar
