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

    SSM state persistence (three-tier API detection at init):
        1. return_final_states=True  — preferred, direct
        2. inference_params           — mamba-ssm ≤ 2.1 style; reads/writes state via
                                        InferenceParams.key_value_memory_dict[layer_idx]
        3. none                       — falls back to stateless with diagnostic printout

    After each call, _last_final_states holds the final SSM state (detached).
    model.forward() reads this attribute immediately after checkpoint() — before the
    use_reentrant=True backward replay can overwrite it — and stores it in state.Z_ssm.

    headdim = d_state (= 128) gives n_heads = d_model/headdim = 5120/128 = 40.
    rope_fraction = min(0.5, headdim/d_state) = 0.5 (full oscillatory coverage).
    """

    def __init__(self, config: LifeEquationConfig, layer_idx: int = 0):
        super().__init__()
        import inspect
        from mamba_ssm import Mamba3
        self.config = config
        self.layer_idx = layer_idx
        headdim = config.d_state          # 128 — Mamba-3 head dim; n_heads = d_model/headdim
        rope_fraction = min(0.5, float(headdim) / config.d_state)   # = 0.5
        # Pass layer_idx so the inference_params path can use it as the state dict key.
        try:
            self.mamba = Mamba3(
                d_model=config.d_model,
                d_state=config.d_state,
                headdim=headdim,
                rope_fraction=rope_fraction,
                layer_idx=layer_idx,
            )
        except TypeError:
            # Older Mamba3 signature without layer_idx.
            self.mamba = Mamba3(
                d_model=config.d_model,
                d_state=config.d_state,
                headdim=headdim,
                rope_fraction=rope_fraction,
            )

        # Probe the available state API once at construction (not per-forward-call).
        _params = set(inspect.signature(self.mamba.forward).parameters.keys())
        if "return_final_states" in _params:
            self._state_api = "return_final_states"
        elif "inference_params" in _params:
            self._state_api = "inference_params"
        else:
            self._state_api = "none"
            if layer_idx == 0:
                print(
                    f"[Mamba3Block] SSM state persistence unavailable — no known API found.\n"
                    f"  Mamba3.forward() params: {sorted(_params)}\n"
                    "  Share this output so the correct API path can be wired in."
                )

        # Side-channel for SSM state capture. Read by model.forward() immediately after
        # checkpoint() — before the use_reentrant=True backward replay overwrites it.
        self._last_final_states: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, initial_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sequence-mode Mamba-3 forward with SSM state persistence.

        Args:
            x:              [B, L, d_model] — P_soft error sequence
            initial_states: SSM state from the previous step (or None to start fresh)
        Returns:
            [B, L, d_model]  (SSM final state stored in self._last_final_states)
        """
        if self._state_api == "return_final_states":
            out, final = self.mamba(x, initial_states=initial_states, return_final_states=True)
            self._last_final_states = final.detach()

        elif self._state_api == "inference_params":
            # Inject initial_states and capture final states via InferenceParams.
            # seqlen_offset=0 → prefill mode: processes full sequence and writes
            # the resulting SSM state to key_value_memory_dict[layer_idx].
            try:
                from mamba_ssm.utils.generation import InferenceParams
            except ImportError:
                from mamba_ssm.utils.generation import InferenceCache as InferenceParams  # type: ignore[no-redef]
            inf_p = InferenceParams(max_seqlen=x.shape[1], max_batch_size=x.shape[0])
            inf_p.seqlen_offset = 0
            if initial_states is not None:
                inf_p.key_value_memory_dict[self.layer_idx] = initial_states
            out = self.mamba(x, inference_params=inf_p)
            final = inf_p.key_value_memory_dict.get(self.layer_idx)
            self._last_final_states = final.detach() if isinstance(final, torch.Tensor) else None

        else:
            out = self.mamba(x)

        return out


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

    def forward(self, u_t: torch.Tensor, steps_since_last: int) -> Tuple[str, torch.Tensor, bool, torch.Tensor]:
        raw_scores = self.policy(u_t)
        # Fire decision uses raw scores so logit biases don't suppress the trigger signal.
        trigger = raw_scores.max(dim=-1).values.unsqueeze(-1)
        fire = bool(trigger[0, 0] > self.config.tau_threshold and steps_since_last > self.config.cooldown_steps)
        # Apply per-action logit biases only for selection/softmax — not for the fire gate.
        # VOL_END bias: requires strong policy signal to win selection (complements step-gate).
        # INSPECT_MEMORY bias: reduces write rate; episodic writes are expensive and should
        # be reserved for genuinely surprising events, not fired at training-stage baseline rates.
        scores = raw_scores.clone()
        scores[:, self.ACTIONS.index("VOLUNTARY_END")] += self.config.vol_end_logit_bias
        scores[:, self.ACTIONS.index("INSPECT_MEMORY")] += self.config.inspect_memory_logit_bias
        # Action selection: sample during training, greedy argmax at inference.
        # Pure argmax + entropy regularization fails when entropy is already near-maximal
        # (ln(4)≈1.386): a persistent tiny logit bias causes 100% argmax repetition even
        # though the distribution is near-uniform — L_ctrl gradients are essentially zero
        # so the bias never clears.  Sampling diversifies actions proportional to learned
        # probabilities during training while preserving deterministic greedy behavior at
        # inference (eval/deployment).
        probs = F.softmax(scores, dim=-1)
        if self.training:
            action_idx = int(torch.multinomial(probs[0], 1).item())
        else:
            action_idx = int(scores.argmax(dim=-1)[0].item())
        # KL-to-prior regularization: pulls policy toward a CONTINUE-biased prior
        # instead of maximising uniform entropy.  KL(p || prior) is zero when
        # p == prior and positive elsewhere; subtracting -KL in L_ctrl becomes
        # +KL, so minimising L_ctrl drives p → prior.
        # Prior order matches ACTIONS: (CONTINUE, INSPECT_MEMORY, LOAD_STATE, VOLUNTARY_END).
        # Training-only: inference uses argmax on scores, unaffected by this term.
        log_probs = probs.log()
        log_prior = torch.tensor(
            list(self.config.ctrl_prior), dtype=probs.dtype, device=probs.device
        ).log()
        entropy = -(probs * (log_probs - log_prior)).sum(dim=-1).mean()
        return self.ACTIONS[action_idx], trigger, fire, entropy


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
        # Clamp V_self to a sane range — unbounded values (-100, etc.) from Z_values
        # drifting to their max ceiling (w_eps_v=10 × eps_chronic=10 = -100) fed into
        # the controller as extreme out-of-range signals before the onset fix.
        return v_self.clamp(min=-10.0, max=10.0), l_transition  # [B, 1], scalar


class SelfDynamicsModel(nn.Module):
    """§Ψ̃_L fully realized: GRU autoregressive self-monitoring trajectory model.

    Replaces the one-step linear stub in ViabilityModule with a genuine forward model
    that tracks its own state trajectory over time.

    Predicts the trajectory of 4 key self-monitoring scalars per step:
      dim 0: d_id      — identity drift  (from IdentityModule.drift)
      dim 1: eps_norm  — Z_eps norm      (from PredictionErrorModule)
      dim 2: c_cont    — continuation confidence (from ControllerModule)
      dim 3: v_self    — viability estimate  (from ViabilityModule)

    Per training step t:
      1. GRU input: (summary_{t-1} [4] + action_embed_{t-1} [4]) — what happened last step
      2. GRU hidden: Z_sdm [B, d_sdm] — accumulated trajectory context
      3. Output: pred_t [B, 4] — predictions for current step t
      4. L_self = MSE(pred_t, actual_t) — trains on its own next-state prediction error

    Integration with V_self (pessimism principle):
      At step t, before the controller fires, V_self is augmented:
        v_self_t = min(v_self_t, state.Z_sdm_pred[:, 3])
      where Z_sdm_pred contains the prediction for t made at t-1.
      This makes the controller and Δ_vol gate forward-looking: a trajectory that is
      about to degrade triggers action before the degradation is fully instantaneous.

    K-step lookahead for evaluation / deployment:
      lookahead(summary, h, k, action_idx) unrolls K GRU steps under a candidate
      action then CONTINUE, returning the full predicted trajectory.  Used by
      chimera/evaluation/runner.py and, at deployment, by the [THINK] window.

    Gradient isolation (§0.5 Convention 2):
      All inputs are detached. L_self trains only SelfDynamicsModel parameters.
      No gradient flows back into Mamba, identity, viability, or controller.
    """

    SUMMARY_DIM = 4    # d_id, eps_norm, c_cont, v_self
    N_ACTIONS   = 4    # matches ControllerModule.ACTIONS cardinality

    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        # Learned action embedding: gives the GRU a distinct embedding per action
        # so it can learn that LOAD_STATE resets drift while CONTINUE accumulates it.
        self.action_embed = nn.Embedding(self.N_ACTIONS, self.N_ACTIONS)
        gru_in = self.SUMMARY_DIM + self.N_ACTIONS
        self.gru = nn.GRUCell(gru_in, config.d_sdm)
        # Prediction head: map GRU hidden → next-step scalar estimates
        self.pred_head = nn.Linear(config.d_sdm, self.SUMMARY_DIM)

    def _pack_input(self, summary: torch.Tensor, action_idx) -> torch.Tensor:
        """[B, 4+4] — summary scalars + action embedding."""
        batch = summary.shape[0]
        if isinstance(action_idx, torch.Tensor):
            action_idx = int(action_idx.reshape(-1)[0].item())
        act = torch.full((batch,), int(action_idx), dtype=torch.long, device=summary.device)
        act_emb = self.action_embed(act)                      # [B, N_ACTIONS]
        return torch.cat([summary, act_emb], dim=-1)          # [B, 8]

    def step(
        self,
        summary: torch.Tensor,   # [B, 4]
        action_idx: int,
        h: torch.Tensor,         # [B, d_sdm]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single GRU step. Returns (pred_next [B, 4], h_next [B, d_sdm])."""
        x = self._pack_input(summary, action_idx)
        h_next = self.gru(x, h)
        pred_next = self.pred_head(h_next)
        return pred_next, h_next

    def lookahead(
        self,
        summary: torch.Tensor,       # [B, 4] — current actual summary, already detached
        h: torch.Tensor,             # [B, d_sdm] — current GRU hidden, already detached
        k: int,
        action_idx_step0: int = 0,   # action for step 0 (0 = CONTINUE)
    ) -> List[torch.Tensor]:
        """Unroll K steps from current state under candidate action.

        Steps 1..K-1 assume CONTINUE (idx=0) — lookahead simulates "what if I keep going."
        Returns list of K predicted [B, 4] tensors.
        All computation is torch.no_grad().
        """
        preds: List[torch.Tensor] = []
        s = summary.detach()
        h_curr = h.detach()
        with torch.no_grad():
            for i in range(k):
                a = action_idx_step0 if i == 0 else 0   # 0 = CONTINUE
                pred, h_curr = self.step(s, a, h_curr)
                preds.append(pred)
                s = pred
        return preds

    def forward(
        self,
        summary_t: torch.Tensor,    # [B, 4] — current actuals (detached upstream)
        prev_pred: torch.Tensor,    # [B, 4] — predictions made at t-1 (state.Z_sdm_pred)
        action_idx: int,            # action taken at t-1 (state.prev_action_idx)
        h_prev: torch.Tensor,       # [B, d_sdm] — GRU hidden from previous step
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-training-step update.

        Returns:
            pred_next [B, 4]  — predictions for t+1 (store in state.Z_sdm_pred)
            h_next    [B, d_sdm] — updated GRU hidden (store in state.Z_sdm)
            l_self    scalar  — L_self = MSE(prev_pred, summary_t)
        """
        # L_self: compare t-1 predictions to current actuals.
        # Zero on first step of lifetime (prev_pred=zeros from zero_state) so we don't
        # penalise random initialisation before the GRU has seen any real trajectory.
        has_prev = (prev_pred.detach().abs().sum() > 1e-6).float()
        l_self = has_prev * F.mse_loss(prev_pred, summary_t.detach())

        # GRU step: input = (t-1 actuals, t-1 action) → pred_t+1, h_t
        # We feed summary_t as the "observed outcome" the GRU conditions on,
        # with the action that was taken to arrive here.
        pred_next, h_next = self.step(summary_t.detach(), action_idx, h_prev.detach())

        return pred_next, h_next, l_self


# ─────────────────────────────────────────────────────────────────────────── #
# §N  Narrative coherence                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class NarrativeModule(nn.Module):
    """§N: Narrative coherence — running self-story and identity expectation.

    Z_narr (d_narr) — running narrative: GRU integrating recent cognitive state.
      Updated every step using late_pool (compressed Z_cog) as input.

    Z_auto (d_auto = d_narr) — identity-grounded expectation: slow EMA toward
      a projection of active_identity.  Answers "what should the narrative look
      like if identity is stable?"

    Coherence = cos_sim(Z_narr, Z_auto) — measures how well the lived narrative
      aligns with identity expectations.  Feeds ViabilityModule (V_self), the
      controller trigger, and L_reg (alpha_N * coherence).

    During consolidation: Z_narr receives a gentle residual from DreamModule so
      replayed episodic memories can integrate into the narrative during sleep.
      Z_auto is not modified during consolidation — the identity reference holds.

    Gradient isolation (§0.5 Convention 2): all inputs are detached.
    """

    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        # Narrative GRU: integrates late_pool [B, d_state] → Z_narr [B, d_narr]
        self.narr_gru = nn.GRUCell(config.d_state, config.d_narr)
        # Auto-expectation: maps active_identity [B, d_state] → identity narrative [B, d_narr]
        # Slow EMA (tau_auto) drives Z_auto toward this projection each step.
        self.id_to_auto = nn.Linear(config.d_state, config.d_narr, bias=False)

    def forward(
        self,
        z_narr: torch.Tensor,            # [B, d_narr]
        z_auto: torch.Tensor,            # [B, d_auto]  (d_auto == d_narr)
        late_pool: torch.Tensor,         # [B, d_state] — detached upstream
        active_identity: torch.Tensor,   # [B, d_state] — detached upstream
        dream_residual: Optional[torch.Tensor],  # [B, d_narr] or None
        consolidating: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cast state tensors to module dtype (handles float32 state under bfloat16 autocast)
        wdtype = self.narr_gru.weight_ih.dtype
        z_narr = z_narr.to(dtype=wdtype)
        z_auto = z_auto.to(dtype=wdtype)
        # Narrative update via GRU (integrates recent cognitive history)
        z_narr_new = self.narr_gru(late_pool.detach(), z_narr)
        if consolidating and dream_residual is not None:
            # Sleep: gentle nudge from dream replay into the narrative
            # Cast dream_residual to wdtype (DreamModule returns float32 after .float() below)
            z_narr_new = z_narr_new + 0.05 * dream_residual.detach().to(dtype=wdtype)

        # Auto-expectation: slow EMA toward identity-grounded narrative target
        target = self.id_to_auto(active_identity.detach())              # [B, d_narr]
        tau = getattr(self.config, "tau_auto", 64.0)
        z_auto_new = z_auto + (target - z_auto[:, : self.config.d_narr]) / tau
        # If d_auto > d_narr, preserve the trailing slice unchanged
        if z_auto.shape[-1] > self.config.d_narr:
            z_auto_new = torch.cat([z_auto_new, z_auto[:, self.config.d_narr:]], dim=-1)

        # Return float32 — state tensors must stay float32 (L_reg multiplies against float32 alpha)
        return z_narr_new.float(), z_auto_new.float()


# ─────────────────────────────────────────────────────────────────────────── #
# §D  Sleep pressure                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class SleepModule(nn.Module):
    """§D: Activity-weighted sleep pressure accumulator.

    Replaces the fixed-rate tick (±1/tau_sleep per step) with a learned
    activity-dependent rate.  The accumulation speed is proportional to how
    hard the system is working: high attention + high prediction error + high
    fatigue → faster sleep pressure buildup.

    Sleep pressure contributes to L_reg via alpha_D * Z_sleep, incentivising
    the system toward consolidation when Z_sleep is elevated.  During
    consolidation Z_sleep releases at a fixed rate (constant commitment length).

    Gradient isolation: all inputs detached.
    """

    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        # Map (att_activity, eps_norm, fatigue) → accumulation rate scalar ∈ [0, 1]
        self.pressure_proj = nn.Linear(3, 1, bias=True)

    def forward(
        self,
        z_sleep: torch.Tensor,   # [B, 1]
        z_att: torch.Tensor,     # [B, d_att]
        z_eps: torch.Tensor,     # [B, d_eps]
        z_pfat: torch.Tensor,    # [B, 1]
        consolidating: bool,
    ) -> torch.Tensor:
        # Cast state tensor to module dtype (handles float32 state under bfloat16 autocast)
        z_sleep = z_sleep.to(dtype=self.pressure_proj.weight.dtype)
        if consolidating:
            return (z_sleep - 1.0 / self.config.tau_sleep).clamp_min(0.0).float()
        # Activity triplet: each component normalised to [0, 1]
        att  = z_att.abs().mean(dim=-1, keepdim=True).detach().clamp(max=5.0) / 5.0
        eps  = z_eps.norm(dim=-1, keepdim=True).detach().clamp(max=10.0) / 10.0
        pfat = z_pfat.detach().clamp(max=10.0) / 10.0
        pressure = torch.sigmoid(self.pressure_proj(torch.cat([att, eps, pfat], dim=-1)))  # [B, 1]
        # Return float32 — state tensors must stay float32 (L_reg multiplies against float32 alpha)
        return (z_sleep + pressure / self.config.tau_sleep).clamp(min=0.0, max=10.0).float()


# ─────────────────────────────────────────────────────────────────────────── #
# §θ  Dream dynamics                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class DreamModule(nn.Module):
    """§θ: Episodic replay during consolidation — dream content.

    Awake:         Z_dream slowly fades toward zero (waking cognition clears it).
    Consolidating: stored episodic values are soft-averaged and compressed into
                   Z_dream (a 'dream content' vector).  This represents the
                   system integrating its autobiography during sleep.

    Z_dream is then available as a narrative residual passed to NarrativeModule:
    during consolidation, replayed memories gently nudge Z_narr so episodic
    experience is woven into the ongoing self-story.

    Gradient isolation: epi_vals are stored detached; dream_to_narr is the only
    trained path and its output is used as a small residual (+0.05 weight).
    """

    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        # Compress episodic value buffer mean → dream content [B, d_dream]
        self.val_to_dream = nn.Linear(config.d_val, config.d_dream, bias=False)
        # Project dream content → narrative residual [B, d_narr]
        self.dream_to_narr = nn.Linear(config.d_dream, config.d_narr, bias=False)

    def forward(
        self,
        z_dream: torch.Tensor,  # [B, d_dream]
        state,                   # FullState (reads epi_vals, epi_index)
        consolidating: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (z_dream_new [B, d_dream], narr_residual [B, d_narr] or None)."""
        # Cast state tensor to module dtype (handles float32 state under bfloat16 autocast)
        wdtype  = self.val_to_dream.weight.dtype
        z_dream = z_dream.to(dtype=wdtype)
        if not consolidating:
            # Awake: dream state fades (τ_dream ≈ 128 steps)
            # Return float32 — state tensors must stay float32 (L_reg multiplies against float32 alpha)
            return (z_dream * (1.0 - 1.0 / 128.0)).float(), None

        n_filled = min(state.epi_index, state.epi_keys.shape[0])
        if n_filled == 0:
            return z_dream.float(), None

        # Uniform replay: mean over all stored episodic values (no query — dreams browse broadly)
        vals    = state.epi_vals[:n_filled].to(device=z_dream.device, dtype=wdtype)  # [n_filled, d_val]
        replay  = self.val_to_dream(vals.mean(dim=0, keepdim=True))     # [1, d_dream]
        replay  = replay.expand(z_dream.shape[0], -1)                   # [B, d_dream]

        # EMA toward replay content (τ = 16 consolidation steps)
        z_dream_new   = z_dream + (replay.detach() - z_dream) / 16.0
        narr_residual = self.dream_to_narr(z_dream_new.detach())         # [B, d_narr]
        # Return float32 — state tensors must stay float32 (L_reg multiplies against float32 alpha)
        return z_dream_new.float(), narr_residual.float()


# ─────────────────────────────────────────────────────────────────────────── #
# §T_ij  Trust dynamics                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class TrustModule(nn.Module):
    """§T_ij / B_ij: Trust dynamics (single-agent: epistemic self-trust).

    T_trust tracks how reliable and self-consistent the system's reasoning has
    been.  It is a slow-moving signal: consistent low prediction error + stable
    identity + coherent narrative → trust converges toward 1.  Fragmented state
    → trust drifts toward 0.

    T_trust feeds L_reg as: -alpha_T * T_trust.mean()
    (higher trust → lower loss; the system is rewarded for being trustworthy).

    Multi-agent extension: the full T_ij matrix update (per-pair directional
    trust based on shared prediction error) is the next step and will be added
    when multi-agent training begins.  For now W_bond stays as an identity matrix.

    Gradient isolation: all inputs detached.
    """

    def __init__(self, config: LifeEquationConfig):
        super().__init__()
        self.config = config
        # Map (eps_reliability, coherence, identity_stability) → trust target ∈ [0, 1]
        self.trust_proj = nn.Linear(3, 1, bias=True)

    def forward(
        self,
        t_trust: torch.Tensor,    # [n_agents, n_agents]  (single-agent: [1, 1])
        z_eps: torch.Tensor,      # [B, d_eps]
        coherence: torch.Tensor,  # [B, 1]
        d_id: torch.Tensor,       # [B, 1]
    ) -> torch.Tensor:
        # Cast state tensor to module dtype (handles float32 state under bfloat16 autocast)
        t_trust = t_trust.to(dtype=self.trust_proj.weight.dtype)
        # Reliability inputs — normalised to [0, 1], all detached
        eps_rel   = 1.0 - z_eps.norm(dim=-1, keepdim=True).detach().clamp(max=10.0) / 10.0  # [B,1]
        coh_norm  = coherence.detach().clamp(-1.0, 1.0)                                       # [B,1]
        id_stable = 1.0 - d_id.detach().clamp(max=5.0) / 5.0                                 # [B,1]

        trust_input  = torch.cat([eps_rel, coh_norm, id_stable], dim=-1)           # [B, 3]
        trust_target = torch.sigmoid(self.trust_proj(trust_input)).mean()           # scalar

        tau = getattr(self.config, "tau_trust", 32.0)
        # Return float32 — state tensors must stay float32 (L_reg multiplies against float32 alpha)
        return (t_trust + (trust_target - t_trust) / tau).clamp(0.0, 1.0).float()


# ── Block Attention Residuals (arXiv:2603.15031) ─────────────────────────────


class BlockAttnResidual(nn.Module):
    """Block Attention Residuals (Block AttnRes, Kimi Team, arXiv:2603.15031).

    Replaces uniform depth-wise residual accumulation with learned softmax attention
    over block-level hidden-state summaries.  Standard residuals accumulate all prior
    layer outputs with fixed unit weights, causing hidden-state magnitude growth with
    depth (PreNorm dilution).  AttnRes lets each layer selectively reweight earlier
    representations via a single learned pseudo-query w_l per layer.

    Adaptation for life_eq_v3
    --------------------------
    The 28 layers are naturally partitioned into 4 blocks by the attention anchors
    {0, 9, 18, 27}.  We apply Block AttnRes once per inter-block boundary (before
    attention anchors 9, 18, 27) rather than per sub-layer.  The block summary b_k
    is the seq tensor immediately after attention anchor k.

    At anchor k ≥ 1 the input to that attention layer is replaced by:

        V  = [b_0, b_1, ..., b_{k-1}, seq_current]   shape [k+1, B, T, d]
        K  = RMSNorm(V)                               per-slot key normalisation
        α  = softmax(w_k · K, dim=0)                  [k+1, B, T] depth-wise weights
        h  = sum_i α_i * V_i                          [B, T, d]   new seq

    Zero initialization of w_k → uniform 1/(k+1) weights at step 0, exactly
    matching an equal-weight average — no training instability at startup.

    Parameters
    -----------
    n_anchors : int
        Number of attention anchors (4 for life_eq_v3).  We allocate n_anchors
        pseudo-query slots; index 0 is never called (no history at first anchor).

    Parameter cost: n_anchors × d_model ≈ 4 × 1536 = 6 144 scalars.
    Arithmetic:     O(N d) per token per anchor, N = number of completed blocks.
    """

    def __init__(self, config: LifeEquationConfig, n_anchors: int = 4):
        super().__init__()
        self.n_anchors = n_anchors
        # Pseudo-query per anchor.  Zero-init → uniform softmax at step 0.
        self.w_q = nn.ParameterList([
            nn.Parameter(torch.zeros(config.d_model))
            for _ in range(n_anchors)
        ])
        # Shared RMSNorm normalises key magnitudes so no single block dominates.
        # nn.RMSNorm(d_model) applies along the last dimension — works on any shape.
        self.key_norm = nn.RMSNorm(config.d_model)

    def forward(
        self,
        anchor_idx: int,
        block_summaries: List[torch.Tensor],  # k completed summaries [B, T, d] each
        partial_block: torch.Tensor,          # current seq [B, T, d]
    ) -> torch.Tensor:
        """Return attention-weighted combination of block summaries + current seq.

        anchor_idx  : which anchor slot's pseudo-query to use (1, 2, or 3).
        block_summaries : seq values saved after each previous attention anchor.
        partial_block   : current seq at this anchor (before the attention layer).
        """
        # V: [k+1, B, T, d_model] — all sources including current partial block
        V = torch.stack(block_summaries + [partial_block], dim=0)
        # Normalise keys; RMSNorm broadcasts over leading dims, acts on last (d_model)
        K = self.key_norm(V)
        # Depth-wise attention logits: [k+1, B, T]
        logits = torch.einsum("d, n b t d -> n b t", self.w_q[anchor_idx], K)
        weights = torch.softmax(logits, dim=0)          # [k+1, B, T]
        # Weighted sum: [B, T, d_model]
        return torch.einsum("n b t, n b t d -> b t d", weights, V)
