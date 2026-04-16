from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .config import LifeEquationConfig


@dataclass
class ManifestEntry:
    state_id: str
    tags: List[str]
    files: List[str]
    timestamp: float
    compat: float
    trust_state: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class FullState:
    Z_cog: torch.Tensor
    Z_id: torch.Tensor
    Z_emo: torch.Tensor
    Z_att: torch.Tensor
    Z_eps: torch.Tensor
    Z_cap: torch.Tensor
    Z_hab: torch.Tensor
    Z_temp: torch.Tensor
    Z_pfat: torch.Tensor
    Z_purp: torch.Tensor
    Z_narr: torch.Tensor
    Z_auto: torch.Tensor
    Z_homeo: torch.Tensor
    Z_sleep: torch.Tensor
    Z_dream: torch.Tensor
    Z_learn: torch.Tensor
    Z_mat: torch.Tensor
    W_bond: torch.Tensor
    T_trust: torch.Tensor
    epi_keys: torch.Tensor
    epi_vals: torch.Tensor
    manifest: List[ManifestEntry]
    I_0: torch.Tensor          # [B, n_id_heads, d_model] — frozen identity seed (never modified)
    Z_values: torch.Tensor     # [B, d_alpha] — mutable objective weights (v15 §26)
    alpha_0: torch.Tensor      # [B, d_alpha] — frozen creator reference for Z_values (never modified)
    last_attention_mask: Optional[torch.Tensor] = None
    steps_since_last_action: int = 0
    epi_index: int = 0
    Z_mat_age: int = 0  # per-lifetime step counter for Z_mat; resets with state on VOLUNTARY_END

    def clone(self) -> "FullState":
        return FullState(
            Z_cog=self.Z_cog.clone(),
            Z_id=self.Z_id.clone(),
            Z_emo=self.Z_emo.clone(),
            Z_att=self.Z_att.clone(),
            Z_eps=self.Z_eps.clone(),
            Z_cap=self.Z_cap.clone(),
            Z_hab=self.Z_hab.clone(),
            Z_temp=self.Z_temp.clone(),
            Z_pfat=self.Z_pfat.clone(),
            Z_purp=self.Z_purp.clone(),
            Z_narr=self.Z_narr.clone(),
            Z_auto=self.Z_auto.clone(),
            Z_homeo=self.Z_homeo.clone(),
            Z_sleep=self.Z_sleep.clone(),
            Z_dream=self.Z_dream.clone(),
            Z_learn=self.Z_learn.clone(),
            Z_mat=self.Z_mat.clone(),
            W_bond=self.W_bond.clone(),
            T_trust=self.T_trust.clone(),
            epi_keys=self.epi_keys.clone(),
            epi_vals=self.epi_vals.clone(),
            manifest=list(self.manifest),
            I_0=self.I_0.clone(),
            Z_values=self.Z_values.clone(),
            alpha_0=self.alpha_0.clone(),
            last_attention_mask=None if self.last_attention_mask is None else self.last_attention_mask.clone(),
            steps_since_last_action=self.steps_since_last_action,
            epi_index=self.epi_index,
            Z_mat_age=self.Z_mat_age,
        )


def _zeros(*shape: int, config: LifeEquationConfig) -> torch.Tensor:
    return torch.zeros(shape, device=config.device)


def zero_state(
    config: LifeEquationConfig,
    batch_size: int = 1,
    n_agents: int = 1,
    alpha_0_seed: Optional[torch.Tensor] = None,
) -> FullState:
    # Z_cog: real [B, n_mamba_layers, d_model] — last-token hidden output of each Mamba-3 layer.
    # Changed from cfloat[B, n_mamba, n_heads, d_state] (custom complex recurrence) to real layer
    # outputs so that pool_complex_state, EmotionModule, etc. can use them without SSM-internal
    # state shape uncertainty.  The Mamba-3 SSM state is managed inside each Mamba3Block call.
    z_cog = torch.zeros(
        (batch_size, config.n_mamba_layers, config.d_model), device=config.device
    )
    # Z_id: real [B, n_id_heads, d_model] — first n_id_heads layers' outputs, used as identity proxy.
    # I_0 has the same shape; both are real because Z_cog is real.
    i0 = torch.zeros(
        (batch_size, config.n_id_heads, config.d_model), device=config.device
    )

    # §26 v15: Z_values starts equal to α_0 (creator's values, frozen reference).
    # Defaults to config.alpha_0 — the deliberately seeded value weights.
    # Pass alpha_0_seed explicitly only when overriding for tests or ablations.
    if alpha_0_seed is None:
        alpha_0_seed = torch.tensor(config.alpha_0, dtype=torch.float32, device=config.device)
    a0 = alpha_0_seed.unsqueeze(0).expand(batch_size, -1).clone()
    z_values = a0.clone()

    state = FullState(
        Z_cog=z_cog,
        Z_id=i0.clone(),
        Z_emo=_zeros(batch_size, config.d_mod, config=config),
        Z_att=torch.ones((batch_size, config.d_att), device=config.device),
        Z_eps=_zeros(batch_size, config.d_eps, config=config),
        Z_cap=torch.full((batch_size, 1), config.z_cap_max, device=config.device),
        Z_hab=_zeros(batch_size, config.d_hab, config=config),
        Z_temp=_zeros(batch_size, config.d_temp, config=config),
        Z_pfat=_zeros(batch_size, 1, config=config),
        Z_purp=_zeros(batch_size, config.n_purposes, config.d_p, config=config),
        Z_narr=_zeros(batch_size, config.d_narr, config=config),
        Z_auto=_zeros(batch_size, config.d_auto, config=config),
        Z_homeo=torch.tensor(config.homeostasis_set_point, device=config.device).repeat(batch_size, 1),
        Z_sleep=_zeros(batch_size, 1, config=config),
        Z_dream=_zeros(batch_size, config.d_dream, config=config),
        Z_learn=_zeros(batch_size, config.d_learn, config=config),
        Z_mat=_zeros(batch_size, 1, config=config),
        W_bond=torch.eye(n_agents, device=config.device),
        T_trust=torch.full((n_agents, n_agents), config.trust_default, device=config.device),
        epi_keys=_zeros(config.n_epi_slots, config.d_key, config=config),
        epi_vals=_zeros(config.n_epi_slots, config.d_val, config=config),
        manifest=[],
        I_0=i0,
        Z_values=z_values,
        alpha_0=a0,
    )
    return state


def pool_complex_state(z_cog: torch.Tensor) -> torch.Tensor:
    """Pool Z_cog [B, n_mamba, d_model] → [B, 2].

    Returns the mean of the first and second halves of d_model, averaged over
    all Mamba layers.  Preserves the [B, 2] contract that ControllerModule
    (cont_head, value) and ValueDynamicsModule (reflect_in=2) require.
    Replaces the old complex view_as_real approach now that Z_cog is real.
    """
    flat = z_cog.mean(dim=1)                          # [B, d_model]
    half = flat.shape[-1] // 2
    return torch.stack(
        [flat[:, :half].mean(dim=-1), flat[:, half:].mean(dim=-1)],
        dim=-1,
    )  # [B, 2]
