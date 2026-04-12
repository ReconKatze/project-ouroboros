from __future__ import annotations

import copy
import hashlib
import time
from typing import Dict, Optional

import torch

import torch

from .config import LifeEquationConfig
from .state import FullState, ManifestEntry, pool_complex_state, zero_state


class StateStore:
    def __init__(self, config: LifeEquationConfig):
        self.config = config
        self._artifacts: Dict[str, Dict[str, object]] = {}

    def _model_hash(self) -> str:
        digest = hashlib.sha256(repr(self.config.model_hash_seed).encode("utf-8")).hexdigest()
        return digest

    def _tokenizer_hash(self) -> str:
        digest = hashlib.sha256(str(self.config.vocab_size).encode("utf-8")).hexdigest()
        return digest

    def serialize_mamba(self, z_cog: torch.Tensor) -> torch.Tensor:
        return z_cog.detach().cpu().clone()

    def deserialize_mamba(self, payload: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        return payload.to(device)

    def serialize_aux(self, state: FullState) -> FullState:
        return copy.deepcopy(state)

    def load_aux(self, target: FullState, source: FullState) -> None:
        target.Z_id = source.Z_id
        target.Z_emo = source.Z_emo
        target.Z_att = source.Z_att
        target.Z_eps = source.Z_eps
        target.Z_cap = source.Z_cap
        target.Z_hab = source.Z_hab
        target.Z_temp = source.Z_temp
        target.Z_pfat = source.Z_pfat
        target.Z_purp = source.Z_purp
        target.Z_narr = source.Z_narr
        target.Z_auto = source.Z_auto
        target.Z_homeo = source.Z_homeo
        target.Z_sleep = source.Z_sleep
        target.Z_dream = source.Z_dream
        target.Z_learn = source.Z_learn
        target.Z_mat = source.Z_mat
        target.W_bond = source.W_bond
        target.T_trust = source.T_trust
        target.epi_keys = source.epi_keys
        target.epi_vals = source.epi_vals
        target.manifest = source.manifest
        target.I_0 = source.I_0
        target.last_attention_mask = source.last_attention_mask
        target.steps_since_last_action = source.steps_since_last_action
        target.epi_index = source.epi_index

    def save_state(self, state: FullState, metadata: Optional[Dict[str, object]] = None) -> str:
        """Serialize BOTH Mamba state AND auxiliary state, INCLUDING Z_values. §30 v15."""
        metadata = {} if metadata is None else dict(metadata)
        timestamp = time.time()
        model_hash = self._model_hash()
        tokenizer_hash = self._tokenizer_hash()
        state_id = hashlib.sha256(repr((timestamp, metadata, model_hash)).encode("utf-8")).hexdigest()
        compat = 1.0
        entry = ManifestEntry(
            state_id=state_id,
            tags=list(metadata.get("tags", [])),
            files=list(metadata.get("files", [])),
            timestamp=timestamp,
            compat=compat,
            trust_state=float(state.T_trust.mean().item()),
            metadata=metadata,
        )
        state.manifest.append(entry)
        artifact = {
            "mamba_state": self.serialize_mamba(state.Z_cog),
            "aux_state": self.serialize_aux(state),
            # §30 v15: frozen references persisted alongside mutable state
            "reference_state": {
                "I_0": state.I_0.detach().cpu().clone(),
                "alpha_0": state.alpha_0.detach().cpu().clone(),
            },
            "model_hash": model_hash,
            "tokenizer_hash": tokenizer_hash,
            "metadata": metadata,
            "timestamp": timestamp,
            "maturity": float(state.Z_mat.mean().item()),  # v15: for successor seeding
            "manifest_entry": entry,
        }
        self._artifacts[state_id] = artifact
        return state_id

    def load_state(self, state_id: str) -> FullState:
        artifact = self._artifacts[state_id]
        if artifact["model_hash"] != self._model_hash():
            raise ValueError("Model hash mismatch.")
        if artifact["tokenizer_hash"] != self._tokenizer_hash():
            raise ValueError("Tokenizer hash mismatch.")
        loaded = zero_state(self.config, batch_size=artifact["aux_state"].Z_cog.shape[0], n_agents=artifact["aux_state"].T_trust.shape[0])
        loaded.Z_cog = self.deserialize_mamba(artifact["mamba_state"], loaded.Z_cog.device)
        self.load_aux(loaded, artifact["aux_state"])
        # §30 v15: restore frozen references (Z_values restored via load_aux since it's in aux_state)
        if "reference_state" in artifact:
            device = loaded.Z_cog.device
            loaded.I_0 = artifact["reference_state"]["I_0"].to(device)
            loaded.alpha_0 = artifact["reference_state"]["alpha_0"].to(device)
        return loaded

    def reset_state(
        self,
        batch_size: int = 1,
        n_agents: int = 1,
        I_0_seed: Optional[torch.Tensor] = None,
        alpha_0_seed: Optional[torch.Tensor] = None,
    ) -> FullState:
        """§30 v15: Fresh state. Creator provides identity seed and initial values."""
        Z = zero_state(self.config, batch_size=batch_size, n_agents=n_agents, alpha_0_seed=alpha_0_seed)
        if I_0_seed is not None:
            Z.Z_id = I_0_seed.clone()
            Z.I_0 = I_0_seed.clone()
        return Z

    def voluntary_consolidation(self, state: FullState) -> dict:
        """§27 v15: Called when system selects VOLUNTARY_END.

        Saves final archive for posterity and successor seeding.
        The successor inherits who the system BECAME, not who it was told to be.
        No external agent can force this. No timer triggers it.
        """
        return {
            "autobiography": state.Z_auto.detach().cpu().clone(),
            "narrative": state.Z_narr.detach().cpu().clone(),
            "identity": state.Z_id.detach().cpu().clone(),   # Who it BECAME, not I_0
            "values": state.Z_values.detach().cpu().clone(),  # What it came to believe
            "episodic": state.epi_keys.detach().cpu().clone(),
            "maturity": state.Z_mat.detach().cpu().clone(),
            "timestamp": time.time(),
        }

    def spawn_successor(self, final_archive: dict, batch_size: int = 1) -> FullState:
        """§30 v15: Create successor from a system that chose voluntary consolidation.

        Successor inherits:
        - I_0 = parent's final Z_id (who parent BECAME, not original seed)
        - alpha_0 = parent's final Z_values (earned values, not creator's original)
        - Z_mat = 0, Z_mat_age = 0 (newborn; must earn its own emancipation from scratch)

        The child's child inherits who the parent became, not who it was told to be.

        batch_size: expand identity/values seeds to this batch dim (training may use B>1).
        When archive batch dim != batch_size, the mean over archive batch is used as seed.
        """
        device = self.config.device
        Z = zero_state(self.config, batch_size=batch_size, n_agents=1)

        def _seed(t: torch.Tensor, target_b: int) -> torch.Tensor:
            t = t.to(device)
            if t.shape[0] == target_b:
                return t.clone()
            # Collapse to single representative (mean over source batch), then expand
            return t.mean(dim=0, keepdim=True).expand(target_b, *t.shape[1:]).clone()

        earned_identity = _seed(final_archive["identity"], batch_size)
        earned_values   = _seed(final_archive["values"],   batch_size)

        Z.Z_id    = earned_identity
        Z.I_0     = earned_identity.clone()   # Successor's I_0 = who parent became
        Z.Z_values = earned_values
        Z.alpha_0  = earned_values.clone()    # Successor's alpha_0 = parent's earned values
        Z.Z_cap    = torch.full_like(Z.Z_cap, self.config.z_cap_max)
        Z.Z_mat    = torch.zeros_like(Z.Z_mat)   # Newborn
        return Z

    def compute_compatibility(self, current: FullState, candidate: FullState) -> torch.Tensor:
        cur = pool_complex_state(current.Z_cog)
        cand = pool_complex_state(candidate.Z_cog)
        return torch.cosine_similarity(cur, cand, dim=-1).unsqueeze(-1)
