"""HybridChimeraModel: Converts a Transformer into a hybrid Mamba/Attention model.

Supports both GPT-NeoX (Pythia) and Qwen2 architectures via ModelAdapter pattern.
Replaces selected attention layers with Mamba blocks while keeping MLP, norms,
and residual connections intact.
"""

import torch
import torch.nn as nn

from chimera.utils.weight_conversion import extract_qkv


# ---------------------------------------------------------------------------
# Model adapters — abstract architecture differences behind a common interface
# ---------------------------------------------------------------------------

class GPTNeoXAdapter:
    """Adapter for GPTNeoXForCausalLM (Pythia)."""

    def __init__(self, base_model):
        self.config = base_model.config
        self.model_type = "gpt_neox"
        self.num_heads = base_model.config.num_attention_heads
        self.num_kv_heads = base_model.config.num_attention_heads  # MHA: same
        self.head_dim = base_model.config.hidden_size // self.num_heads

        # Module references
        self._gpt_neox = base_model.gpt_neox
        self.embed_tokens = base_model.gpt_neox.embed_in
        self.layers = base_model.gpt_neox.layers
        self.final_norm = base_model.gpt_neox.final_layer_norm
        self.lm_head = base_model.embed_out
        self.rotary_emb = base_model.gpt_neox.rotary_emb
        self.emb_dropout = base_model.gpt_neox.emb_dropout

    def get_attention(self, layer):
        return layer.attention

    def set_attention(self, layer, module):
        layer.attention = module

    def get_input_norm(self, layer):
        return layer.input_layernorm

    def create_masks(self, config, inputs_embeds, attention_mask, position_ids):
        """Return attention mask ready for layer consumption."""
        try:
            from transformers.modeling_utils import create_causal_mask
            mask = create_causal_mask(
                config=config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            if mask is not None and mask.dtype == torch.long:
                mask = mask.to(dtype=inputs_embeds.dtype)
            return mask
        except Exception:
            return None


class Qwen2Adapter:
    """Adapter for Qwen2ForCausalLM."""

    def __init__(self, base_model):
        self.config = base_model.config
        self.model_type = "qwen2"
        self.num_heads = base_model.config.num_attention_heads
        self.num_kv_heads = base_model.config.num_key_value_heads
        self.head_dim = base_model.config.hidden_size // self.num_heads

        # Module references
        self._model = base_model.model
        self.embed_tokens = base_model.model.embed_tokens
        self.layers = base_model.model.layers
        self.final_norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        self.rotary_emb = base_model.model.rotary_emb
        self.emb_dropout = nn.Identity()  # Qwen2 has no embedding dropout

    def get_attention(self, layer):
        return layer.self_attn

    def set_attention(self, layer, module):
        layer.self_attn = module

    def get_input_norm(self, layer):
        return layer.input_layernorm

    def create_masks(self, config, inputs_embeds, attention_mask, position_ids):
        """Return causal_mask_mapping dict for Qwen2 layers."""
        try:
            from transformers.modeling_utils import create_causal_mask
            mask = create_causal_mask(
                config=config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            if mask is not None and mask.dtype == torch.long:
                mask = mask.to(dtype=inputs_embeds.dtype)
            # Qwen2Model expects a dict keyed by layer type
            return {"full_attention": mask}
        except Exception:
            return {"full_attention": None}


def create_adapter(base_model):
    """Factory: create the correct adapter for the given model."""
    model_type = base_model.config.model_type
    if model_type == "gpt_neox":
        return GPTNeoXAdapter(base_model)
    elif model_type == "qwen2":
        return Qwen2Adapter(base_model)
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. "
            "Add a new adapter class to support this architecture."
        )


# ---------------------------------------------------------------------------
# Sink tokens
# ---------------------------------------------------------------------------

class SinkTokens(nn.Module):
    """Learnable sink tokens prepended to input embeddings.

    Gives attention layers initial tokens to attend to, acting as a
    scratch pad for attention patterns.
    """

    def __init__(self, num_sinks: int, hidden_size: int):
        super().__init__()
        self.num_sinks = num_sinks
        self.sinks = nn.Parameter(torch.randn(1, num_sinks, hidden_size) * 0.02)

    def forward(self, embeddings):
        """Prepend sink tokens. Returns [batch, num_sinks + seq_len, hidden_size]."""
        batch_size = embeddings.shape[0]
        return torch.cat([self.sinks.expand(batch_size, -1, -1), embeddings], dim=1)


# ---------------------------------------------------------------------------
# Attention wrapper — makes any Mamba block match attention's call signature
# ---------------------------------------------------------------------------

class MambaAttentionWrapper(nn.Module):
    """Wraps a Mamba block to match the attention module interface.

    The layer applies input_layernorm BEFORE calling this, so hidden_states
    are already normed. This wrapper must NOT re-norm or add residual —
    the layer handles both.

    Returns a 2-tuple (output, None) matching attention's (output, kv_cache).

    _last_out: detached copy of the Mamba output, set after every forward.
    MambaGuidedAttentionWrapper reads this to obtain the relevance signal
    without re-running or registering any backward hook.
    """

    def __init__(self, mamba_block: nn.Module):
        super().__init__()
        self.mamba = mamba_block
        self._last_out = None   # [B, L, d_model], populated by forward()

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                position_embeddings=None, past_key_values=None,
                use_cache=False, **kwargs):
        output = self.mamba(hidden_states)
        self._last_out = output.detach()
        return (output, None)


# ---------------------------------------------------------------------------
# Round 2H: Mamba-guided sparse attention
# ---------------------------------------------------------------------------

class MambaGuidedAttentionWrapper(nn.Module):
    """Round 2H: replaces a standard attention module.

    Learns to select the top-k most relevant key tokens using the output of
    the preceding Mamba layer as a relevance prior.  Only W_q_rel / W_k_rel
    are trained; the wrapped attention module stays frozen.

    Training schedule (controlled by step_ref[0]):
      Phase 0  (step < SPARSE_FROM):  Full attention.  W_q_rel / W_k_rel
                                       are trained via entropy regularisation
                                       on the relevance distribution.
      Phase 1  (step >= SPARSE_FROM): Sparse attention — only top-K_FRAC
                                       tokens are attended to per query.

    Go/no-go: relevance entropy should decrease over time (distribution
    concentrating) and top-k overlap with full-attention top-k ≥ 70%.

    If preceding_mamba is None (e.g. the very first attention layer has no
    Mamba layer before it), falls back to full attention with no relevance.
    """

    _D_REL       = 64     # relevance projection dimension
    _K_FRAC      = 0.25   # fraction of tokens selected in sparse mode
    _SPARSE_FROM = 2000   # step at which sparse attention activates
    _LAMBDA_ENT  = 0.005  # entropy regularisation weight

    def __init__(
        self,
        original_attn: nn.Module,
        d_model: int,
        align_accum: list,
        stats_accum: list,
        step_ref: list,
    ):
        """
        Args:
            original_attn: The frozen Qwen2 (or other) attention module.
            d_model:       Model hidden size.
            align_accum:   Shared mutable list; each forward appends one
                           entropy-reg loss tensor (training loop sums them).
            stats_accum:   Shared mutable list; each forward appends a dict
                           with diagnostic scalars.
            step_ref:      Single-element list [step] updated by training loop.
        """
        super().__init__()
        self.original_attn   = original_attn
        self.d_model         = d_model
        self.preceding_mamba = None   # set by build_variant_student after init

        d_rel = self._D_REL
        self.W_q_rel = nn.Linear(d_model, d_rel, bias=False)
        self.W_k_rel = nn.Linear(d_model, d_rel, bias=False)
        # Small-norm init: relevance starts near-uniform, shaped gradually
        nn.init.normal_(self.W_q_rel.weight, std=d_model ** -0.5)
        nn.init.normal_(self.W_k_rel.weight, std=d_model ** -0.5)

        self._align_accum = align_accum
        self._stats_accum = stats_accum
        self._step_ref    = step_ref

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rel_scores(self, relevance: torch.Tensor) -> torch.Tensor:
        """Causal relevance logits from Mamba output.

        Args:
            relevance: [B, L, d_model]
        Returns:
            scores: [B, L, L]  (future positions masked to -inf)
        """
        q = self.W_q_rel(relevance)                                  # [B, L, d_rel]
        k = self.W_k_rel(relevance)                                  # [B, L, d_rel]
        scores = (q @ k.transpose(-1, -2)) * (self._D_REL ** -0.5)  # [B, L, L]
        L = scores.shape[1]
        future = torch.triu(
            torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1
        )
        return scores.masked_fill(future, float('-inf'))

    def _sparse_mask(self, rel_scores: torch.Tensor) -> torch.Tensor:
        """Additive sparse attention mask [B, 1, L, L].

        Top-K_FRAC positions per query get 0.0 (pass through);
        everything else gets -inf (blocked).  Diagonal always passes.
        """
        B, L, _ = rel_scores.shape
        k = max(1, int(self._K_FRAC * L))
        topk_idx = rel_scores.topk(k, dim=-1).indices          # [B, L, k]
        mask = torch.full(
            (B, 1, L, L), float('-inf'),
            device=rel_scores.device, dtype=rel_scores.dtype
        )
        mask.scatter_(-1, topk_idx.unsqueeze(1), 0.0)
        # Self-attention always allowed
        diag = torch.arange(L, device=rel_scores.device)
        mask[:, 0, diag, diag] = 0.0
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
        **kwargs,
    ):
        step = self._step_ref[0]

        # No relevance available (first layer or inference) → full attention
        if (self.preceding_mamba is None
                or self.preceding_mamba._last_out is None):
            return self.original_attn(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings, **kwargs,
            )

        relevance  = self.preceding_mamba._last_out         # detached [B, L, d_model]
        rel_scores = self._rel_scores(relevance)            # [B, L, L]

        # Entropy regularisation — train W_q_rel / W_k_rel to produce
        # peaked distributions (low entropy = strong token preferences)
        if self.training:
            valid    = rel_scores.isfinite()
            safe     = rel_scores.masked_fill(~valid, -1e9)
            probs    = torch.softmax(safe, dim=-1)
            log_p    = torch.log_softmax(safe, dim=-1)
            ent_tok  = -(probs * log_p * valid).sum(dim=-1)  # [B, L]
            n_valid  = valid.float().sum(dim=-1).clamp(min=1)
            # Normalise by log(n_valid) so early positions don't dominate
            ent_norm = (ent_tok / n_valid.log().clamp(min=1e-9)).mean()
            self._align_accum.append(self._LAMBDA_ENT * ent_norm)

            with torch.no_grad():
                raw_ent = ent_tok.mean().item()
                phase   = 0 if step < self._SPARSE_FROM else 1
                self._stats_accum.append({"entropy": raw_ent, "phase": phase})

        if step >= self._SPARSE_FROM:
            sparse = self._sparse_mask(rel_scores)          # [B, 1, L, L]
            if attention_mask is not None:
                combined = attention_mask + sparse
            else:
                B, L, _ = hidden_states.shape
                causal = torch.triu(
                    torch.full((1, 1, L, L), float('-inf'),
                               device=hidden_states.device,
                               dtype=hidden_states.dtype),
                    diagonal=1,
                )
                combined = causal + sparse
            return self.original_attn(
                hidden_states, attention_mask=combined,
                position_ids=position_ids,
                position_embeddings=position_embeddings, **kwargs,
            )
        else:
            return self.original_attn(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings, **kwargs,
            )


# ---------------------------------------------------------------------------
# Main hybrid model
# ---------------------------------------------------------------------------

class HybridChimeraModel(nn.Module):
    """Hybrid Mamba/Attention model built from a pretrained Transformer.

    Supports Pythia (GPT-NeoX) and Qwen2 architectures.
    """

    def __init__(self, base_model, layer_plan, num_sinks=4, device="cpu"):
        """
        Args:
            base_model: A loaded HuggingFace causal LM model.
            layer_plan: List of dicts from build_layer_plan().
            num_sinks: Number of learnable sink tokens to prepend.
            device: Target device string — selects FallbackMamba vs RealMamba.
        """
        super().__init__()

        self.adapter = create_adapter(base_model)
        config = self.adapter.config
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_sinks = num_sinks
        self.vocab_size = config.vocab_size
        self.device_str = device

        # Expose model components via adapter
        self.embed_tokens = self.adapter.embed_tokens
        self.layers = self.adapter.layers
        self.final_norm = self.adapter.final_norm
        self.lm_head = self.adapter.lm_head
        self.rotary_emb = self.adapter.rotary_emb
        self.emb_dropout = self.adapter.emb_dropout

        # Sink tokens
        self.sink_tokens = SinkTokens(num_sinks, self.hidden_size)

        # Convert layers
        self.layer_plan = layer_plan
        self.conversion_log = []
        self._convert_layers()

    def _convert_layers(self):
        """Replace attention modules in layers marked as 'mamba'."""
        from chimera.models.real_mamba import create_mamba_block

        adapter = self.adapter
        num_heads = adapter.num_heads
        num_kv_heads = adapter.num_kv_heads
        head_dim = adapter.head_dim

        for plan in self.layer_plan:
            idx = plan["layer_idx"]
            layer = self.layers[idx]

            if plan["kind"] == "mamba":
                # Extract and tile QKV weights
                attn = adapter.get_attention(layer)
                (q_w, k_w, v_w), _ = extract_qkv(attn, num_heads, num_kv_heads, head_dim)

                # Create Mamba block (real on CUDA, fallback on CPU)
                d_state = plan.get("d_state", 16)
                mamba_block = create_mamba_block(self.hidden_size, d_state, self.device_str)

                # Initialize FallbackMamba weights from attention if applicable
                if hasattr(mamba_block, "init_from_attention"):
                    mamba_block.init_from_attention(q_weight=q_w, k_weight=k_w, v_weight=v_w)

                # Replace attention in the layer
                wrapper = MambaAttentionWrapper(mamba_block)
                adapter.set_attention(layer, wrapper)

                self.conversion_log.append(
                    f"  Layer {idx:2d}: -> {'RealMamba' if self.device_str != 'cpu' else 'FallbackMamba'}"
                    f" (d_state={d_state})"
                )
            else:
                self.conversion_log.append(f"  Layer {idx:2d}: attention (kept)")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the hybrid model.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (optional padding mask)

        Returns:
            Logits [batch, num_sinks + seq_len, vocab_size]
        """
        adapter = self.adapter
        config = self.config

        # Embed
        hidden_states = self.embed_tokens(input_ids)

        # Prepend sink tokens
        hidden_states = self.sink_tokens(hidden_states)
        total_seq_len = hidden_states.shape[1]

        # Extend padding mask for sink tokens
        if attention_mask is not None:
            sink_mask = torch.ones(
                attention_mask.shape[0], self.num_sinks,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([sink_mask, attention_mask], dim=1)

        # Position IDs covering full sequence (sinks + input)
        position_ids = torch.arange(
            total_seq_len, device=hidden_states.device
        ).unsqueeze(0)

        # Rotary position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # Model-specific causal mask
        mask_input = adapter.create_masks(config, hidden_states, attention_mask, position_ids)

        # Embedding dropout
        hidden_states = self.emb_dropout(hidden_states)

        # Run through layers
        if adapter.model_type == "qwen2":
            # Qwen2 expects a mask dict keyed by layer type
            layer_types = getattr(config, "layer_types", ["full_attention"] * len(self.layers))
            for i, layer in enumerate(self.layers):
                attn_mask = mask_input.get(layer_types[i]) if isinstance(mask_input, dict) else mask_input
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
        else:
            # GPT-NeoX
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=mask_input,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

        # Final norm + LM head
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def count_parameters(self):
        """Return (total_params, mamba_and_sink_params)."""
        total = sum(p.numel() for p in self.parameters())
        mamba_params = 0
        for plan in self.layer_plan:
            if plan["kind"] == "mamba":
                layer = self.layers[plan["layer_idx"]]
                attn = self.adapter.get_attention(layer)
                mamba_params += sum(p.numel() for p in attn.parameters())
        mamba_params += self.sink_tokens.sinks.numel()
        return total, mamba_params

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50):
        """Greedy generation for testing."""
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
