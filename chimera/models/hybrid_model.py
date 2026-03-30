"""HybridChimeraModel: Converts a Transformer into a hybrid Mamba/Attention model.

Replaces selected attention layers with FallbackMamba blocks while keeping
MLP, norms, and the layer's residual structure intact.
"""

import torch
import torch.nn as nn

from chimera.models.fallback_mamba import FallbackMamba
from chimera.utils.weight_conversion import extract_qkv_pythia


class SinkTokens(nn.Module):
    """Learnable sink tokens prepended to input embeddings.

    These give attention layers initial tokens to attend to,
    acting as a scratch pad for attention patterns.
    """

    def __init__(self, num_sinks: int, hidden_size: int):
        super().__init__()
        self.sinks = nn.Parameter(torch.randn(1, num_sinks, hidden_size) * 0.02)

    def forward(self, embeddings):
        """Prepend sink tokens to embeddings.

        Args:
            embeddings: [batch, seq_len, hidden_size]

        Returns:
            [batch, num_sinks + seq_len, hidden_size]
        """
        batch_size = embeddings.shape[0]
        sinks = self.sinks.expand(batch_size, -1, -1)
        return torch.cat([sinks, embeddings], dim=1)


class MambaAttentionWrapper(nn.Module):
    """Wraps FallbackMamba to match the interface GPTNeoXAttention expects.

    GPTNeoXLayer applies input_layernorm BEFORE calling attention,
    so hidden_states arriving here are already normed. The wrapper
    must NOT re-norm. It also must NOT add residual - the layer does that.

    Returns a 2-tuple (output, None) matching GPTNeoXAttention's return.
    """

    def __init__(self, fallback_mamba: FallbackMamba):
        super().__init__()
        self.mamba = fallback_mamba

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                head_mask=None, layer_past=None, use_cache=False,
                output_attentions=False, position_embeddings=None, **kwargs):
        """Match GPTNeoXAttention.forward() signature.

        Input hidden_states are ALREADY normed by the layer's input_layernorm.
        We apply the gated linear unit directly, no extra norm or residual.
        """
        # hidden_states is already normed by the layer
        gate = torch.sigmoid(self.mamba.in_proj_b(hidden_states))
        mixed = self.mamba.in_proj_x(hidden_states) * gate
        output = self.mamba.out_proj_c(mixed)

        # GPTNeoXAttention returns (attn_output, present_key_value)
        return (output, None)


class HybridChimeraModel(nn.Module):
    """Hybrid Mamba/Attention model built from a pretrained Transformer.

    Takes a GPTNeoXForCausalLM and replaces selected attention layers
    with FallbackMamba blocks. The MLP, norms, and residual connections
    in each layer are preserved.
    """

    def __init__(self, base_model, layer_plan, num_sinks=4):
        """
        Args:
            base_model: A loaded GPTNeoXForCausalLM model.
            layer_plan: List of dicts from build_layer_plan().
            num_sinks: Number of learnable sink tokens to prepend.
        """
        super().__init__()

        config = base_model.config
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_sinks = num_sinks
        self.vocab_size = config.vocab_size

        # Keep reference to the base model's components
        self.gpt_neox = base_model.gpt_neox
        self.lm_head = base_model.embed_out

        # Sink tokens
        self.sink_tokens = SinkTokens(num_sinks, self.hidden_size)

        # Convert layers according to plan
        self.layer_plan = layer_plan
        self._convert_layers(config)

    def _convert_layers(self, config):
        """Replace attention modules in layers marked as 'mamba'."""
        num_heads = config.num_attention_heads
        head_size = config.hidden_size // num_heads
        layers = self.gpt_neox.layers

        self.conversion_log = []

        for plan in self.layer_plan:
            idx = plan["layer_idx"]
            layer = layers[idx]

            if plan["kind"] == "mamba":
                # Extract QKV weights from the attention module
                (q_w, k_w, v_w), (q_b, k_b, v_b) = extract_qkv_pythia(
                    layer.attention, num_heads, head_size
                )

                # Create FallbackMamba and initialize from attention weights
                # Note: we don't copy the layer norm into FallbackMamba because
                # GPTNeoXLayer applies input_layernorm before calling attention
                fallback = FallbackMamba(self.hidden_size)
                fallback.init_from_attention(
                    q_weight=q_w,
                    k_weight=k_w,
                    v_weight=v_w,
                )

                # Replace the attention module with wrapped FallbackMamba
                wrapper = MambaAttentionWrapper(fallback)
                layer.attention = wrapper

                self.conversion_log.append(
                    f"  Layer {idx}: -> FallbackMamba (d_state={plan.get('d_state', 'N/A')})"
                )
            else:
                self.conversion_log.append(f"  Layer {idx}: attention (kept)")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the hybrid model.

        Mirrors GPTNeoXModel.forward() but with sink tokens prepended.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (optional)

        Returns:
            Logits tensor [batch, num_sinks + seq_len, vocab_size]
        """
        # Get embeddings from the base model
        hidden_states = self.gpt_neox.embed_in(input_ids)

        # Prepend sink tokens
        hidden_states = self.sink_tokens(hidden_states)

        total_seq_len = hidden_states.shape[1]

        # Extend attention mask for sink tokens if provided
        if attention_mask is not None:
            sink_mask = torch.ones(
                attention_mask.shape[0], self.num_sinks,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([sink_mask, attention_mask], dim=1)

        # Compute position IDs for the full sequence (sinks + input)
        position_ids = torch.arange(total_seq_len, device=hidden_states.device).unsqueeze(0)

        # Compute rotary position embeddings (required by GPTNeoX attention)
        position_embeddings = self.gpt_neox.rotary_emb(hidden_states, position_ids=position_ids)

        # Create causal mask for attention layers
        try:
            from transformers.modeling_utils import create_causal_mask
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
        except ImportError:
            causal_mask = None

        # Ensure mask dtype is compatible with attention (must be float or bool)
        if causal_mask is not None and causal_mask.dtype == torch.long:
            causal_mask = causal_mask.to(dtype=hidden_states.dtype)

        # Apply embedding dropout
        hidden_states = self.gpt_neox.emb_dropout(hidden_states)

        # Run through all layers
        for layer in self.gpt_neox.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        # Final layer norm
        hidden_states = self.gpt_neox.final_layer_norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        return logits

    def count_parameters(self):
        """Count total and Mamba-specific parameters."""
        total = sum(p.numel() for p in self.parameters())
        mamba_params = 0
        for plan in self.layer_plan:
            if plan["kind"] == "mamba":
                layer = self.gpt_neox.layers[plan["layer_idx"]]
                mamba_params += sum(p.numel() for p in layer.attention.parameters())
        sink_params = self.sink_tokens.sinks.numel()
        mamba_params += sink_params
        return total, mamba_params

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """Simple greedy generation for testing.

        Args:
            input_ids: [1, seq_len] input token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (1.0 = greedy via argmax).

        Returns:
            Generated token IDs [1, seq_len + max_new_tokens].
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            # Take logits for the last position
            next_token_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated
