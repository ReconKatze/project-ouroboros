"""Weight extraction utilities for converting attention layers to Mamba."""

import torch


def extract_qkv_pythia(attention_module, num_heads: int, head_size: int):
    """Extract Q, K, V weights from Pythia/GPT-NeoX fused query_key_value.

    GPT-NeoX uses interleaved per-head format:
        [Q0, K0, V0, Q1, K1, V1, ..., Q_n, K_n, V_n]
    where each Q/K/V block is head_size rows.

    A naive torch.chunk(weight, 3, dim=0) would be WRONG here.

    Args:
        attention_module: The GPTNeoXAttention module.
        num_heads: Number of attention heads.
        head_size: Dimension per head.

    Returns:
        Tuple of (q_weight, k_weight, v_weight), each [hidden_size, hidden_size].
        And (q_bias, k_bias, v_bias) if biases exist, else (None, None, None).
    """
    hidden_size = num_heads * head_size

    # Weight: [3 * hidden_size, hidden_size] = [num_heads * 3 * head_size, hidden_size]
    weight = attention_module.query_key_value.weight
    w = weight.view(num_heads, 3, head_size, hidden_size)
    q_w = w[:, 0, :, :].contiguous().view(-1, hidden_size)  # [hidden, hidden]
    k_w = w[:, 1, :, :].contiguous().view(-1, hidden_size)
    v_w = w[:, 2, :, :].contiguous().view(-1, hidden_size)

    # Bias: [3 * hidden_size] if present
    q_b, k_b, v_b = None, None, None
    if attention_module.query_key_value.bias is not None:
        bias = attention_module.query_key_value.bias
        b = bias.view(num_heads, 3, head_size)
        q_b = b[:, 0, :].contiguous().view(-1)
        k_b = b[:, 1, :].contiguous().view(-1)
        v_b = b[:, 2, :].contiguous().view(-1)

    return (q_w, k_w, v_w), (q_b, k_b, v_b)


def extract_qkv_separate(attention_module):
    """Extract Q, K, V from separate projection layers (Qwen-style).

    Stub for Step 2. Qwen uses separate q_proj, k_proj, v_proj with GQA.
    """
    q_w = attention_module.q_proj.weight
    k_w = attention_module.k_proj.weight
    v_w = attention_module.v_proj.weight

    q_b = getattr(attention_module.q_proj, "bias", None)
    k_b = getattr(attention_module.k_proj, "bias", None)
    v_b = getattr(attention_module.v_proj, "bias", None)

    return (q_w, k_w, v_w), (q_b, k_b, v_b)
