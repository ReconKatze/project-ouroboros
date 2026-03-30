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

    Qwen uses separate q_proj, k_proj, v_proj. With GQA, K/V projections
    are smaller than Q: K/V are [num_kv_heads * head_dim, hidden_size]
    while Q is [num_heads * head_dim, hidden_size].

    Returns raw weights without tiling — call tile_gqa_weights() after
    if GQA tiling is needed.
    """
    q_w = attention_module.q_proj.weight
    k_w = attention_module.k_proj.weight
    v_w = attention_module.v_proj.weight

    q_b = getattr(attention_module.q_proj, "bias", None)
    k_b = getattr(attention_module.k_proj, "bias", None)
    v_b = getattr(attention_module.v_proj, "bias", None)

    return (q_w, k_w, v_w), (q_b, k_b, v_b)


def tile_gqa_weights(k_w, v_w, num_heads_q, num_heads_kv, head_dim):
    """Tile K/V weights from GQA dimensions to match full Q dimensions.

    GQA uses fewer KV heads than Q heads. This function replicates each
    KV head to match the Q head count, expanding the weight matrices.

    Example for Qwen2.5-0.5B:
        K/V: [128, 896] (2 KV heads x 64 dim) -> [896, 896] (14 heads x 64 dim)
        Replication factor: 14 / 2 = 7

    Args:
        k_w: K weight [num_kv_heads * head_dim, hidden_size]
        v_w: V weight [num_kv_heads * head_dim, hidden_size]
        num_heads_q: Number of query heads (e.g., 14)
        num_heads_kv: Number of KV heads (e.g., 2)
        head_dim: Dimension per head (e.g., 64)

    Returns:
        Tiled (k_w, v_w), each [num_heads_q * head_dim, hidden_size]
    """
    hidden_size = k_w.shape[-1]
    repeat_factor = num_heads_q // num_heads_kv

    # Reshape by head: [num_kv_heads, head_dim, hidden_size]
    k = k_w.view(num_heads_kv, head_dim, hidden_size)
    v = v_w.view(num_heads_kv, head_dim, hidden_size)

    # Replicate each head: [num_heads_q, head_dim, hidden_size]
    k = k.repeat_interleave(repeat_factor, dim=0)
    v = v.repeat_interleave(repeat_factor, dim=0)

    # Flatten back: [num_heads_q * head_dim, hidden_size]
    return k.reshape(-1, hidden_size), v.reshape(-1, hidden_size)


def tile_gqa_biases(k_b, v_b, num_heads_q, num_heads_kv, head_dim):
    """Tile K/V biases for GQA, same logic as tile_gqa_weights."""
    if k_b is None or v_b is None:
        return k_b, v_b

    repeat_factor = num_heads_q // num_heads_kv

    k = k_b.view(num_heads_kv, head_dim)
    v = v_b.view(num_heads_kv, head_dim)

    k = k.repeat_interleave(repeat_factor, dim=0)
    v = v.repeat_interleave(repeat_factor, dim=0)

    return k.reshape(-1), v.reshape(-1)


def extract_qkv(attention_module, num_heads, num_kv_heads, head_dim):
    """Unified QKV extraction that auto-detects format and handles GQA.

    Works for both Pythia (fused QKV) and Qwen (separate projections + GQA).

    Args:
        attention_module: The attention module from a decoder layer.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads (same as num_heads for MHA).
        head_dim: Dimension per head.

    Returns:
        (q_w, k_w, v_w): All [hidden_size, hidden_size] (tiled if GQA).
        (q_b, k_b, v_b): Biases or (None, None, None).
    """
    if hasattr(attention_module, "query_key_value"):
        # Pythia/GPT-NeoX fused format
        return extract_qkv_pythia(attention_module, num_heads, head_dim)

    if all(hasattr(attention_module, name) for name in ("q_proj", "k_proj", "v_proj")):
        # Qwen-style separate projections
        (q_w, k_w, v_w), (q_b, k_b, v_b) = extract_qkv_separate(attention_module)

        # Tile K/V if GQA (fewer KV heads than Q heads)
        if num_kv_heads < num_heads:
            k_w, v_w = tile_gqa_weights(k_w, v_w, num_heads, num_kv_heads, head_dim)
            k_b, v_b = tile_gqa_biases(k_b, v_b, num_heads, num_kv_heads, head_dim)

        return (q_w, k_w, v_w), (q_b, k_b, v_b)

    raise TypeError(f"Unsupported attention module type: {type(attention_module)}")
