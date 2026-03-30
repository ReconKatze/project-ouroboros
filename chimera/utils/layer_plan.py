"""Layer plan builder for hybrid Mamba/Transformer models."""

# Default attention anchor indices per model
ATTN_KEEP_DEFAULTS = {
    "gpt_neox": {0, 3, 7, 11},          # Pythia-160M (12 layers)
    "qwen2":    {0, 4, 8, 12, 16, 23},  # Qwen2.5-0.5B (24 layers)
}


def build_layer_plan(
    num_layers: int,
    attn_keep_indices: set[int] | None = None,
    d_state: int | list[int] = 16,
    model_type: str | None = None,
):
    """Build a layer plan specifying which layers are attention vs Mamba.

    Args:
        num_layers: Total number of layers in the model.
        attn_keep_indices: Set of layer indices to keep as attention.
            If None, uses the default for model_type, or {0, 3, 7, 11}.
        d_state: d_state for Mamba layers. Either a single int (uniform across
            all Mamba layers) or a list[int] with one value per Mamba layer
            (for Step 4 gradient variants). Length must equal num_mamba_layers.
        model_type: HuggingFace model type string ('gpt_neox', 'qwen2').
            Used to select defaults when attn_keep_indices is None.

    Returns:
        List of dicts with 'kind' ('attn' or 'mamba') and optional 'd_state'.
    """
    if attn_keep_indices is None:
        attn_keep_indices = ATTN_KEEP_DEFAULTS.get(model_type, {0, 3, 7, 11})

    plan = []
    mamba_idx = 0
    for i in range(num_layers):
        if i in attn_keep_indices:
            plan.append({"kind": "attn", "layer_idx": i})
        else:
            ds = d_state[mamba_idx] if isinstance(d_state, list) else d_state
            mamba_idx += 1
            plan.append({"kind": "mamba", "layer_idx": i, "d_state": ds})
    return plan
