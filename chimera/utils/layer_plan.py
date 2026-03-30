"""Layer plan builder for hybrid Mamba/Transformer models."""


def build_layer_plan(
    num_layers: int,
    attn_keep_indices: set[int] | None = None,
    d_state: int = 16,
):
    """Build a layer plan specifying which layers are attention vs Mamba.

    Args:
        num_layers: Total number of layers in the model.
        attn_keep_indices: Set of layer indices to keep as attention.
            Defaults to {0, 3, 7, 11} for 12-layer models.
        d_state: Default d_state for Mamba layers (uniform for Step 1).

    Returns:
        List of dicts with 'kind' ('attn' or 'mamba') and optional 'd_state'.
    """
    if attn_keep_indices is None:
        attn_keep_indices = {0, 3, 7, 11}

    plan = []
    for i in range(num_layers):
        if i in attn_keep_indices:
            plan.append({"kind": "attn", "layer_idx": i})
        else:
            plan.append({"kind": "mamba", "layer_idx": i, "d_state": d_state})
    return plan
