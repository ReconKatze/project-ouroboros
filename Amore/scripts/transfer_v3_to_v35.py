#!/usr/bin/env python3
"""
transfer_v3_to_v35.py — warm-initialise a V3.5 (~7B) model from a V3 (~1.5B) checkpoint.

Strategy
--------
For every parameter in the V3.5 state-dict:
  1. Remap the layer index using the attention-anchor alignment below.
  2. If a matching V3 key exists and shapes are identical  → copy directly.
  3. If V3.5 tensor is strictly larger in every dimension  → copy V3 weights
     into the top-left corner; the remainder stays at the model's random init.
  4. If shapes are incompatible in any dimension           → skip (random init).

Layer remapping
---------------
  V3  attention anchors : (0,  9, 18, 27)   24 Mamba + 4 attention in 28 layers
  V3.5 attention anchors: (0, 12, 24, 35)   32 Mamba + 4 attention in 36 layers

  Attention layers transfer 1-for-1 by anchor position (1st→1st, 2nd→2nd, …).
  Mamba layers cycle: V3.5 Mamba slot i gets V3 Mamba slot (i % 24).

Non-layer parameters (embed, lm_head, LE modules, controller) transfer by the
same exact/padded/skip logic without any index remapping.

Usage
-----
  python scripts/transfer_v3_to_v35.py \\
      --src  checkpoints/step4_le_v3_step010000.pt \\
      --dst  checkpoints/v35_warm_init.pt \\
      --variant round3_full \\
      --d-state 128
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "V3.5"))   # V3.5 wins; used for build_*

import torch


# ── Architecture constants (hardcoded — don't change without updating the map) ──

V3_ATTN_ANCHORS  = (0,  9, 18, 27)
V3_N_LAYERS      = 28

V35_ATTN_ANCHORS = (0, 12, 24, 35)
V35_N_LAYERS     = 36


# ── Layer index mapping ──────────────────────────────────────────────────────

def _build_layer_map() -> dict[int, int]:
    """Return {v35_layer_idx: v3_layer_idx} for all 36 V3.5 layers."""
    v3_attn  = set(V3_ATTN_ANCHORS)
    v35_attn = set(V35_ATTN_ANCHORS)

    v3_mamba  = [i for i in range(V3_N_LAYERS)  if i not in v3_attn]   # 24
    v35_mamba = [i for i in range(V35_N_LAYERS) if i not in v35_attn]  # 32

    mapping: dict[int, int] = {}

    # Attention: align by anchor position (0th→0th, 1st→1st, …)
    for v35_a, v3_a in zip(sorted(V35_ATTN_ANCHORS), sorted(V3_ATTN_ANCHORS)):
        mapping[v35_a] = v3_a

    # Mamba: cycle through V3's 24 Mamba layers
    for slot, v35_idx in enumerate(v35_mamba):
        mapping[v35_idx] = v3_mamba[slot % len(v3_mamba)]

    return mapping


# ── Weight transfer ──────────────────────────────────────────────────────────

def _transfer_tensor(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    """
    Copy src into the matching subspace of dst.

    Returns (result_tensor, status) where status is one of:
      "exact"   — shapes matched, copied directly
      "padded"  — src fits inside dst, copied with zero-pad remainder
      "skip"    — incompatible shape, dst unchanged (random init kept)
    """
    if src.shape == dst.shape:
        return src.clone(), "exact"

    if all(s <= d for s, d in zip(src.shape, dst.shape)):
        result = dst.clone()
        slices = tuple(slice(0, s) for s in src.shape)
        result[slices] = src
        pct = 100.0 * src.numel() / dst.numel()
        return result, f"padded ({pct:.1f}% filled)"

    return dst, "skip"


def _remap_key(key: str, layer_map: dict[int, int]) -> str:
    """
    Rewrite a parameter key's layer index using layer_map.
    e.g. "layers.12.in_proj.weight" → "layers.9.in_proj.weight" (if 12→9 in map).
    Returns the original key if it doesn't start with "layers.<int>.".
    """
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
        v35_idx = int(parts[1])
        if v35_idx in layer_map:
            parts[1] = str(layer_map[v35_idx])
            return ".".join(parts)
    return key


# ── Main ─────────────────────────────────────────────────────────────────────

def transfer(src_path: str, dst_path: str, variant: str, d_state: int) -> None:
    print("=" * 64)
    print("V3 → V3.5 weight transfer")
    print(f"  Source  : {src_path}")
    print(f"  Dest    : {dst_path}")
    print(f"  Variant : {variant}  d_state={d_state}")
    print("=" * 64)

    # ── Load V3 checkpoint ───────────────────────────────────────────────────
    src_ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    src_sd   = src_ckpt["model_state"]
    src_step = src_ckpt.get("step", "?")
    print(f"\nV3 checkpoint: step={src_step}, "
          f"{sum(t.numel() for t in src_sd.values()) / 1e9:.2f}B parameters loaded")

    # ── Build V3.5 model (for its state-dict shape reference) ────────────────
    from life_eq_v3.factory import build_config, build_model

    config  = build_config(variant)
    config  = dc_replace(config, d_state=d_state)
    model   = build_model(variant, config)
    dst_sd  = model.state_dict()

    n_params_v35 = sum(p.numel() for p in model.parameters())
    print(f"V3.5 model   : {n_params_v35 / 1e9:.2f}B parameters\n")

    # ── Build layer map ──────────────────────────────────────────────────────
    layer_map = _build_layer_map()

    # ── Transfer ─────────────────────────────────────────────────────────────
    new_sd: dict[str, torch.Tensor] = {}

    n_exact  = 0
    n_padded = 0
    n_skip   = 0
    params_inherited = 0

    for dst_key, dst_tensor in dst_sd.items():
        src_key = _remap_key(dst_key, layer_map)

        if src_key in src_sd:
            result, status = _transfer_tensor(src_sd[src_key], dst_tensor)
            new_sd[dst_key] = result

            if status == "exact":
                n_exact += 1
                params_inherited += dst_tensor.numel()
            elif status.startswith("padded"):
                n_padded += 1
                params_inherited += src_sd[src_key].numel()
                print(f"  padded  {dst_key}")
                print(f"          {list(src_sd[src_key].shape)} → {list(dst_tensor.shape)}  "
                      f"[{status}]")
            else:
                n_skip += 1
                print(f"  skip    {dst_key}")
                print(f"          {list(src_sd[src_key].shape)} incompatible with "
                      f"{list(dst_tensor.shape)}")
        else:
            # No matching V3 key — keep random init
            new_sd[dst_key] = dst_tensor
            n_skip += 1

    print(f"\nSummary:")
    print(f"  Exact copy : {n_exact} tensors")
    print(f"  Padded     : {n_padded} tensors")
    print(f"  Random init: {n_skip} tensors (no matching V3 key or incompatible shape)")
    print(f"  Inherited  : {params_inherited / 1e9:.2f}B / {n_params_v35 / 1e9:.2f}B params "
          f"({100.0 * params_inherited / n_params_v35:.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state"    : new_sd,
        "step"           : 0,
        "best_loss"      : float("inf"),
        "transfer_source": str(src_path),
        "transfer_step"  : src_step,
        "variant"        : variant,
    }
    torch.save(payload, dst_path)
    print(f"\nSaved → {dst_path}")
    print("=" * 64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Warm-initialise V3.5 (~7B) from a V3 (~1.5B) checkpoint."
    )
    p.add_argument("--src", required=True,
                   help="Path to the V3 checkpoint .pt file (source)")
    p.add_argument("--dst", required=True,
                   help="Output path for the V3.5 warm-init checkpoint .pt")
    p.add_argument("--variant", default="round3_full",
                   help="V3.5 variant name (default: round3_full)")
    p.add_argument("--d-state", type=int, default=128,
                   help="d_state for Mamba SSM (default: 128)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    transfer(
        src_path=args.src,
        dst_path=args.dst,
        variant=args.variant,
        d_state=args.d_state,
    )
