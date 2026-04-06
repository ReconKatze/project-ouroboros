#!/usr/bin/env python3
"""Transfer P_soft weights from a completed checkpoint into a new variant.

Copies pred_proj, gamma, and sink_tokens — all d_state-independent weights.
Mamba block internals (d_state-dependent shapes) are kept at their fresh
conversion values. pred_proj starts warm, skipping the costly bootstrap phase.

Usage (on Colab after B_psoft finishes):
    python scripts/transfer_psoft.py \\
        --source  checkpoints/B_psoft_final.pt \\
        --output  checkpoints/D_bell2_32_psoft_init.pt \\
        --variant D_bell2_32_psoft \\
        --student Qwen/Qwen2.5-1.5B

Then run the new variant from the warm init:
    python scripts/run_experiment.py \\
        --variants D_bell2_32_psoft --steps 15000 --batch-size 4 \\
        --student Qwen/Qwen2.5-1.5B --teacher Qwen/Qwen2.5-7B \\
        --attn-indices 0,9,18,27 \\
        --resume checkpoints/D_bell2_32_psoft_init.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from argparse import Namespace

from scripts.run_experiment import build_variant_student, VARIANTS


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Transfer P_soft weights between variants")
    p.add_argument("--source",       required=True,
                   help="Path to completed source checkpoint (e.g. B_psoft_final.pt)")
    p.add_argument("--output",       required=True,
                   help="Path to write the warm-init checkpoint")
    p.add_argument("--variant",      default="D_bell2_32_psoft",
                   help="Target variant name (must be in VARIANTS dict)")
    p.add_argument("--student",      default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--attn-indices", default=None,
                   help="Comma-separated attn layer indices (default: model preset)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.variant not in VARIANTS:
        print(f"Unknown variant '{args.variant}'. Available: {list(VARIANTS.keys())}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load source state dict
    # ------------------------------------------------------------------
    print(f"Loading source checkpoint: {args.source}")
    src_ckpt = torch.load(args.source, map_location="cpu", weights_only=False)
    src_state = src_ckpt["model_state"]
    src_step  = src_ckpt.get("step", "?")
    src_var   = src_ckpt.get("variant", "?")
    print(f"  Source variant: {src_var}  |  step: {src_step}")

    # ------------------------------------------------------------------
    # 2. Build target model architecture on CPU
    # ------------------------------------------------------------------
    print(f"\nBuilding target model: {args.variant} ...")
    build_args = Namespace(student=args.student, attn_indices=args.attn_indices)
    target = build_variant_student(build_args, VARIANTS[args.variant],
                                   torch.device("cpu"))

    # ------------------------------------------------------------------
    # 3. Selective weight transfer: copy any key whose name AND shape match
    #    pred_proj and gamma are d_state-independent → will match
    #    Mamba block internals are d_state-dependent → shapes differ, skipped
    # ------------------------------------------------------------------
    print("\nTransferring weights ...")
    target_state   = target.state_dict()
    transferred    = {}
    skipped_shape  = []
    skipped_absent = []

    for key, val in src_state.items():
        if key not in target_state:
            skipped_absent.append(key)
        elif target_state[key].shape != val.shape:
            skipped_shape.append(key)
        else:
            transferred[key] = val

    target_state.update(transferred)
    target.load_state_dict(target_state)

    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    psoft_keys = [k for k in transferred
                  if "pred_proj" in k or k.endswith(".gamma") or "sink" in k]
    mamba_keys = [k for k in transferred
                  if k not in psoft_keys]

    print(f"  Transferred total:       {len(transferred):4d} tensors")
    print(f"    P_soft / gate / sinks: {len(psoft_keys):4d}  ← warm-started")
    print(f"    Other (frozen base):   {len(mamba_keys):4d}")
    print(f"  Skipped (shape differ):  {len(skipped_shape):4d}  ← kept fresh")
    print(f"  Skipped (not in target): {len(skipped_absent):4d}")

    if psoft_keys:
        print(f"\n  Sample transferred P_soft keys:")
        for k in psoft_keys[:6]:
            print(f"    {k}  {list(transferred[k].shape)}")
        if len(psoft_keys) > 6:
            print(f"    ... and {len(psoft_keys) - 6} more")

    # ------------------------------------------------------------------
    # 5. Save warm-init checkpoint
    #    step=0 → load_checkpoint will warm-start (model weights only)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "variant":         args.variant,
        "step":            0,
        "model_state":     target.state_dict(),
        "optimizer_state": None,
        "scheduler_state": None,
        "val_log":         {},
        "ema_loss":        None,
        "chunk_idx":       0,
    }, args.output)

    print(f"\nWarm-init checkpoint saved → {args.output}")
    print(f"\nNext step:")
    print(f"  python scripts/run_experiment.py \\")
    print(f"    --variants {args.variant} --steps 15000 --batch-size 4 \\")
    print(f"    --student {args.student} --teacher Qwen/Qwen2.5-7B \\")
    print(f"    --attn-indices 0,9,18,27 \\")
    print(f"    --resume {args.output}")


if __name__ == "__main__":
    main()
