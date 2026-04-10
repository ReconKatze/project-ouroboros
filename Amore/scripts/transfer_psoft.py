#!/usr/bin/env python3
"""Transfer compatible weights between LifeEquationModel checkpoints.

P_soft (predictive coding via L_pred) is baked into the LE model's forward pass
and is no longer a separate module to transfer. This script handles the general
case: transfer all weights whose keys AND shapes match between two LE checkpoints.

Typical use-case: warm-start a new variant from a completed lower-d_state run.
Weights that are d_state-independent (embed, output_head, all auxiliary modules)
will transfer; MambaStep weights whose shapes depend on d_state will be skipped.

Usage:
    python scripts/transfer_psoft.py \\
        --source  checkpoints/C_final.pt \\
        --target-variant C_fast \\
        --output  checkpoints/C_fast_warminit.pt

Then resume from that warm init:
    python scripts/run_experiment.py \\
        --variants C_fast --steps 10000 \\
        --resume checkpoints/C_fast_warminit.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "V2"))

import torch
from argparse import Namespace

from life_eq_v2.config import LifeEquationConfig
from life_eq_v2.model import LifeEquationModel


def parse_args():
    p = argparse.ArgumentParser(description="Transfer compatible LE weights between checkpoints")
    p.add_argument("--source",          required=True,
                   help="Source checkpoint (.pt) — must contain 'model_state' key")
    p.add_argument("--target-variant",  required=True,
                   help="Target variant name from run_experiment.py VARIANTS dict")
    p.add_argument("--output",          required=True,
                   help="Output checkpoint path")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Import VARIANTS from run_experiment ---
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_experiment import VARIANTS

    if args.target_variant not in VARIANTS:
        print(f"Unknown variant '{args.target_variant}'. Available: {list(VARIANTS.keys())}")
        sys.exit(1)

    # --- Load source ---
    print(f"Loading source: {args.source}")
    src_ckpt   = torch.load(args.source, map_location="cpu", weights_only=False)
    src_state  = src_ckpt["model_state"]
    src_step   = src_ckpt.get("step", "?")
    src_var    = src_ckpt.get("variant", "?")
    print(f"  Source variant: {src_var}  |  step: {src_step}")

    # --- Build target model ---
    device = torch.device("cpu")
    cfg    = VARIANTS[args.target_variant]["config_kwargs"]
    config = LifeEquationConfig(**{**cfg, "device": "cpu"})
    target = LifeEquationModel(config)
    target.eval()

    # --- Selective transfer ---
    print(f"\nTransferring weights to variant '{args.target_variant}'...")
    target_state    = target.state_dict()
    transferred     = {}
    skipped_shape   = []
    skipped_absent  = []

    for key, val in src_state.items():
        if key not in target_state:
            skipped_absent.append(key)
        elif target_state[key].shape != val.shape:
            skipped_shape.append(key)
        else:
            transferred[key] = val

    target_state.update(transferred)
    target.load_state_dict(target_state)

    # --- Report ---
    print(f"  Transferred:    {len(transferred):4d} tensors")
    print(f"  Shape mismatch: {len(skipped_shape):4d} tensors (d_state-dependent — expected)")
    print(f"  Absent in src:  {len(skipped_absent):4d} tensors")

    # Group transferred keys by module prefix for readability
    from collections import Counter
    prefixes = Counter(k.split(".")[0] for k in transferred)
    print("\n  Transferred by module:")
    for mod, count in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"    {mod:<30}: {count}")

    if skipped_shape:
        print(f"\n  Skipped (shape mismatch) — first 5:")
        for k in skipped_shape[:5]:
            print(f"    {k}: src={src_state[k].shape}  target={VARIANTS[args.target_variant]}")

    # --- Save output (as a warm-init checkpoint compatible with load_checkpoint) ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "variant":     args.target_variant,
        "step":        src_ckpt.get("steps", 0),   # marks as completed → warm-start on resume
        "model_state": target.state_dict(),
    }, args.output)
    print(f"\nWarm-init checkpoint saved: {args.output}")
    print(f"\nResume with:")
    print(f"  python scripts/run_experiment.py \\")
    print(f"      --variants {args.target_variant} --steps 10000 \\")
    print(f"      --resume {args.output}")


if __name__ == "__main__":
    main()
