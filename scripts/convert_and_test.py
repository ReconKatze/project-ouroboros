#!/usr/bin/env python3
"""Ouroboros conversion + forward pass test.

Supports any architecture with a HybridChimeraModel adapter.
Steps 1 & 2a use CPU + FallbackMamba.
Step 2b uses --device cuda with real Mamba kernels on Colab T4.

Usage:
    # Step 1 (Pythia, CPU):
    python scripts/convert_and_test.py --model EleutherAI/pythia-160m

    # Step 2a (Qwen, CPU, FallbackMamba):
    python scripts/convert_and_test.py --model Qwen/Qwen2.5-0.5B

    # Step 2b (Qwen, CUDA, real Mamba — Colab):
    python scripts/convert_and_test.py --model Qwen/Qwen2.5-0.5B --device cuda
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from chimera.models.hybrid_model import HybridChimeraModel
from chimera.utils.layer_plan import build_layer_plan


def cosine_similarity(a, b):
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser(description="Ouroboros conversion test")
    parser.add_argument("--model", default="EleutherAI/pythia-160m")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-sinks", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    results = {}

    # --- Load model ---
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    original_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32
    ).to(device)
    original_model.eval()

    config = original_model.config
    model_type = config.model_type
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads

    print(f"Architecture:  {model_type}")
    print(f"Layers:        {num_layers}")
    print(f"Hidden size:   {hidden_size}")
    print(f"Heads:         {num_heads}Q / {num_kv_heads}KV  (head_dim={head_dim})")
    if num_kv_heads < num_heads:
        print(f"GQA ratio:     {num_heads // num_kv_heads}x tiling required for K/V")

    # --- Original model forward for comparison ---
    test_text = "def add(a, b): return a + b"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    with torch.no_grad():
        original_logits = original_model(**inputs).logits
    print(f"\nOriginal logits shape: {original_logits.shape}")

    # --- Build layer plan ---
    layer_plan = build_layer_plan(num_layers, model_type=model_type)
    num_mamba = sum(1 for p in layer_plan if p["kind"] == "mamba")
    num_attn  = sum(1 for p in layer_plan if p["kind"] == "attn")
    print(f"\nLayer plan: {num_mamba} Mamba + {num_attn} Attention")

    # --- Convert ---
    print("Converting layers...")
    hybrid = HybridChimeraModel(
        original_model, layer_plan,
        num_sinks=args.num_sinks,
        device=args.device,
    ).to(device)
    hybrid.eval()

    for line in hybrid.conversion_log:
        print(line)

    total_params, mamba_params = hybrid.count_parameters()
    print(f"\nTotal parameters:         {total_params:,}")
    print(f"Mamba + sink parameters:  {mamba_params:,}")
    results["total_params"] = total_params
    results["mamba_params"] = mamba_params

    # --- Forward pass ---
    print("\n--- Forward Pass Test ---")
    try:
        with torch.no_grad():
            hybrid_logits = hybrid(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        seq_len = inputs["input_ids"].shape[1]
        expected_shape = (1, seq_len + args.num_sinks, config.vocab_size)
        assert hybrid_logits.shape == expected_shape, \
            f"Expected {expected_shape}, got {hybrid_logits.shape}"
        results["shape_check"] = "PASS"
        print(f"Shape:      {hybrid_logits.shape}  OK")
    except Exception as e:
        results["shape_check"] = f"FAIL: {e}"
        print(f"Shape check FAILED: {e}")
        _print_summary(results)
        return results

    # --- NaN check ---
    if torch.isfinite(hybrid_logits).all():
        results["nan_check"] = "PASS"
        print("NaN/Inf:    PASS")
    else:
        n = (~torch.isfinite(hybrid_logits)).sum().item()
        results["nan_check"] = f"FAIL ({n} non-finite)"
        print(f"NaN/Inf:    FAIL — {n} non-finite values")
        _print_summary(results)
        return results

    # --- Cosine similarity ---
    hybrid_comparable = hybrid_logits[:, args.num_sinks:, :]
    cos_sim = cosine_similarity(original_logits, hybrid_comparable)
    results["cosine_similarity"] = f"{cos_sim:.4f}"
    print(f"Cosine sim: {cos_sim:.4f}", end="  ")
    if cos_sim >= 0.95:
        print("(high — expected with kept attn layers + untouched MLPs)")
    elif cos_sim >= 0.4:
        print("(moderate)")
    else:
        print("(low — large divergence)")

    # --- Top-1 divergence ---
    orig_top1 = original_logits.argmax(dim=-1)
    hyb_top1  = hybrid_comparable.argmax(dim=-1)
    top1_match = (orig_top1 == hyb_top1).float().mean().item()
    results["top1_agreement"] = f"{top1_match:.1%}"
    print(f"Top-1 agreement: {top1_match:.1%}", end="  ")
    print("(conversion changed predictions)" if top1_match < 1.0 else "(no change — check conversion)")

    # --- Generation ---
    print("\n--- Generation Test ---")
    prompt = "def hello():"
    gen_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    try:
        gen_ids = hybrid.generate(gen_inputs["input_ids"], max_new_tokens=40)
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        results["generation"] = "PASS"
        print(f"Prompt:    {prompt}")
        print(f"Generated: {gen_text[:200]}")
    except Exception as e:
        results["generation"] = f"FAIL: {e}"
        print(f"Generation FAILED: {e}")

    _print_summary(results)
    return results


def _print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        ok = "FAIL" not in str(v)
        print(f"  [{'OK' if ok else '!!'}] {k}: {v}")
    all_pass = all("FAIL" not in str(v) for v in results.values())
    print()
    if all_pass:
        print("PASS — Conversion scaffold works for this model.")
    else:
        print("ISSUES DETECTED — see above.")


if __name__ == "__main__":
    main()
