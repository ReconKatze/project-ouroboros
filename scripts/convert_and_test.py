#!/usr/bin/env python3
"""Step 1: Convert Pythia-160M to hybrid Mamba/Attention and verify.

Usage:
    python scripts/convert_and_test.py --model EleutherAI/pythia-160m --device cpu
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from chimera.models.hybrid_model import HybridChimeraModel
from chimera.utils.layer_plan import build_layer_plan


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser(description="Step 1: Pythia-160M Mamba Conversion")
    parser.add_argument("--model", default="EleutherAI/pythia-160m",
                        help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--num-sinks", type=int, default=4,
                        help="Number of sink tokens")
    args = parser.parse_args()

    device = torch.device(args.device)
    results = {}

    # --- Load original model ---
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    original_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(device)
    original_model.eval()

    config = original_model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_size = hidden_size // num_heads
    use_parallel = getattr(config, "use_parallel_residual", True)

    print(f"Model config: {num_layers} layers, hidden_size={hidden_size}, "
          f"num_heads={num_heads}, head_size={head_size}")
    print(f"Parallel residual: {use_parallel}")

    # --- Get original model output for comparison ---
    test_text = "def add(a, b): return a + b"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    with torch.no_grad():
        original_output = original_model(**inputs)
        original_logits = original_output.logits

    print(f"Original logits shape: {original_logits.shape}")
    print(f"Original logits range: [{original_logits.min():.2f}, {original_logits.max():.2f}]")

    # --- Build layer plan ---
    attn_keep = {0, 3, 7, 11}
    layer_plan = build_layer_plan(num_layers, attn_keep)
    num_mamba = sum(1 for p in layer_plan if p["kind"] == "mamba")
    num_attn = sum(1 for p in layer_plan if p["kind"] == "attn")
    print(f"\nLayer plan: {num_mamba} Mamba + {num_attn} Attention")

    # --- Convert to hybrid ---
    print("Converting layers...")
    hybrid_model = HybridChimeraModel(
        original_model, layer_plan, num_sinks=args.num_sinks
    ).to(device)
    hybrid_model.eval()

    for line in hybrid_model.conversion_log:
        print(line)
    print("Conversion complete.")

    # --- Parameter counts ---
    total_params, mamba_params = hybrid_model.count_parameters()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Mamba + sink parameters: {mamba_params:,}")
    results["total_params"] = total_params
    results["mamba_params"] = mamba_params

    # --- Forward pass test ---
    print("\n--- Forward Pass Test ---")
    try:
        with torch.no_grad():
            hybrid_logits = hybrid_model(inputs["input_ids"],
                                         attention_mask=inputs.get("attention_mask"))
        print(f"Hybrid logits shape: {hybrid_logits.shape}")
        expected_seq_len = inputs["input_ids"].shape[1] + args.num_sinks
        assert hybrid_logits.shape == (1, expected_seq_len, config.vocab_size), \
            f"Shape mismatch: expected (1, {expected_seq_len}, {config.vocab_size})"
        results["shape_check"] = "PASS"
        print(f"Shape check: PASS")
    except Exception as e:
        results["shape_check"] = f"FAIL: {e}"
        print(f"Shape check: FAIL - {e}")
        return results

    # --- NaN check ---
    if torch.isfinite(hybrid_logits).all():
        results["nan_check"] = "PASS"
        print(f"NaN/Inf check: PASS")
    else:
        nan_count = (~torch.isfinite(hybrid_logits)).sum().item()
        results["nan_check"] = f"FAIL: {nan_count} non-finite values"
        print(f"NaN/Inf check: FAIL ({nan_count} non-finite values)")
        return results

    print(f"Hybrid logits range: [{hybrid_logits.min():.2f}, {hybrid_logits.max():.2f}]")

    # --- Cosine similarity ---
    # Compare only the overlapping sequence positions (skip sink tokens)
    hybrid_comparable = hybrid_logits[:, args.num_sinks:, :]
    cos_sim = cosine_similarity(original_logits, hybrid_comparable)
    results["cosine_similarity"] = cos_sim
    print(f"\nCosine similarity with original: {cos_sim:.4f}")

    if cos_sim >= 0.95:
        print("  -> High similarity (expected with parallel residual + 4 kept attention layers)")
        print("     MLP paths are untouched and dominate in GPT-NeoX parallel residual.")
    elif cos_sim >= 0.5:
        print("  -> Moderate similarity. Conversion changed logit distribution noticeably.")
    elif cos_sim > 0.0:
        print("  -> Low similarity. Large divergence from original.")
    else:
        print("  -> Very low/negative. Weights may not have transferred correctly.")

    # Check top-1 token divergence (more sensitive than cosine sim)
    orig_top1 = original_logits.argmax(dim=-1)
    hybrid_top1 = hybrid_comparable.argmax(dim=-1)
    top1_match = (orig_top1 == hybrid_top1).float().mean().item()
    results["top1_agreement"] = top1_match
    print(f"Top-1 token agreement: {top1_match:.2%}")
    if top1_match < 1.0:
        print("  -> Tokens diverge - conversion IS changing predictions (good).")

    # --- Generation test ---
    print("\n--- Generation Test ---")
    prompt = "def hello():"
    gen_inputs = tokenizer(prompt, return_tensors="pt").to(device)

    try:
        generated_ids = hybrid_model.generate(
            gen_inputs["input_ids"], max_new_tokens=50
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        results["generation"] = "PASS"
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text[:200]}")
        print("  -> Generation completed (gibberish is expected at this stage)")
    except Exception as e:
        results["generation"] = f"FAIL: {e}"
        print(f"Generation: FAIL - {e}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("STEP 1 SUMMARY")
    print("=" * 60)
    all_pass = True
    for key, value in results.items():
        status = value if isinstance(value, str) else str(value)
        is_pass = "PASS" in str(status) if isinstance(status, str) else True
        marker = "OK" if is_pass else "!!"
        print(f"  [{marker}] {key}: {status}")
        if isinstance(status, str) and "FAIL" in status:
            all_pass = False

    if all_pass:
        print("\nStep 1: PASS - Scaffold conversion works!")
    else:
        print("\nStep 1: ISSUES DETECTED - See above for details.")

    return results


if __name__ == "__main__":
    main()
