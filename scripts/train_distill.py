#!/usr/bin/env python3
"""Step 3: Distillation from Qwen2.5-3B teacher into HybridChimeraModel.

Trains only Mamba blocks + sink tokens via temperature-scaled KL divergence.
All attention layers, MLPs, norms, embeddings, and lm_head are frozen.

Primary proof goal: gradient flow through mamba-ssm CUDA kernels.

Usage (Colab T4):
    python scripts/train_distill.py
    python scripts/train_distill.py --teacher Qwen/Qwen2.5-3B --steps 500
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from chimera.models.hybrid_model import HybridChimeraModel
from chimera.utils.layer_plan import build_layer_plan


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_student(args, device):
    """Load Qwen2.5-0.5B, convert to hybrid, freeze all except Mamba+sinks."""
    print(f"Building student from {args.student}...")

    # Load in float32 — conversion weight extraction requires float32
    base = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.float32
    ).to(device)

    layer_plan = build_layer_plan(
        base.config.num_hidden_layers,
        model_type=base.config.model_type,
    )

    hybrid = HybridChimeraModel(
        base, layer_plan, num_sinks=4, device=str(device)
    )
    for line in hybrid.conversion_log:
        print(line)

    # --- Freeze pattern ---
    # Step 1: freeze everything
    for p in hybrid.parameters():
        p.requires_grad_(False)

    # Step 2: unfreeze Mamba wrappers (via layer_plan)
    for plan in hybrid.layer_plan:
        if plan["kind"] == "mamba":
            wrapper = hybrid.adapter.get_attention(hybrid.layers[plan["layer_idx"]])
            for p in wrapper.parameters():
                p.requires_grad_(True)

    # Step 3: unfreeze sink tokens
    hybrid.sink_tokens.sinks.requires_grad_(True)

    # Keep student in float32 — GradScaler requires float32 parameters.
    # autocast handles float16 math during the forward pass.
    hybrid = hybrid.to(device)
    hybrid.train()

    trainable = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in hybrid.parameters())
    print(f"Student trainable params: {trainable:,} / {total:,}")

    return hybrid


def build_teacher(args, device):
    """Load Qwen2.5-3B in float16, eval mode, fully frozen."""
    print(f"Loading teacher {args.teacher} (float16)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.float16
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("Teacher loaded.")
    return teacher


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_token_chunks(tokenizer, seq_len):
    """Yield fixed-length LongTensor chunks from streaming wikitext-103."""
    ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1",
        split="train", streaming=True,
        trust_remote_code=False,
    )
    buffer = []
    for example in ds:
        text = example["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(ids)
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            buffer = buffer[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def distill_loss(student_logits, teacher_logits, T=2.0):
    """Temperature-scaled KL divergence with per-token std normalization.

    Both inputs must be float32. Normalization stabilizes early training when
    Mamba logit magnitudes differ from the teacher.

    Args:
        student_logits: [batch, seq_len, vocab]  (sink positions already removed)
        teacher_logits: [batch, seq_len, vocab]
        T: Temperature (default 2.0)

    Returns:
        Scalar loss.
    """
    s = student_logits / (student_logits.std(dim=-1, keepdim=True) + 1e-6)
    t = teacher_logits / (teacher_logits.std(dim=-1, keepdim=True) + 1e-6)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t     = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)


# ---------------------------------------------------------------------------
# Gradient flow verification
# ---------------------------------------------------------------------------

def verify_gradient_flow(hybrid):
    """Print Mamba param gradient norms after step 1.

    Non-zero + finite norms prove gradients flow through mamba-ssm CUDA kernels.

    The accessor chain is:
        hybrid.layers[i].self_attn          -> MambaAttentionWrapper
        .mamba                               -> RealMambaBlock
        .mamba                               -> mamba_ssm.Mamba  (the actual SSM)
    """
    print("\n--- Gradient Flow Verification (Step 1) ---")
    any_pass = False
    checked  = 0

    for plan in hybrid.layer_plan:
        if plan["kind"] != "mamba":
            continue
        idx     = plan["layer_idx"]
        wrapper = hybrid.adapter.get_attention(hybrid.layers[idx])
        mamba   = wrapper.mamba        # RealMambaBlock
        ssm     = mamba.mamba          # mamba_ssm.Mamba

        for name, p in ssm.named_parameters():
            if p.grad is None:
                print(f"  [!!] layer {idx:2d}.{name}: grad is None")
                checked += 1
                continue
            norm = p.grad.data.norm(2).item()
            ok   = math.isfinite(norm) and norm > 0
            if ok:
                any_pass = True
            tag = "OK" if ok else ("NaN" if not math.isfinite(norm) else "ZERO")
            # Only print first layer verbosely to keep output manageable
            if idx == hybrid.layer_plan[1]["layer_idx"]:
                print(f"  [{tag}] layer {idx:2d}.{name}: grad_norm={norm:.4e}")
            checked += 1

    # Sink tokens
    if hybrid.sink_tokens.sinks.grad is not None:
        snorm = hybrid.sink_tokens.sinks.grad.norm().item()
        print(f"  sink_tokens.sinks: grad_norm={snorm:.4e}")

    if any_pass:
        print("\nGradient flow: PASS — mamba-ssm backward kernels are live")
    else:
        print("\nGradient flow: FAIL — all Mamba gradients are zero or None")
        print("  Check: student is not inside torch.no_grad(); autocast is applied correctly")
    return any_pass


# ---------------------------------------------------------------------------
# Generation test
# ---------------------------------------------------------------------------

def run_generation_test(hybrid, tokenizer, device, label=""):
    prompt = "def add(a, b):"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hybrid.eval()
    with torch.no_grad():
        gen_ids = hybrid.generate(inputs["input_ids"], max_new_tokens=40)
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    hybrid.train()
    tag = f" ({label})" if label else ""
    print(f"\nGeneration{tag}:")
    print(f"  Prompt:    {prompt}")
    print(f"  Generated: {text[:300]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Step 3: Distillation training")
    p.add_argument("--teacher",      default="Qwen/Qwen2.5-3B")
    p.add_argument("--student",      default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--steps",        type=int,   default=500)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--seq-len",      type=int,   default=512)
    p.add_argument("--temperature",  type=float, default=2.0)
    p.add_argument("--batch-size",   type=int,   default=1,
                   help="Batches per step (each batch = seq-len tokens)")
    p.add_argument("--log-every",    type=int,   default=10)
    p.add_argument("--out",          default="checkpoints/step3.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # --- Build models ---
    teacher = build_teacher(args, device)
    student = build_student(args, device)

    # Tokenizer (Qwen2.5-0.5B and 3B share the same tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Pre-training generation ---
    run_generation_test(student, tokenizer, device, label="pre-training")

    # --- Optimizer (AdamW on trainable params only) ---
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # float16 AMP scaler (T4 = Turing, no bfloat16)
    scaler = torch.amp.GradScaler("cuda")

    # --- Dataset ---
    data_gen = make_token_chunks(tokenizer, args.seq_len)

    def next_batch():
        """Return [batch_size, seq_len] input_ids, restarting generator if needed."""
        nonlocal data_gen
        chunks = []
        for _ in range(args.batch_size):
            try:
                chunks.append(next(data_gen))
            except StopIteration:
                data_gen = make_token_chunks(tokenizer, args.seq_len)
                chunks.append(next(data_gen))
        return torch.stack(chunks).to(device)  # [batch, seq_len]

    # --- Training loop ---
    print(f"\nStarting distillation: {args.steps} steps | "
          f"lr={args.lr} | T={args.temperature} | seq_len={args.seq_len}")
    print("=" * 60)

    loss_step1   = None
    loss_last    = None
    grad_verified = False

    for step in range(1, args.steps + 1):
        input_ids = next_batch()   # [batch, seq_len]

        # Teacher forward (no grad, stays float16)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids).logits  # [batch, seq_len, vocab]

        # Student forward (AMP float16)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            full_logits = student(input_ids)                       # [batch, num_sinks+seq_len, vocab]
            # Remove sink token positions to align with teacher
            student_logits = full_logits[:, student.num_sinks:, :] # [batch, seq_len, vocab]

        # KL loss in float32 (softmax over 151,936 tokens underflows in float16)
        loss = distill_loss(
            student_logits.float(),
            teacher_logits.float(),
            T=args.temperature,
        )

        if not torch.isfinite(loss):
            print(f"Step {step}: NaN/Inf loss — stopping. Check LR or model state.")
            break

        scaler.scale(loss).backward()

        # Unscale BEFORE gradient check and clip (idempotent per step)
        scaler.unscale_(optimizer)

        # Gradient flow verification on step 1
        if step == 1 and not grad_verified:
            grad_verified = verify_gradient_flow(student)
            print()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        if step == 1:
            loss_step1 = loss_val
        loss_last = loss_val

        # Compute total gradient norm for logging
        with torch.no_grad():
            grad_norm = math.sqrt(
                sum(p.grad.norm().item() ** 2
                    for p in trainable_params
                    if p.grad is not None)
            )

        if step % args.log_every == 0 or step == 1:
            print(f"Step {step:4d}/{args.steps} | loss={loss_val:.4f} | grad_norm={grad_norm:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Loss at step 1:    {loss_step1:.4f}")
    print(f"  Loss at step {args.steps}: {loss_last:.4f}")
    decreased = loss_last < loss_step1
    print(f"  Loss decreased:    {'YES' if decreased else 'NO — check gradient flow'}")
    print(f"  Gradient verified: {'YES' if grad_verified else 'NO'}")

    all_pass = decreased and grad_verified
    print()
    if all_pass:
        print("PASS — Mamba distillation scaffold works.")
    else:
        print("ISSUES DETECTED — see above.")

    # --- Save checkpoint ---
    torch.save(student.state_dict(), args.out)
    print(f"\nCheckpoint saved: {args.out}")

    # --- Post-training generation ---
    run_generation_test(student, tokenizer, device, label="post-training")

    print("\nStep 3 complete.")
    return all_pass


if __name__ == "__main__":
    main()
