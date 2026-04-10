#!/usr/bin/env python3
"""Step 3 (LE v15): Single-variant distillation proof-of-concept.

Trains one LifeEquationModel configuration via KL distillation from a teacher.
Primary proof goal: L_total decreases, gradient flows through all LE modules.

Use this for quick smoke-tests and single-config runs.
For multi-variant comparisons, use run_experiment.py.

Usage (Colab A100/T4):
    python scripts/train_distill.py
    python scripts/train_distill.py --teacher Qwen/Qwen2.5-7B --steps 1000

AMP: bfloat16 on A100/H100, float16 on T4.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "V2"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from life_eq_v2.config import LifeEquationConfig
from life_eq_v2.model import LifeEquationModel
from life_eq_v2.state import FullState


# ---------------------------------------------------------------------------
# AMP dtype detection
# ---------------------------------------------------------------------------

def get_amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_token_chunks(tokenizer, seq_len: int):
    """Yield fixed-length LongTensor chunks from streaming wikitext-103."""
    ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1",
        split="train", streaming=True, trust_remote_code=False,
    )
    buf = []
    for ex in ds:
        text = ex["text"].strip()
        if not text:
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(buf) >= seq_len:
            yield torch.tensor(buf[:seq_len], dtype=torch.long)
            buf = buf[seq_len:]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def kl_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """Temperature-scaled KL. Handles vocab size mismatch by truncating to min."""
    vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    s = student_logits[..., :vocab]
    t = teacher_logits[..., :vocab]
    s = s / (s.std(dim=-1, keepdim=True) + 1e-6)
    t = t / (t.std(dim=-1, keepdim=True) + 1e-6)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t     = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)


# ---------------------------------------------------------------------------
# Gradient flow verification
# ---------------------------------------------------------------------------

def verify_gradient_flow(model: LifeEquationModel) -> bool:
    """Verify non-zero finite gradients exist in all major LE module groups."""
    print("\n--- Gradient Flow Verification (Step 1) ---")
    groups = {
        "mamba_layers":     model.mamba_layers,
        "pred_heads":       model.pred_heads,
        "attention_module": model.attention_module,
        "identity_module":  model.identity_module,
        "value_module":     model.value_module,
        "controller":       model.controller,
    }
    all_pass = True
    for group_name, module in groups.items():
        params_with_grad = [
            (n, p) for n, p in module.named_parameters()
            if p.grad is not None
        ]
        if not params_with_grad:
            print(f"  [!!] {group_name}: no gradients")
            all_pass = False
            continue
        norms = [p.grad.norm().item() for _, p in params_with_grad]
        mean_norm = sum(norms) / len(norms)
        ok = math.isfinite(mean_norm) and mean_norm > 0
        tag = "OK" if ok else ("NaN" if not math.isfinite(mean_norm) else "ZERO")
        print(f"  [{tag}] {group_name}: mean_grad_norm={mean_norm:.4e}  ({len(params_with_grad)} params)")
        if not ok:
            all_pass = False
    print(f"\nGradient flow: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Step 3 (LE v15): Single-variant distillation")
    p.add_argument("--teacher",     default="Qwen/Qwen2.5-7B")
    p.add_argument("--tokenizer",   default="Qwen/Qwen2.5-1.5B",
                   help="Tokenizer to use (must match LE config vocab_size=151936)")
    p.add_argument("--steps",       type=int,   default=500)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--seq-len",     type=int,   default=1024)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--d-state",     type=int,   default=64,
                   help="LE model d_state (spec default: 64)")
    p.add_argument("--log-every",   type=int,   default=10)
    p.add_argument("--out",         default="checkpoints/step3_le.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    amp_dtype  = get_amp_dtype()
    use_scaler = (amp_dtype == torch.float16)
    dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
    print(f"AMP dtype: {dtype_name}")

    # --- Build LE student ---
    print(f"\nBuilding LifeEquationModel (d_state={args.d_state})...")
    config  = LifeEquationConfig(d_state=args.d_state, device=str(device))
    student = LifeEquationModel(config).to(device)
    student.train()
    total_p  = sum(p.numel() for p in student.parameters())
    print(f"  Total params: {total_p:,}")

    # --- Load teacher ---
    print(f"\nLoading teacher: {args.teacher} ({dtype_name})...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=amp_dtype
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("Teacher loaded.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Optimizer ---
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scaler    = torch.amp.GradScaler("cuda") if use_scaler else None

    # --- Data ---
    data_gen = make_token_chunks(tokenizer, args.seq_len)

    def next_chunk():
        nonlocal data_gen
        try:
            return next(data_gen)
        except StopIteration:
            data_gen = make_token_chunks(tokenizer, args.seq_len)
            return next(data_gen)

    # --- Training state (persistent across steps) ---
    le_state = student.init_state(batch_size=1)

    print(f"\nStarting distillation: {args.steps} steps | lr={args.lr} | T={args.temperature}")
    print("=" * 60)

    loss_step1    = None
    loss_last     = None
    grad_verified = False

    for step in range(1, args.steps + 1):
        input_ids = next_chunk().unsqueeze(0).to(device)   # [1, seq_len]

        with torch.no_grad():
            teacher_last = teacher(input_ids=input_ids).logits[:, -1, :].float()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = student(input_ids, state=le_state, step=step)

        if outputs.action == "VOLUNTARY_END":
            print(f"  Step {step}: VOLUNTARY_END — resetting state.")
            le_state = student.init_state(batch_size=1)
            continue

        # Detach state for truncated BPTT
        le_state = FullState(
            **{k: (v.detach() if isinstance(v, torch.Tensor) else v)
               for k, v in vars(outputs.state).items()}
        )

        kl   = kl_distill_loss(outputs.logits.float(), teacher_last, T=args.temperature)
        loss = kl + outputs.losses["L_total"]

        if not torch.isfinite(loss):
            print(f"Step {step}: NaN/Inf loss — stopping.")
            break

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if step == 1 and not grad_verified:
            grad_verified = verify_gradient_flow(student)
            print()

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        loss_val = loss.item()
        if loss_step1 is None:
            loss_step1 = loss_val
        loss_last = loss_val

        if step % args.log_every == 0 or step == 1:
            kl_val = kl.item()
            l_reg  = outputs.losses.get("L_reg", torch.tensor(0.0)).item()
            print(f"Step {step:4d}/{args.steps} | loss={loss_val:.4f} | "
                  f"kl={kl_val:.4f} | L_reg={l_reg:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Loss at step 1:      {loss_step1:.4f}")
    print(f"  Loss at step {args.steps}: {loss_last:.4f}")
    decreased = loss_last is not None and loss_step1 is not None and loss_last < loss_step1
    print(f"  Loss decreased:      {'YES' if decreased else 'NO'}")
    print(f"  Gradient flow:       {'PASS' if grad_verified else 'FAIL'}")
    print()
    if decreased and grad_verified:
        print("PASS — LE distillation scaffold works.")
    else:
        print("ISSUES DETECTED — see above.")

    # Save model weights only (LE state is ephemeral during step 3)
    torch.save({"model_state": student.state_dict(), "step": args.steps}, args.out)
    print(f"\nCheckpoint saved: {args.out}")
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
