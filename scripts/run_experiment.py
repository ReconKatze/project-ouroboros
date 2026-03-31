#!/usr/bin/env python3
"""Step 4: d_state gradient experiment — 4-variant head-to-head comparison.

Trains four Mamba hybrid configurations under identical distillation conditions
and compares validation loss to answer: does an exponential d_state schedule
(Variant D) outperform uniform d_state (Variant A)?

Key improvements over v1:
  - Data pre-tokenized once, shared across all variants (fair + fast)
  - Held-out validation set evaluated every --eval-every steps (clean signal)
  - EMA-smoothed training loss display
  - LR warmup (first 200 steps) to stabilise variant D's larger layers
  - Optional gradient accumulation (--grad-accum) for even smoother updates
  - Comparison table driven by val loss, not noisy per-step training loss

Variants:
    A          — uniform small:   d_state=16  for all 18 Mamba layers
    B          — uniform large:   d_state=64  for all 18 Mamba layers
    C          — three-tier:      d_state 16 (×6), 64 (×6), 128 (×6)
    D_constant — exponential [16→256] + per-layer constant gate sigmoid(γ_i)
    D_proper   — exponential [16→256] + shared-β depth gate sigmoid(β·depth + γ_i)

Usage (Colab A100):
    python scripts/run_experiment.py --steps 10000 --batch-size 8
    python scripts/run_experiment.py --steps 10000 --batch-size 8 --variants D_constant,D_proper

AMP: bfloat16 on A100/H100 (no GradScaler needed), float16 on T4 (GradScaler).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from chimera.models.hybrid_model import HybridChimeraModel
from chimera.models.beta_mamba import BetaGatedMamba
from chimera.utils.layer_plan import build_layer_plan, ATTN_KEEP_DEFAULTS


# ---------------------------------------------------------------------------
# AMP dtype detection
# ---------------------------------------------------------------------------

def get_amp_dtype():
    """bfloat16 on A100/H100 (native HW, no scaler needed), float16 on T4/V100."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def make_d_states(spec: dict, n_mamba: int) -> list:
    """Generate a d_state list for n_mamba layers from a schedule spec.

    Specs:
        {"type": "uniform",     "d": 64}
        {"type": "exponential", "d_min": 16, "d_max": 256}
        {"type": "three_tier"}   -- uses d=16/64/128 split evenly across thirds
    """
    t = spec["type"]
    if t == "uniform":
        return [spec["d"]] * n_mamba
    elif t == "exponential":
        d_min, d_max = spec["d_min"], spec["d_max"]
        return [
            int(round(d_min * (d_max / d_min) ** (i / max(n_mamba - 1, 1))))
            for i in range(n_mamba)
        ]
    elif t == "bell":
        # Reverse bell (U-shape): high at edges, low in middle.
        # Uses a cosine: d_max at i=0 and i=n-1, d_min at i=(n-1)/2.
        d_min, d_max = spec["d_min"], spec["d_max"]
        return [
            int(round(d_max - (d_max - d_min) *
                      (1 - math.cos(2 * math.pi * i / max(n_mamba - 1, 1))) / 2))
            for i in range(n_mamba)
        ]
    elif t == "three_tier":
        tier = n_mamba // 3
        return [16] * tier + [64] * tier + [128] * (n_mamba - 2 * tier)
    else:
        raise ValueError(f"Unknown d_state_spec type: {t!r}")


VARIANTS = {
    "A": {
        "d_state_spec": {"type": "uniform", "d": 16},
        "gate_mode":    None,
        "label":        "Uniform small (d=16)",
    },
    "B": {
        "d_state_spec": {"type": "uniform", "d": 64},
        "gate_mode":    None,
        "label":        "Uniform large (d=64)",
    },
    "C": {
        "d_state_spec": {"type": "three_tier"},
        "gate_mode":    None,
        "label":        "Three-tier (d=16/64/128)",
    },
    "D_constant": {
        "d_state_spec": {"type": "exponential", "d_min": 16, "d_max": 256},
        "gate_mode":    "constant",
        "label":        "Exp gradient + constant gate (γ only)",
    },
    "D_proper": {
        "d_state_spec": {"type": "exponential", "d_min": 16, "d_max": 256},
        "gate_mode":    "shared_beta",
        "label":        "Exp gradient + shared-β depth gate",
    },
    "D_bell": {
        "d_state_spec": {"type": "bell", "d_min": 16, "d_max": 256},
        "gate_mode":    "constant",
        "label":        "Reverse bell (d=256→16→256) + constant gate",
    },
    "D_exp_inv": {
        "d_state_spec": {"type": "exponential", "d_min": 256, "d_max": 16},
        "gate_mode":    "constant",
        "label":        "Inv. exponential (d=256→16) + constant gate",
    },
}


# ---------------------------------------------------------------------------
# Data: pre-tokenise once, share across all variants
# ---------------------------------------------------------------------------

def load_data(tokenizer, seq_len, n_train, n_val):
    """Pre-tokenise wikitext-103 train and test splits into fixed-length chunks.

    Returns:
        train_chunks: list of n_train LongTensors, shape [seq_len]
        val_chunks:   list of n_val   LongTensors, shape [seq_len]
    """
    def _collect(split, n):
        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1",
            split=split, streaming=True, trust_remote_code=False,
        )
        chunks, buf = [], []
        for ex in ds:
            if len(chunks) >= n:
                break
            text = ex["text"].strip()
            if not text:
                continue
            buf.extend(tokenizer.encode(text, add_special_tokens=False))
            while len(buf) >= seq_len and len(chunks) < n:
                chunks.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                buf = buf[seq_len:]
        return chunks

    print(f"Pre-tokenising data (train={n_train}, val={n_val}, seq_len={seq_len})...")
    train_chunks = _collect("train", n_train)
    val_chunks   = _collect("test",  n_val)
    print(f"  Loaded {len(train_chunks)} train chunks, {len(val_chunks)} val chunks.")
    return train_chunks, val_chunks


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def distill_loss(student_logits, teacher_logits, T=2.0):
    """Temperature-scaled KL with per-token std normalisation."""
    s = student_logits / (student_logits.std(dim=-1, keepdim=True) + 1e-6)
    t = teacher_logits / (teacher_logits.std(dim=-1, keepdim=True) + 1e-6)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t     = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(student, teacher, val_chunks, device, temperature, amp_dtype):
    """Compute mean distillation loss on the held-out val set (no grad)."""
    student.eval()
    total = 0.0
    with torch.no_grad():
        for chunk in val_chunks:
            input_ids = chunk.unsqueeze(0).to(device)
            teacher_logits = teacher(input_ids=input_ids).logits
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                full_logits    = student(input_ids)
                student_logits = full_logits[:, student.num_sinks:, :]
            loss = distill_loss(student_logits.float(), teacher_logits.float(), temperature)
            total += loss.item()
    student.train()
    return total / len(val_chunks)


# ---------------------------------------------------------------------------
# Student builder
# ---------------------------------------------------------------------------

def build_variant_student(args, cfg, device):
    """Build HybridChimeraModel for one variant with the correct d_state schedule."""
    import torch.nn as nn

    base = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.float32
    ).to(device)

    # Resolve attention layer indices — explicit override or model-type default
    if args.attn_indices is not None:
        attn_keep = {int(x) for x in args.attn_indices.split(",")}
    else:
        attn_keep = None  # build_layer_plan uses model-type preset

    effective_attn = (
        attn_keep if attn_keep is not None
        else ATTN_KEEP_DEFAULTS.get(base.config.model_type, {0, 3, 7, 11})
    )
    n_mamba  = base.config.num_hidden_layers - len(effective_attn)
    d_states = make_d_states(cfg["d_state_spec"], n_mamba)
    print(f"  Attn layers: {sorted(effective_attn)}  "
          f"({len(effective_attn)} attn, {n_mamba} Mamba) | "
          f"d_state: {d_states[0]}→{d_states[-1]}")

    layer_plan = build_layer_plan(
        base.config.num_hidden_layers,
        attn_keep_indices=attn_keep,
        model_type=base.config.model_type,
        d_state=d_states,
    )

    hybrid = HybridChimeraModel(base, layer_plan, num_sinks=4, device=str(device))

    # Gate variants: wrap Mamba blocks with BetaGatedMamba BEFORE freeze
    gate_mode = cfg.get("gate_mode")
    if gate_mode in ("constant", "shared_beta"):
        total_mamba = sum(1 for p in hybrid.layer_plan if p["kind"] == "mamba")
        shared_beta = nn.Parameter(torch.zeros(1)) if gate_mode == "shared_beta" else None
        mamba_idx = 0
        for plan in hybrid.layer_plan:
            if plan["kind"] == "mamba":
                wrapper = hybrid.adapter.get_attention(hybrid.layers[plan["layer_idx"]])
                wrapper.mamba = BetaGatedMamba(
                    wrapper.mamba, mamba_idx, total_mamba, shared_beta=shared_beta
                )
                mamba_idx += 1
        if shared_beta is not None:
            hybrid.register_parameter("shared_beta", shared_beta)

    # Freeze all, then unfreeze Mamba wrappers + sink tokens
    for p in hybrid.parameters():
        p.requires_grad_(False)
    for plan in hybrid.layer_plan:
        if plan["kind"] == "mamba":
            wrapper = hybrid.adapter.get_attention(hybrid.layers[plan["layer_idx"]])
            for p in wrapper.parameters():
                p.requires_grad_(True)
    hybrid.sink_tokens.sinks.requires_grad_(True)
    # shared_beta is registered on hybrid (not on any wrapper) — unfreeze explicitly
    if hasattr(hybrid, "shared_beta"):
        hybrid.shared_beta.requires_grad_(True)

    hybrid = hybrid.to(device)
    hybrid.train()

    trainable = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in hybrid.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")
    return hybrid


# ---------------------------------------------------------------------------
# Training loop for one variant
# ---------------------------------------------------------------------------

def train_variant(name, cfg, args, teacher, train_chunks, val_chunks, device, amp_dtype, effective_lr):
    """Train one variant. Returns dict of val losses keyed by step."""
    print(f"\n{'='*60}")
    print(f"Variant {name}: {cfg['label']}")
    print(f"{'='*60}")

    student          = build_variant_student(args, cfg, device)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer        = AdamW(trainable_params, lr=effective_lr, weight_decay=0.01)
    use_scaler       = (amp_dtype == torch.float16)
    scaler           = torch.amp.GradScaler("cuda") if use_scaler else None

    # Linear LR warmup over first warmup_steps optimizer steps
    warmup_steps = min(200, args.steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, (s + 1) / warmup_steps) if warmup_steps > 0 else 1.0,
    )

    val_log   = {}   # {step: val_loss}
    ema_loss  = None
    ema_alpha = 0.95
    chunk_idx = 0
    accum_steps = 0

    # Initial validation before any training
    print(f"  [{name}] Step 0 — initial val loss...")
    val_loss = evaluate(student, teacher, val_chunks, device, args.temperature, amp_dtype)
    val_log[0] = val_loss
    print(f"  [{name}] Step    0/{args.steps} | val={val_loss:.4f}  (pre-training)")

    for step in range(1, args.steps + 1):
        batch_chunks = [
            train_chunks[(chunk_idx + i) % len(train_chunks)]
            for i in range(args.batch_size)
        ]
        input_ids = torch.stack(batch_chunks).to(device)   # [batch_size, seq_len]
        chunk_idx += args.batch_size

        # Teacher forward
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids).logits

        # Student forward
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            full_logits    = student(input_ids)
            student_logits = full_logits[:, student.num_sinks:, :]

        loss = distill_loss(
            student_logits.float(),
            teacher_logits.float(),
            T=args.temperature,
        )

        if not torch.isfinite(loss):
            print(f"  [{name}] Step {step}: NaN/Inf loss — stopping variant.")
            break

        # Accumulate gradients (scale by grad_accum so effective LR is consistent)
        if use_scaler:
            scaler.scale(loss / args.grad_accum).backward()
        else:
            (loss / args.grad_accum).backward()
        accum_steps += 1

        if accum_steps == args.grad_accum:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum_steps = 0

        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * loss_val

        # Validation eval
        if step % args.eval_every == 0:
            val_loss = evaluate(student, teacher, val_chunks, device, args.temperature, amp_dtype)
            val_log[step] = val_loss
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{name}] Step {step:5d}/{args.steps} | "
                  f"train_ema={ema_loss:8.3f} | val={val_loss:8.3f} | lr={lr_now:.2e}")

        elif step % args.log_every == 0 or step == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{name}] Step {step:5d}/{args.steps} | "
                  f"train_ema={ema_loss:8.3f} | lr={lr_now:.2e}")

    # Final validation if not already done at last step
    if args.steps not in val_log:
        val_loss = evaluate(student, teacher, val_chunks, device, args.temperature, amp_dtype)
        val_log[args.steps] = val_loss
        print(f"  [{name}] Step {args.steps:5d}/{args.steps} | val={val_loss:.4f}  (final)")

    # Gate variants: report learned β, γ, α per Mamba layer
    gate_mode = cfg.get("gate_mode")
    if gate_mode in ("constant", "shared_beta"):
        if gate_mode == "shared_beta":
            shared_b = student.shared_beta.item()
            print(f"\n  Shared β = {shared_b:+.4f}")
        print(f"  Learned γ and α per Mamba layer:")
        for plan in student.layer_plan:
            if plan["kind"] == "mamba":
                wrapper = student.adapter.get_attention(student.layers[plan["layer_idx"]])
                bg = wrapper.mamba
                g  = bg.gamma.item()
                if gate_mode == "shared_beta":
                    alpha = torch.sigmoid(student.shared_beta * bg.depth + bg.gamma).item()
                    print(f"    layer {plan['layer_idx']:2d} depth={bg.depth:.2f}: "
                          f"γ={g:+.3f}  →  α={alpha:.3f}")
                else:
                    alpha = torch.sigmoid(bg.gamma).item()
                    print(f"    layer {plan['layer_idx']:2d}: γ={g:+.3f}  →  α={alpha:.3f}")

    del student, optimizer, scheduler
    if scaler is not None:
        del scaler
    torch.cuda.empty_cache()

    return val_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Step 4: d_state gradient experiment")
    p.add_argument("--teacher",     default="Qwen/Qwen2.5-3B")
    p.add_argument("--student",     default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--steps",       type=int,   default=5000)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--seq-len",     type=int,   default=512)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--grad-accum",  type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = grad_accum × 1)")
    p.add_argument("--eval-every",  type=int,   default=250,
                   help="Evaluate on val set every N steps (clean signal)")
    p.add_argument("--log-every",   type=int,   default=100,
                   help="Log training EMA loss every N steps")
    p.add_argument("--n-val",       type=int,   default=100,
                   help="Number of held-out val chunks")
    p.add_argument("--batch-size",  type=int,   default=4,
                   help="Sequences per gradient step. Default 4 suits A100 80GB comfortably.")
    p.add_argument("--variants",     default="D_constant,D_proper",
                   help="Comma-separated variants to run, e.g. 'B,D_constant' or 'A,B,C,D_constant,D_proper'")
    p.add_argument("--attn-indices", default=None,
                   help="Comma-separated attention layer indices, e.g. '0,11,23' for 3-attention "
                        "variant. Default: model-type preset ({0,4,8,12,16,23} for Qwen2.5-0.5B).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    variant_names = [v.strip() for v in args.variants.split(",")]
    selected = {k: VARIANTS[k] for k in variant_names if k in VARIANTS}
    unknown = [k for k in variant_names if k not in VARIANTS]
    if unknown:
        print(f"Unknown variants: {unknown}. Available: {list(VARIANTS.keys())}")
    if not selected:
        print("No valid variants selected.")
        return

    # --- AMP dtype ---
    amp_dtype  = get_amp_dtype()
    dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
    print(f"AMP dtype: {dtype_name}  |  GradScaler: {'no (bfloat16)' if amp_dtype == torch.bfloat16 else 'yes'}")

    n_train = args.steps * args.batch_size + 500   # enough chunks, never repeat within a variant

    # Linear LR scaling: base --lr was tuned at batch_size=1
    effective_lr = args.lr * args.batch_size
    print(f"LR: {args.lr} × batch_size {args.batch_size} → effective LR {effective_lr:.2e}")
    print(f"\nVariants: {list(selected.keys())}  |  Steps each: {args.steps}  |  "
          f"Grad accum: {args.grad_accum}  |  Val every: {args.eval_every}")

    # Load teacher once
    print(f"\nLoading teacher: {args.teacher} ({dtype_name})...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=amp_dtype
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("Teacher loaded.")

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenise once, shared across all variants
    train_chunks, val_chunks = load_data(tokenizer, args.seq_len, n_train, args.n_val)

    # Run all selected variants
    all_val_logs = {}
    for name, cfg in selected.items():
        all_val_logs[name] = train_variant(
            name, cfg, args, teacher, train_chunks, val_chunks, device, amp_dtype, effective_lr
        )

    # Comparison table (val loss)
    print("\n" + "=" * 72)
    print("EXPERIMENT RESULTS  (validation loss — lower is better)")
    print("=" * 72)

    # Collect final val losses
    results = {}
    for name, val_log in all_val_logs.items():
        sorted_steps = sorted(val_log.keys())
        mid_step  = sorted_steps[len(sorted_steps) // 2]
        results[name] = {
            "label":      VARIANTS[name]["label"],
            "val_start":  val_log.get(0,           float("nan")),
            "val_mid":    val_log.get(mid_step,     float("nan")),
            "val_final":  val_log[sorted_steps[-1]],
        }

    baseline = results.get("A", {}).get("val_final", None)

    print(f"{'Var':<4} {'d_state schedule':<32} {'Val@0':>8} {'Val@mid':>8} {'Val@end':>8} {'vs A':>8}")
    print("-" * 72)
    for name, r in results.items():
        if baseline and math.isfinite(r["val_final"]) and math.isfinite(baseline) and name != "A":
            pct   = (r["val_final"] - baseline) / baseline * 100
            vs_a  = f"{pct:+.1f}%"
        elif name == "A":
            vs_a = "(base)"
        else:
            vs_a = "N/A"
        label_s = r["label"][:30]
        print(f"{name:<4} {label_s:<32} {r['val_start']:>8.2f} "
              f"{r['val_mid']:>8.2f} {r['val_final']:>8.2f} {vs_a:>8}")

    print("=" * 72)

    # Go/no-go: B vs D_constant answers whether exponential d_state beats uniform
    if "D_constant" in results and "B" in results:
        b_val  = results["B"]["val_final"]
        dc_val = results["D_constant"]["val_final"]
        if math.isfinite(b_val) and math.isfinite(dc_val):
            improvement = (b_val - dc_val) / b_val * 100
            print(f"\nD_constant vs B: {improvement:+.2f}%")
            if improvement >= 10.0:
                print("GO — Exponential d_state clearly outperforms uniform. Use for 9B Chimera.")
            elif improvement > 2.0:
                print("WEAK SIGNAL — D_constant better but under 10% threshold.")
            elif improvement > 0:
                print("MARGINAL — Within noise floor. d_state schedule has no clear effect.")
            else:
                print("NULL RESULT — d_state schedule has no effect at this configuration.")

    # Go/no-go: D_proper vs D_constant answers whether depth-aware gating adds value
    if "D_proper" in results and "D_constant" in results:
        dc = results["D_constant"]["val_final"]
        dp = results["D_proper"]["val_final"]
        if math.isfinite(dc) and math.isfinite(dp):
            improvement = (dc - dp) / dc * 100
            print(f"\nD_proper improvement over D_constant: {improvement:+.2f}%")
            if improvement > 0:
                print("RESULT: Shared-β depth gate adds value — use depth-aware gating for 9B Chimera.")
            else:
                print("RESULT: Depth-awareness adds no value over per-layer constants.")
                print("        Use sigmoid(γ_i) per layer; drop β entirely.")
    elif "A" in results:
        best = min(results, key=lambda k: results[k]["val_final"])
        a = results["A"]["val_final"]
        b = results[best]["val_final"]
        if math.isfinite(a) and math.isfinite(b):
            improvement = (a - b) / a * 100
            print(f"\nBest variant: {best} — {improvement:.1f}% improvement over A")
            if improvement >= 10.0:
                print("GO — Use this d_state schedule for 9B Chimera.")
            else:
                print("NOTE — No variant reached ≥10% improvement threshold over A.")

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
