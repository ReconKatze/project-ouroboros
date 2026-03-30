#!/usr/bin/env python3
"""Step 4: d_state gradient experiment — 4-variant head-to-head comparison.

Trains four Mamba hybrid configurations under identical distillation conditions
and compares final KL loss to answer: does an exponential d_state schedule
(Variant D) outperform uniform d_state (Variant A)?

Variants:
    A — uniform small:  d_state=16  for all 18 Mamba layers
    B — uniform large:  d_state=64  for all 18 Mamba layers
    C — three-tier:     d_state 16 (×6), 64 (×6), 128 (×6)
    D — exponential gradient: [16,16,24,24,...,256,256] + β depth gate

Usage (Colab T4):
    python scripts/run_experiment.py
    python scripts/run_experiment.py --steps 200 --variants AD
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
from chimera.utils.layer_plan import build_layer_plan


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS = {
    "A": {
        "d_states":  [16] * 18,
        "use_beta":  False,
        "label":     "Uniform small (d=16)",
    },
    "B": {
        "d_states":  [64] * 18,
        "use_beta":  False,
        "label":     "Uniform large (d=64)",
    },
    "C": {
        "d_states":  [16]*6 + [64]*6 + [128]*6,
        "use_beta":  False,
        "label":     "Three-tier (d=16/64/128)",
    },
    "D": {
        "d_states":  [16, 16, 24, 24, 32, 32, 48, 48,
                      64, 64, 96, 96, 128, 128, 192, 192, 256, 256],
        "use_beta":  True,
        "label":     "Exponential gradient + β gate",
    },
}


# ---------------------------------------------------------------------------
# Dataset (same as train_distill.py)
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
            yield torch.tensor(buffer[:seq_len], dtype=torch.long)
            buffer = buffer[seq_len:]


# ---------------------------------------------------------------------------
# Loss (same as train_distill.py)
# ---------------------------------------------------------------------------

def distill_loss(student_logits, teacher_logits, T=2.0):
    s = student_logits / (student_logits.std(dim=-1, keepdim=True) + 1e-6)
    t = teacher_logits / (teacher_logits.std(dim=-1, keepdim=True) + 1e-6)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t     = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)


# ---------------------------------------------------------------------------
# Student builder
# ---------------------------------------------------------------------------

def build_variant_student(args, cfg, device):
    """Build HybridChimeraModel for one variant with correct d_state schedule."""
    base = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.float32
    ).to(device)

    layer_plan = build_layer_plan(
        base.config.num_hidden_layers,
        model_type=base.config.model_type,
        d_state=cfg["d_states"],
    )

    hybrid = HybridChimeraModel(base, layer_plan, num_sinks=4, device=str(device))

    # Variant D: wrap Mamba blocks with BetaGatedMamba BEFORE freeze
    if cfg["use_beta"]:
        total_mamba = sum(1 for p in hybrid.layer_plan if p["kind"] == "mamba")
        mamba_idx = 0
        for plan in hybrid.layer_plan:
            if plan["kind"] == "mamba":
                wrapper = hybrid.adapter.get_attention(hybrid.layers[plan["layer_idx"]])
                wrapper.mamba = BetaGatedMamba(wrapper.mamba, mamba_idx, total_mamba)
                mamba_idx += 1

    # Freeze all, then unfreeze Mamba wrappers + sink tokens
    for p in hybrid.parameters():
        p.requires_grad_(False)

    for plan in hybrid.layer_plan:
        if plan["kind"] == "mamba":
            wrapper = hybrid.adapter.get_attention(hybrid.layers[plan["layer_idx"]])
            for p in wrapper.parameters():
                p.requires_grad_(True)

    hybrid.sink_tokens.sinks.requires_grad_(True)

    hybrid = hybrid.to(device)
    hybrid.train()

    trainable = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in hybrid.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")
    return hybrid


# ---------------------------------------------------------------------------
# Training loop for one variant
# ---------------------------------------------------------------------------

def train_variant(name, cfg, args, teacher, tokenizer, device):
    """Train one variant and return its loss history."""
    print(f"\n{'='*60}")
    print(f"Variant {name}: {cfg['label']}")
    print(f"{'='*60}")

    student  = build_variant_student(args, cfg, device)
    data_gen = make_token_chunks(tokenizer, args.seq_len)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scaler    = torch.amp.GradScaler("cuda")

    loss_history = []

    for step in range(1, args.steps + 1):
        # Next batch
        try:
            chunk = next(data_gen)
        except StopIteration:
            data_gen = make_token_chunks(tokenizer, args.seq_len)
            chunk = next(data_gen)
        input_ids = chunk.unsqueeze(0).to(device)

        # Teacher forward
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids).logits

        # Student forward
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            full_logits    = student(input_ids)
            student_logits = full_logits[:, student.num_sinks:, :]

        loss = distill_loss(
            student_logits.float(),
            teacher_logits.float(),
            T=args.temperature,
        )

        if not torch.isfinite(loss):
            print(f"  [{name}] Step {step}: NaN/Inf loss — stopping variant.")
            loss_history.append(float("nan"))
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % args.log_every == 0 or step == 1:
            print(f"  [{name}] Step {step:4d}/{args.steps} | loss={loss_val:.4f}")

    # Variant D: report learned β and γ per layer
    if cfg["use_beta"]:
        print(f"\n  Variant D — learned β, γ, and α per Mamba layer:")
        for plan in student.layer_plan:
            if plan["kind"] == "mamba":
                wrapper = student.adapter.get_attention(
                    student.layers[plan["layer_idx"]]
                )
                bg    = wrapper.mamba
                b     = bg.beta.item()
                g     = bg.gamma.item()
                alpha = torch.sigmoid(bg.beta * bg.depth + bg.gamma).item()
                print(f"    layer {plan['layer_idx']:2d} depth={bg.depth:.2f}: "
                      f"β={b:+.3f}  γ={g:+.3f}  →  α={alpha:.3f}")

    # Free GPU memory before next variant
    del student, optimizer, scaler
    torch.cuda.empty_cache()

    return loss_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Step 4: d_state gradient experiment")
    p.add_argument("--teacher",      default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--student",      default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--steps",        type=int,   default=200)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--seq-len",      type=int,   default=512)
    p.add_argument("--temperature",  type=float, default=2.0)
    p.add_argument("--log-every",    type=int,   default=20)
    p.add_argument("--variants",     default="ABCD",
                   help="Which variants to run, e.g. 'ABCD' or 'AD'")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    selected = {k: VARIANTS[k] for k in args.variants if k in VARIANTS}
    if not selected:
        print(f"No valid variants in '{args.variants}'. Choose from A B C D.")
        return

    print(f"\nRunning variants: {list(selected.keys())}")
    print(f"Steps per variant: {args.steps}  |  Total: {args.steps * len(selected)}")

    # Load teacher once — kept for entire experiment
    print(f"\nLoading teacher: {args.teacher} (float16)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.float16
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("Teacher loaded.")

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run all selected variants
    results = {}
    for name, cfg in selected.items():
        history = train_variant(name, cfg, args, teacher, tokenizer, device)
        mid_idx = max(0, len(history) // 2 - 1)
        results[name] = {
            "label":      cfg["label"],
            "loss_start": history[0]      if history else float("nan"),
            "loss_mid":   history[mid_idx] if history else float("nan"),
            "loss_final": history[-1]     if history else float("nan"),
        }

    # Comparison table
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    baseline = results.get("A", {}).get("loss_final", None)

    header = f"{'Var':<4} {'d_state schedule':<32} {'Loss@1':>9} {'Loss@mid':>9} {'Loss@end':>9} {'vs A':>8}"
    print(header)
    print("-" * 70)

    for name, r in results.items():
        if baseline and math.isfinite(r["loss_final"]) and math.isfinite(baseline) and name != "A":
            pct = (r["loss_final"] - baseline) / baseline * 100
            vs_a = f"{pct:+.1f}%"
        elif name == "A":
            vs_a = "(base)"
        else:
            vs_a = "N/A"

        label_short = r["label"][:30]
        print(f"{name:<4} {label_short:<32} {r['loss_start']:>9.2f} "
              f"{r['loss_mid']:>9.2f} {r['loss_final']:>9.2f} {vs_a:>8}")

    print("=" * 70)

    # Go/no-go verdict
    if "A" in results and "D" in results:
        a_final = results["A"]["loss_final"]
        d_final = results["D"]["loss_final"]
        if math.isfinite(a_final) and math.isfinite(d_final):
            improvement = (a_final - d_final) / a_final * 100
            print(f"\nVariant D improvement over A: {improvement:.1f}%")
            if improvement >= 10.0:
                print("GO — Variant D beats baseline by ≥10%. Use exponential d_state + β for 9B Chimera.")
            else:
                best = min(results, key=lambda k: results[k]["loss_final"])
                print(f"NOTE — Variant D did not reach ≥10% threshold.")
                print(f"Best variant: {best} ({results[best]['label']})")

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
