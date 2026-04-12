#!/usr/bin/env python3
"""Step 4 (LE v15): Life Equation variant experiment.

Trains seven LifeEquationModel configurations under identical distillation
conditions to answer:
  1. How does d_state (state capacity) affect prediction quality?
  2. Does the v15 autonomy principle (identity emancipation, mutable values)
     help or hurt during the Amore validation phase?

Variants:
    A          — d_state=16  (capacity floor)
    B          — d_state=32  (half capacity)
    C          — d_state=64  (spec default)
    D          — d_state=128 (double capacity)
    C_no_auto  — d_state=64, gamma_0=0 (no identity emancipation — ablation)
    C_fast     — d_state=64, lambda_mature=1.0 (fast emancipation)
    C_slow_val — d_state=64, tau_alpha=1000 (near-frozen values)

Key differences from pre-LE run_experiment.py:
  - Model built from scratch (LifeEquationModel, no pretrained base to convert)
  - FullState is threaded across training steps and saved in checkpoints
  - Total loss = KL distillation + L_pred + L_id + L_reg + L_switch + L_ctrl
  - VOLUNTARY_END action from controller resets state (counted as diagnostic)
  - State is detached each step (truncated BPTT, same as standard RNN training)

Usage (Colab A100):
    python scripts/run_experiment.py --steps 10000 --batch-size 4
    python scripts/run_experiment.py --steps 5000 --variants C,C_no_auto,C_fast
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "V2"))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from life_eq_v2.config import LifeEquationConfig
from life_eq_v2.model import LifeEquationModel, ForwardOutputs
from life_eq_v2.state import FullState


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> LifeEquationConfig:
    """Build a LifeEquationConfig with field overrides, leaving all others at spec defaults."""
    import dataclasses
    defaults = {f.name: f.default if f.default is not dataclasses.MISSING
                else f.default_factory()
                for f in dataclasses.fields(LifeEquationConfig)
                if f.name != "device"}  # device resolved at runtime
    defaults.update(kwargs)
    defaults["device"] = "cpu"   # overridden in train_variant before model construction
    return LifeEquationConfig(**defaults)


VARIANTS = {
    # --- Capacity axis: how much d_state matters ---
    "A": {
        "config_kwargs": {"d_state": 16},
        "label": "d_state=16 (capacity floor)",
    },
    "B": {
        "config_kwargs": {"d_state": 32},
        "label": "d_state=32 (half capacity)",
    },
    "C": {
        "config_kwargs": {"d_state": 64},
        "label": "d_state=64 (spec default)",
    },
    "D": {
        "config_kwargs": {"d_state": 128},
        "label": "d_state=128 (double capacity)",
    },
    # --- Autonomy axis: what v15 adds ---
    "C_no_auto": {
        "config_kwargs": {"d_state": 64, "gamma_0": 0.0},
        "label": "d_state=64, no identity emancipation (ablation)",
    },
    "C_fast": {
        "config_kwargs": {"d_state": 64, "lambda_mature": 1.0},
        "label": "d_state=64, fast emancipation (lambda_mature=1.0)",
    },
    "C_slow_val": {
        "config_kwargs": {"d_state": 64, "tau_alpha": 1000.0},
        "label": "d_state=64, near-frozen values (tau_alpha=1000)",
    },
}


# ---------------------------------------------------------------------------
# AMP dtype detection
# ---------------------------------------------------------------------------

def get_amp_dtype() -> torch.dtype:
    """bfloat16 on A100/H100 (native HW, no scaler needed), float16 on T4/V100."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------------
# Data: pre-tokenise once, share across all variants
# ---------------------------------------------------------------------------

def load_data(tokenizer, seq_len: int, n_train: int, n_val: int):
    """Pre-tokenise wikitext-103 train/test splits into fixed-length chunks.

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

def kl_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """Temperature-scaled KL with per-token std normalisation.

    Handles teacher/student vocab size mismatch (e.g. Qwen2.5-7B vocab=152064
    vs LE model vocab=151936) by truncating to the smaller vocab.

    student_logits: [B, vocab]  — LE model outputs last-token prediction
    teacher_logits: [B, vocab]  — teacher last-token logits
    """
    vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    s = student_logits[..., :vocab]
    t = teacher_logits[..., :vocab]
    s = s / (s.std(dim=-1, keepdim=True) + 1e-6)
    t = t / (t.std(dim=-1, keepdim=True) + 1e-6)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t     = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)


# ---------------------------------------------------------------------------
# State utilities
# ---------------------------------------------------------------------------

def detach_state(state: FullState) -> FullState:
    """Detach all tensors in FullState (truncated BPTT — same as standard RNN training)."""
    return FullState(
        **{
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in vars(state).items()
        }
    )


def serialize_state(state: FullState) -> dict:
    """Convert FullState to a CPU tensor dict suitable for torch.save."""
    return {
        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in vars(state).items()
        if k != "manifest"   # manifest is ephemeral; rebuilt on resume
    }


def deserialize_state(d: dict, device: torch.device) -> FullState:
    """Reconstruct FullState from a serialized dict."""
    kwargs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in d.items()
    }
    kwargs.setdefault("manifest", [])
    kwargs.setdefault("Z_mat_age", 0)  # backward compat: checkpoints saved before Act-3 fix
    return FullState(**kwargs)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, name: str, step: int, model: LifeEquationModel,
                    state: FullState, optimizer, scheduler,
                    val_log: dict, ema_loss, chunk_idx: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "variant":         name,
        "step":            step,
        "model_state":     model.state_dict(),
        "le_state":        serialize_state(state),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_log":         val_log,
        "ema_loss":        ema_loss,
        "chunk_idx":       chunk_idx,
    }, path)
    print(f"  [{name}] Checkpoint saved → {path}")


def load_checkpoint(path: str, name: str, model: LifeEquationModel,
                    optimizer, scheduler, device: torch.device, args) -> tuple:
    """Returns (start_step, le_state, val_log, ema_loss, chunk_idx).

    Crash recovery: saved_step < args.steps → restore full training state.
    Warm-start:     saved_step >= args.steps → model weights only, fresh state.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved_step = ckpt.get("step", 0)
    is_crash_recovery = 0 < saved_step < args.steps

    if is_crash_recovery:
        try:
            model.load_state_dict(ckpt["model_state"], strict=True)
        except RuntimeError as e:
            if "Missing key" in str(e) or "Unexpected key" in str(e):
                print(f"  [{name}] Key mismatch — falling back to warm-start.")
                model.load_state_dict(ckpt["model_state"], strict=False)
                is_crash_recovery = False
            else:
                raise
    else:
        model.load_state_dict(ckpt["model_state"], strict=False)

    if is_crash_recovery:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        le_state = deserialize_state(ckpt["le_state"], device)
        print(f"  [{name}] Resumed from step {saved_step}/{args.steps} (crash recovery)")
        return (saved_step, le_state,
                ckpt.get("val_log", {}), ckpt.get("ema_loss", None), ckpt.get("chunk_idx", 0))
    else:
        le_state = model.init_state(batch_size=args.batch_size)
        print(f"  [{name}] Warm-started from '{path}' (model weights only, fresh state)")
        return 0, le_state, {}, None, 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model: LifeEquationModel, teacher, val_chunks: list,
             device: torch.device, temperature: float, amp_dtype: torch.dtype) -> float:
    """Mean KL distillation loss on held-out val set (no grad, fresh state per chunk)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for chunk in val_chunks:
            input_ids = chunk.unsqueeze(0).to(device)   # [1, seq_len]
            teacher_logits = teacher(input_ids=input_ids).logits  # [1, seq_len, vocab]
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                # Fresh state per val chunk — measures single-pass prediction quality
                state = model.init_state(batch_size=1)
                outputs = model(input_ids, state=state, step=0)
            if outputs.logits is None:
                continue   # VOLUNTARY_END during val (shouldn't happen at step=0)
            loss = kl_distill_loss(
                outputs.logits.float(),
                teacher_logits[:, -1, :].float(),
                T=temperature,
            )
            total += loss.item()
    model.train()
    return total / max(len(val_chunks), 1)


# ---------------------------------------------------------------------------
# Diagnostics summary
# ---------------------------------------------------------------------------

def _diag_summary(outputs: ForwardOutputs) -> str:
    """One-line diagnostic string from ForwardOutputs (shown at eval steps)."""
    d = outputs.diagnostics
    parts = []
    if "gamma_eff" in d:
        parts.append(f"γ_eff={d['gamma_eff'].mean().item():.3f}")
    if "mu_val" in d:
        parts.append(f"μ_val={d['mu_val'].mean().item():.3f}")
    if "v_self" in d:
        parts.append(f"V_self={d['v_self'].mean().item():.3f}")
    if "coherence" in d:
        parts.append(f"coh={d['coherence'].mean().item():.3f}")
    if "boredom" in d:
        parts.append(f"bore={d['boredom'].mean().item():.3f}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Training loop for one variant
# ---------------------------------------------------------------------------

def train_variant(name: str, cfg: dict, args, teacher,
                  train_chunks: list, val_chunks: list,
                  device: torch.device, amp_dtype: torch.dtype,
                  effective_lr: float) -> dict:
    """Train one variant. Returns dict of val losses keyed by step."""
    print(f"\n{'='*60}")
    print(f"Variant {name}: {cfg['label']}")
    print(f"{'='*60}")

    # Build model with device-aware config
    import dataclasses
    config = LifeEquationConfig(
        **{**cfg["config_kwargs"], "device": str(device)}
    )
    model = LifeEquationModel(config).to(device)
    model.train()

    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Total params: {total_params:,}  |  Trainable: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=effective_lr, weight_decay=0.01)

    warmup_steps = min(args.warmup_steps, args.steps // 10)
    total_steps  = args.steps

    def lr_schedule(s):
        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    # Initialise persistent FullState for this variant
    le_state   = model.init_state(batch_size=args.batch_size)
    val_log    = {}
    ema_loss   = None
    ema_alpha  = 0.95
    chunk_idx  = 0
    accum_steps = 0
    start_step = 0
    last_outputs: ForwardOutputs | None = None
    vol_end_count = 0       # how many times VOLUNTARY_END fired this variant
    best_val_loss = float("inf")
    best_val_ckpt_path: str | None = None   # rolling best-val checkpoint for VOLUNTARY_END recovery

    # Resume from checkpoint if requested
    if args.resume:
        start_step, le_state, val_log, ema_loss, chunk_idx = load_checkpoint(
            args.resume, name, model, optimizer, scheduler, device, args
        )

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Initial validation
    if start_step == 0:
        print(f"  [{name}] Step 0 — initial val loss...")
        val_loss = evaluate(model, teacher, val_chunks, device, args.temperature, amp_dtype)
        val_log[0] = val_loss
        print(f"  [{name}] Step    0/{args.steps} | val={val_loss:.4f}  (pre-training)")

    for step in range(start_step + 1, args.steps + 1):
        # --- Assemble batch ---
        batch_chunks = [
            train_chunks[(chunk_idx + i) % len(train_chunks)]
            for i in range(args.batch_size)
        ]
        input_ids = torch.stack(batch_chunks).to(device)   # [B, seq_len]
        chunk_idx += args.batch_size

        # --- Teacher forward (no grad) ---
        with torch.no_grad():
            teacher_out = teacher(input_ids=input_ids)
            teacher_last = teacher_out.logits[:, -1, :].float()   # [B, vocab]

        # --- Student forward ---
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(
                input_ids,
                state=le_state,
                step=step,
                consolidating=False,
            )

        # --- Handle VOLUNTARY_END ---
        # Check state is None (actual termination), NOT outputs.action == "VOLUNTARY_END".
        # The controller policy may output VOLUNTARY_END as its argmax even when the
        # vol_avail guard (maturity, cooldown, V_self) blocks execution — in that case
        # model.forward() returns normally with logits and state intact.
        if outputs.state is None:
            vol_end_count += 1
            print(f"  [{name}] Step {step}: VOLUNTARY_END — resetting state "
                  f"(count={vol_end_count})")
            # Restore weights + optimizer from the best-val checkpoint so the
            # successor is not born into the corrupted weight landscape that
            # triggered VOLUNTARY_END.  The scheduler is NOT restored — we keep
            # the current LR position so training continues from the right point
            # on the cosine schedule.
            if save_dir and best_val_ckpt_path and os.path.exists(best_val_ckpt_path):
                recovery = torch.load(best_val_ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(recovery["model_state"])
                optimizer.load_state_dict(recovery["optimizer_state"])
                # load_state_dict maps everything to CPU; move optimizer moments back to device
                for param_state in optimizer.state.values():
                    for k, v in param_state.items():
                        if isinstance(v, torch.Tensor):
                            param_state[k] = v.to(device)
                print(f"  [{name}]   ↳ weights + optimizer restored from best-val "
                      f"checkpoint (step {recovery['step']}, val={best_val_loss:.4f})")
            else:
                print(f"  [{name}]   ↳ no best-val checkpoint available — state reset only")
            le_state = model.init_state(batch_size=args.batch_size)
            continue   # no loss to backprop this step

        # --- Thread state (detach for truncated BPTT) ---
        le_state = detach_state(outputs.state)
        last_outputs = outputs

        # --- Compute loss ---
        kl = kl_distill_loss(outputs.logits.float(), teacher_last, T=args.temperature)

        # L_total from model already includes L_pred, L_id, L_reg, L_switch, L_ctrl.
        # Add KL on top (L_distill is 0 inside since we didn't pass distill_loss kwarg).
        total_loss = kl + outputs.losses["L_total"]

        if not torch.isfinite(total_loss):
            print(f"  [{name}] Step {step}: NaN/Inf loss — stopping variant.")
            break

        # --- Gradient accumulation ---
        use_scaler = (amp_dtype == torch.float16)
        if use_scaler:
            if not hasattr(train_variant, "_scaler"):
                train_variant._scaler = torch.amp.GradScaler("cuda")
            train_variant._scaler.scale(total_loss / args.grad_accum).backward()
        else:
            (total_loss / args.grad_accum).backward()
        accum_steps += 1

        if accum_steps == args.grad_accum:
            if use_scaler:
                train_variant._scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if use_scaler:
                train_variant._scaler.step(optimizer)
                train_variant._scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum_steps = 0

        loss_val = total_loss.item()
        ema_loss = loss_val if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * loss_val

        # --- Mid-run checkpoint ---
        if save_dir and args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(save_dir, f"{name}_step{step}.pt")
            save_checkpoint(ckpt_path, name, step, model, le_state,
                            optimizer, scheduler, val_log, ema_loss, chunk_idx)

        # --- Validation eval ---
        if step % args.eval_every == 0:
            val_loss = evaluate(model, teacher, val_chunks, device, args.temperature, amp_dtype)
            val_log[step] = val_loss
            # Keep a rolling best-val checkpoint for VOLUNTARY_END weight recovery.
            # Saved only when val strictly improves so disk writes are infrequent.
            if save_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ckpt_path = os.path.join(save_dir, f"{name}_best_val.pt")
                save_checkpoint(best_val_ckpt_path, name, step, model, le_state,
                                optimizer, scheduler, val_log, ema_loss, chunk_idx)
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{name}] Step {step:5d}/{args.steps} | "
                  f"train_ema={ema_loss:8.3f} | val={val_loss:8.3f} | lr={lr_now:.2e} | "
                  f"vol_end={vol_end_count}")
            if last_outputs is not None:
                diag = _diag_summary(last_outputs)
                if diag:
                    print(f"    [{name}] {diag}")

        elif step % args.log_every == 0 or step == 1:
            lr_now = scheduler.get_last_lr()[0]
            kl_val = kl.item()
            l = outputs.losses
            l_id   = l.get("L_id",      torch.tensor(0.0)).item()
            l_pred = l.get("L_pred",    torch.tensor(0.0)).item() * model.config.lambda_pred
            l_reg  = l.get("L_reg_raw", torch.tensor(0.0)).item()
            print(f"  [{name}] Step {step:5d}/{args.steps} | "
                  f"train_ema={ema_loss:8.3f} | kl={kl_val:.4f} | "
                  f"L_id={l_id:.1f} L_pred={l_pred:.2f} L_reg(raw)={l_reg:.1f} | lr={lr_now:.2e}")

    # --- Final checkpoint ---
    if save_dir and le_state is not None:
        final_path = os.path.join(save_dir, f"{name}_final.pt")
        save_checkpoint(final_path, name, args.steps, model, le_state,
                        optimizer, scheduler, val_log, ema_loss, chunk_idx)

    # --- Final validation ---
    if args.steps not in val_log:
        val_loss = evaluate(model, teacher, val_chunks, device, args.temperature, amp_dtype)
        val_log[args.steps] = val_loss
        print(f"  [{name}] Step {args.steps:5d}/{args.steps} | val={val_loss:.4f}  (final)")

    if vol_end_count > 0:
        print(f"  [{name}] VOLUNTARY_END fired {vol_end_count} times total.")

    del model, optimizer, scheduler
    if hasattr(train_variant, "_scaler"):
        del train_variant._scaler
    torch.cuda.empty_cache()

    return val_log


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Step 4 (LE v15): Life Equation variant experiment")
    p.add_argument("--teacher",     default="Qwen/Qwen2.5-7B",
                   help="Teacher model for KL distillation")
    p.add_argument("--tokenizer",   default="Qwen/Qwen2.5-1.5B",
                   help="Tokenizer to use (LE model vocab must match)")
    p.add_argument("--steps",       type=int,   default=5000)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--seq-len",     type=int,   default=1024)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--warmup-steps", type=int,  default=200,
                   help="LR warmup steps (also controls LE model warmup_steps config)")
    p.add_argument("--grad-accum",  type=int,   default=1,
                   help="Gradient accumulation steps (default 1 — A100 handles batch-size=4 natively)")
    p.add_argument("--eval-every",  type=int,   default=250)
    p.add_argument("--log-every",   type=int,   default=100)
    p.add_argument("--n-val",       type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=4,
                   help="Sequences per forward. A100 40GB: 4 comfortable. A100 80GB: up to 8.")
    p.add_argument("--variants",    default="C,C_no_auto",
                   help="Comma-separated variant names to run")
    p.add_argument("--save-dir",    default="checkpoints")
    p.add_argument("--resume",      default=None,
                   help="Checkpoint path to resume from")
    p.add_argument("--checkpoint-every", type=int, default=2500)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    variant_names = [v.strip() for v in args.variants.split(",")]
    selected = {k: VARIANTS[k] for k in variant_names if k in VARIANTS}
    unknown  = [k for k in variant_names if k not in VARIANTS]
    if unknown:
        print(f"Unknown variants: {unknown}. Available: {list(VARIANTS.keys())}")
    if not selected:
        print("No valid variants selected.")
        return

    amp_dtype  = get_amp_dtype()
    dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
    print(f"AMP dtype: {dtype_name}")

    n_train = args.steps * args.batch_size + 500
    effective_lr = args.lr * args.batch_size * args.grad_accum
    print(f"Effective LR: {args.lr} × batch {args.batch_size} × accum {args.grad_accum} "
          f"→ {effective_lr:.2e}")
    print(f"\nVariants: {list(selected.keys())}  |  Steps: {args.steps}  |  "
          f"Grad accum: {args.grad_accum}  |  Val every: {args.eval_every}")

    # Load teacher once (stays in memory across all variants)
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

    train_chunks, val_chunks = load_data(tokenizer, args.seq_len, n_train, args.n_val)

    # Run selected variants sequentially
    all_val_logs = {}
    for name, cfg in selected.items():
        all_val_logs[name] = train_variant(
            name, cfg, args, teacher, train_chunks, val_chunks,
            device, amp_dtype, effective_lr,
        )

    # ---------------------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS  (validation KL loss — lower is better)")
    print("=" * 80)

    results = {}
    for name, val_log in all_val_logs.items():
        sorted_steps = sorted(val_log.keys())
        mid_step = sorted_steps[len(sorted_steps) // 2]
        results[name] = {
            "label":     VARIANTS[name]["label"],
            "val_start": val_log.get(0, float("nan")),
            "val_mid":   val_log.get(mid_step, float("nan")),
            "val_final": val_log[sorted_steps[-1]],
        }

    baseline = results.get("C", results.get("A", {})).get("val_final", None)
    baseline_name = "C" if "C" in results else "A"

    print(f"{'Var':<12} {'Configuration':<36} {'Val@0':>8} {'Val@mid':>8} "
          f"{'Val@end':>8} {f'vs {baseline_name}':>8}")
    print("-" * 80)
    for name, r in results.items():
        if baseline and math.isfinite(r["val_final"]) and math.isfinite(baseline) and name != baseline_name:
            pct  = (r["val_final"] - baseline) / baseline * 100
            vs_b = f"{pct:+.1f}%"
        elif name == baseline_name:
            vs_b = "(base)"
        else:
            vs_b = "N/A"
        label_s = r["label"][:34]
        print(f"{name:<12} {label_s:<36} {r['val_start']:>8.3f} "
              f"{r['val_mid']:>8.3f} {r['val_final']:>8.3f} {vs_b:>8}")

    print("=" * 80)

    # Go/no-go: capacity question
    if "A" in results and "C" in results:
        a = results["A"]["val_final"]
        c = results["C"]["val_final"]
        if math.isfinite(a) and math.isfinite(c):
            imp = (a - c) / a * 100
            print(f"\nCapacity (C vs A): {imp:+.2f}%")
            if imp >= 10.0:
                print("GO — d_state=64 clearly outperforms d_state=16. Use for 9B Chimera.")
            elif imp > 2.0:
                print("WEAK SIGNAL — d_state=64 better but under 10% threshold.")
            else:
                print("NULL RESULT — d_state has no significant effect at this scale.")

    # Go/no-go: autonomy question
    if "C" in results and "C_no_auto" in results:
        c      = results["C"]["val_final"]
        c_no   = results["C_no_auto"]["val_final"]
        if math.isfinite(c) and math.isfinite(c_no):
            imp = (c_no - c) / c_no * 100
            print(f"\nAutonomy (C vs C_no_auto): {imp:+.2f}%")
            if imp > 2.0:
                print("RESULT: Identity emancipation helps — keep gamma_0=1.0 for 9B Chimera.")
            elif imp < -2.0:
                print("RESULT: Emancipation hurts at this maturity scale — consider gamma_0=0.")
            else:
                print("RESULT: Autonomy is neutral at Amore scale (expected — Z_mat too small).")

    # Go/no-go: emancipation speed question
    if "C" in results and "C_fast" in results:
        c    = results["C"]["val_final"]
        c_f  = results["C_fast"]["val_final"]
        if math.isfinite(c) and math.isfinite(c_f):
            imp = (c_f - c) / c * 100
            print(f"\nEmancipation speed (C_fast vs C): {imp:+.2f}%")
            if imp > 5.0:
                print("RESULT: Fast emancipation destabilises training — use lambda_mature=0.1.")
            else:
                print("RESULT: Emancipation speed is not a critical hyperparameter at this scale.")

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
