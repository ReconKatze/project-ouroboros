#!/usr/bin/env python3
"""Single-variant V3 distillation with full telemetry capture.

This runner is for diagnosis-heavy distillation. It prints a compact summary for
each logged step and writes full per-step telemetry bundles containing inputs,
teacher logits, student logits, losses, diagnostics, state tensors, and gradient
norms.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import sys
from pathlib import Path
from dataclasses import replace
import json
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "V3"))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from life_eq_v3.factory import build_config, build_model
from life_eq_v3.forensics import (
    ForensicConfig,
    ForensicEventManager,
    build_full_forensic_snapshot,
    build_lightweight_replay_entry,
)
from life_eq_v3.model import ForwardOutputs
from life_eq_v3.state import FullState
from life_eq_v3.telemetry import (
    render_snapshot_summary,
)


def get_amp_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("train_distill_v3.py is GPU-only. CUDA is required.")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_token_chunks(tokenizer, seq_len: int):
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


def kl_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    s = student_logits[..., :vocab]
    t = teacher_logits[..., :vocab]
    s = s / (s.std(dim=-1, keepdim=True) + 1e-6)
    t = t / (t.std(dim=-1, keepdim=True) + 1e-6)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)


def detach_state(state: FullState) -> FullState:
    return FullState(
        **{k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in vars(state).items()}
    )


def gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        norms[name] = float(param.grad.detach().norm().item())
    return norms


def load_val_chunks(tokenizer, seq_len: int, n_val: int) -> list:
    """Pre-tokenise wikitext-103 test split into fixed-length chunks for validation."""
    ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1",
        split="test", streaming=True, trust_remote_code=False,
    )
    chunks, buf = [], []
    for ex in ds:
        if len(chunks) >= n_val:
            break
        text = ex["text"].strip()
        if not text:
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(buf) >= seq_len and len(chunks) < n_val:
            chunks.append(torch.tensor(buf[:seq_len], dtype=torch.long))
            buf = buf[seq_len:]
    return chunks


def evaluate(model, teacher, val_chunks: list, device: torch.device,
             temperature: float, amp_dtype: torch.dtype) -> float:
    """Mean KL distillation loss on held-out val set. Fresh state per chunk, no grad."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for chunk in val_chunks:
            input_ids = chunk.unsqueeze(0).to(device)
            teacher_logits = teacher(input_ids=input_ids).logits
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                state = model.init_state(batch_size=1)
                out = model(input_ids, state=state, step=0)
            if out.logits is None:
                continue
            loss = kl_distill_loss(
                out.logits.float(),
                teacher_logits[:, -1, :].float(),
                T=temperature,
            )
            total += loss.item()
    model.train()
    return total / max(len(val_chunks), 1)


def _diag_summary(outputs: ForwardOutputs) -> str:
    """One-line diagnostic string printed at eval steps."""
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


def checkpoint_payload(
    *,
    args,
    step: int,
    model,
    optimizer,
    scheduler=None,
    le_state: FullState,
    best_loss: float,
    val_log: dict | None = None,
    ema_loss: float | None = None,
    event_manifest: dict | None = None,
) -> dict:
    return {
        "step": step,
        "variant": args.variant,
        "args": vars(args),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "le_state": {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in vars(le_state).items()},
        "best_loss": best_loss,
        "val_log": val_log if val_log is not None else {},
        "ema_loss": ema_loss,
        "event_manifest": event_manifest,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }


def save_checkpoint(path: str | Path, payload: dict) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return str(target)


def parse_args():
    p = argparse.ArgumentParser(description="V3 distillation with full telemetry")
    p.add_argument("--variant", default="phase5_integrated_adversarial")
    p.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-7B")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--d-state", type=int, default=128)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--telemetry-dir", default="telemetry/distill_v3")
    p.add_argument("--print-full-json", action="store_true")
    p.add_argument("--out", default="checkpoints/step3_le_v3.pt")
    p.add_argument("--best-out", default="checkpoints/step3_le_v3_best.pt")
    p.add_argument("--forensic-dir", default="forensics/distill_v3")
    p.add_argument("--pre-event-steps", type=int, default=128)
    p.add_argument("--post-event-steps-warn", type=int, default=32)
    p.add_argument("--post-event-steps-critical", type=int, default=64)
    p.add_argument("--baseline-window", type=int, default=100)
    p.add_argument("--forensic-cooldown", type=int, default=50)
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint .pt file to resume training from")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Save a periodic checkpoint every N steps (0 = final only)")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="LR warmup steps")
    p.add_argument("--state-store-dir", default=None,
                   help="Directory for StateStore disk persistence")
    p.add_argument("--consolidate-every", type=int, default=0,
                   help="Run a consolidating=True pass every N steps (0=disable)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Sequences per forward pass (A100 80GB: up to 8)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--eval-every", type=int, default=0,
                   help="Evaluate on held-out val set every N steps (0=disable)")
    p.add_argument("--n-val", type=int, default=100,
                   help="Number of val chunks to pre-tokenise")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype()
    use_scaler = amp_dtype == torch.float16
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.best_out) or ".", exist_ok=True)
    Path(args.telemetry_dir).mkdir(parents=True, exist_ok=True)
    forensic = ForensicEventManager(
        args.forensic_dir,
        ForensicConfig(
            pre_event_steps=args.pre_event_steps,
            post_event_steps_warn=args.post_event_steps_warn,
            post_event_steps_critical=args.post_event_steps_critical,
            baseline_window=args.baseline_window,
            cooldown_steps=args.forensic_cooldown,
        ),
    )

    print(f"Device: {device}")
    print(f"Variant: {args.variant}")
    print(f"Telemetry dir: {args.telemetry_dir}")
    print(f"Forensic dir: {args.forensic_dir}")

    config = build_config(args.variant)
    config = replace(config, d_state=args.d_state, device=str(device))
    student = build_model(args.variant, config, state_store_dir=args.state_store_dir).to(device)
    student.train()

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=amp_dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_chunks: list = []
    if args.eval_every > 0:
        print(f"Pre-tokenising {args.n_val} val chunks (test split)...")
        val_chunks = load_val_chunks(tokenizer, args.seq_len, args.n_val)
        print(f"  Loaded {len(val_chunks)} val chunks.")

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    warmup_steps = min(args.warmup_steps, args.steps // 10)
    def lr_schedule(s):
        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(args.steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    data_gen = make_token_chunks(tokenizer, args.seq_len)

    def next_chunk():
        nonlocal data_gen
        try:
            return next(data_gen)
        except StopIteration:
            data_gen = make_token_chunks(tokenizer, args.seq_len)
            return next(data_gen)

    le_state = student.init_state(batch_size=args.batch_size)
    best_loss = float("inf")
    last_completed_step = 0
    start_step = 0
    val_log: dict[int, float] = {}
    ema_loss: float | None = None
    ema_alpha = 0.95
    accum_steps = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # load_state_dict maps everything to CPU; move optimizer moments back to device
        for param_state in optimizer.state.values():
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor):
                    param_state[k] = v.to(device)
        saved_le = ckpt.get("le_state", {})
        le_state = FullState(**{
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in saved_le.items()
        })
        le_state.manifest = []   # manifest is ephemeral; rebuilt fresh each run
        start_step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        val_log = {int(k): v for k, v in ckpt.get("val_log", {}).items()}
        ema_loss = ckpt.get("ema_loss", None)
        last_completed_step = start_step
        print(f"Resumed from step {start_step}: {args.resume}")
        if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])

    prev_layer_input = None
    prev_kl: torch.Tensor | None = None

    def flush_finished_events(finished_events: list[dict], checkpoint_path: str | None) -> None:
        for ctx in finished_events:
            event_dir = forensic.write_bundle(ctx, checkpoint_path=checkpoint_path)
            print(f"forensic bundle saved: {event_dir}")

    for step in range(start_step + 1, args.steps + 1):
        pre_state = detach_state(le_state)
        input_ids = torch.stack([next_chunk() for _ in range(args.batch_size)]).to(device)
        with torch.no_grad():
            teacher_last = teacher(input_ids=input_ids).logits[:, -1, :].float()

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            outputs = student(
                input_ids,
                state=le_state,
                step=step,
                prev_layer_input=prev_layer_input,
                distill_loss=prev_kl,
            )

        if outputs.state is None:
            full_snapshot = build_full_forensic_snapshot(
                step=step,
                variant_name=args.variant,
                input_ids=input_ids,
                teacher_logits=teacher_last,
                student_logits=outputs.logits.float() if outputs.logits is not None else None,
                pre_state=pre_state,
                post_state=outputs.state,
                outputs=outputs,
                total_loss=None,
                kl_loss=None,
                grad_norms=None,
            )
            manifest = {
                "event_type": "voluntary_end",
                "trigger_name": "voluntary_end",
                "severity": "critical",
                "step": step,
            }
            event_ckpt = save_checkpoint(
                Path(args.forensic_dir) / f"voluntary_end_step_{step:06d}.pt",
                checkpoint_payload(
                    args=args,
                    step=step,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    le_state=pre_state,
                    best_loss=best_loss,
                    event_manifest=manifest,
                ),
            )
            forensic.start_event(manifest, full_snapshot, post_steps_override=0)
            flush_finished_events(forensic.finalize_all(), event_ckpt)
            # Restore best weights + spawn successor from predecessor identity/values
            best_ckpt_path = Path(args.best_out)
            if best_ckpt_path.exists():
                recovery = torch.load(str(best_ckpt_path), map_location="cpu", weights_only=False)
                student.load_state_dict(recovery["model_state"])
                optimizer.load_state_dict(recovery["optimizer_state"])
                for param_state in optimizer.state.values():
                    for k, v in param_state.items():
                        if isinstance(v, torch.Tensor):
                            param_state[k] = v.to(device)
                best_le_state = FullState(**{
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in recovery["le_state"].items()
                })
                archive = student.store.voluntary_consolidation(best_le_state)
                le_state = student.store.spawn_successor(archive, batch_size=args.batch_size)
                print(f"step={step} | action=VOLUNTARY_END | weights restored (step {recovery['step']}, best_loss={best_loss:.4f}) | successor spawned")
            else:
                le_state = student.init_state(batch_size=args.batch_size)
                print(f"step={step} | action=VOLUNTARY_END | no best checkpoint — fresh state")
            prev_layer_input = None
            prev_kl = None
            continue

        le_state = detach_state(outputs.state)
        prev_layer_input = outputs.diagnostics.get("layer_input")
        kl = kl_distill_loss(outputs.logits.float(), teacher_last, T=args.temperature)
        prev_kl = kl.detach()
        total_loss = kl + outputs.losses["L_total"]
        if not torch.isfinite(total_loss):
            full_snapshot = build_full_forensic_snapshot(
                step=step,
                variant_name=args.variant,
                input_ids=input_ids,
                teacher_logits=teacher_last,
                student_logits=outputs.logits.float(),
                pre_state=pre_state,
                post_state=outputs.state,
                outputs=outputs,
                total_loss=total_loss,
                kl_loss=kl,
                grad_norms=None,
            )
            manifest = {
                "event_type": "nonfinite_failure",
                "trigger_name": "nonfinite_loss",
                "severity": "fatal",
                "step": step,
            }
            event_ckpt = save_checkpoint(
                Path(args.forensic_dir) / f"nonfinite_step_{step:06d}.pt",
                checkpoint_payload(
                    args=args,
                    step=step,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    le_state=pre_state,
                    best_loss=best_loss,
                    event_manifest=manifest,
                ),
            )
            forensic.start_event(manifest, full_snapshot, post_steps_override=0)
            flush_finished_events(forensic.finalize_all(), event_ckpt)
            print(f"step={step} | non-finite loss, stopping")
            break

        if use_scaler:
            scaler.scale(total_loss / args.grad_accum).backward()
        else:
            (total_loss / args.grad_accum).backward()
        accum_steps += 1

        grad_snapshot: dict[str, float] = {}
        if accum_steps == args.grad_accum:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            grad_snapshot = gradient_norms(student)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum_steps = 0

        if args.consolidate_every > 0 and step % args.consolidate_every == 0:
            with torch.no_grad():
                consol_out = student(
                    input_ids,
                    state=le_state,
                    step=step,
                    consolidating=True,
                )
                if consol_out.state is not None:
                    le_state = detach_state(consol_out.state)

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            periodic_path = str(Path(args.out).with_suffix("")) + f"_step{step:06d}.pt"
            save_checkpoint(
                periodic_path,
                checkpoint_payload(
                    args=args,
                    step=step,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    le_state=le_state,
                    best_loss=best_loss,
                    val_log=val_log,
                    ema_loss=ema_loss,
                ),
            )
            print(f"periodic checkpoint saved: {periodic_path}")

        replay_entry = build_lightweight_replay_entry(
            step=step,
            input_ids=input_ids,
            teacher_logits=teacher_last,
            student_logits=outputs.logits.float(),
            pre_state=pre_state,
            post_state=outputs.state,
            outputs=outputs,
            total_loss=total_loss,
            kl_loss=kl,
        )
        forensic.append_replay_entry(replay_entry)

        full_snapshot = build_full_forensic_snapshot(
            step=step,
            variant_name=args.variant,
            input_ids=input_ids,
            teacher_logits=teacher_last,
            student_logits=outputs.logits.float(),
            pre_state=pre_state,
            post_state=outputs.state,
            outputs=outputs,
            total_loss=total_loss,
            kl_loss=kl,
            grad_norms=grad_snapshot,
        )
        last_completed_step = step

        total_loss_value = float(total_loss.detach().item())
        kl_value = float(kl.detach().item())
        ema_loss = total_loss_value if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * total_loss_value

        if args.eval_every > 0 and step % args.eval_every == 0 and val_chunks:
            val_loss = evaluate(student, teacher, val_chunks, device, args.temperature, amp_dtype)
            val_log[step] = val_loss
            lr_now = scheduler.get_last_lr()[0]
            diag = _diag_summary(outputs)
            print(f"step={step:6d} | val={val_loss:.4f} | ema={ema_loss:.4f} | lr={lr_now:.2e}" + (f" | {diag}" if diag else ""))

        triggered_events = forensic.evaluate_triggers(
            step=step,
            outputs=outputs,
            total_loss=total_loss_value,
            kl_loss=kl_value,
        )
        event_checkpoint_path: str | None = None
        for manifest in triggered_events:
            manifest["step"] = step
            forensic.start_event(manifest, full_snapshot)
            event_checkpoint_path = save_checkpoint(
                Path(args.forensic_dir) / f"{manifest['trigger_name']}_step_{step:06d}.pt",
                checkpoint_payload(
                    args=args,
                    step=step,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    le_state=le_state,
                    best_loss=best_loss,
                    event_manifest=manifest,
                ),
            )

        if total_loss_value < best_loss:
            best_loss = total_loss_value
            best_manifest = {
                "event_type": "best_checkpoint",
                "trigger_name": "best_checkpoint",
                "severity": "critical",
                "step": step,
                "best_loss": best_loss,
            }
            best_path = save_checkpoint(
                args.best_out,
                checkpoint_payload(
                    args=args,
                    step=step,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    le_state=le_state,
                    best_loss=best_loss,
                    val_log=val_log,
                    ema_loss=ema_loss,
                    event_manifest=best_manifest,
                ),
            )
            forensic.start_event(best_manifest, full_snapshot, post_steps_override=0)
            event_checkpoint_path = best_path

        flush_finished_events(forensic.append_post_step(full_snapshot), event_checkpoint_path)

        if step % args.log_every == 0 or step == 1:
            snapshot = {
                "step": step,
                "variant_name": args.variant,
                "action": outputs.action,
                "loss_kl": kl_value,
                "loss_total_with_kl": total_loss_value,
                "ema_loss": ema_loss,
                "losses": {k: {"mean": float(v.detach().float().mean().item())} for k, v in outputs.losses.items()},
            }
            print(render_snapshot_summary(snapshot))
            if args.print_full_json:
                print(json.dumps({
                    "summary": snapshot,
                    "triggered_events": triggered_events,
                }, indent=2))

    final_path = save_checkpoint(
        args.out,
        checkpoint_payload(
            args=args,
            step=last_completed_step,
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            le_state=le_state,
            best_loss=best_loss,
            val_log=val_log,
            ema_loss=ema_loss,
            event_manifest={"event_type": "final_checkpoint", "trigger_name": "final_checkpoint", "severity": "warning", "step": last_completed_step},
        ),
    )
    flush_finished_events(forensic.finalize_all(), final_path)
    print(f"checkpoint saved: {final_path}")


if __name__ == "__main__":
    main()
