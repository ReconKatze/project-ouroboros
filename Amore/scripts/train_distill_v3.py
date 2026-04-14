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


def checkpoint_payload(
    *,
    args,
    step: int,
    model,
    optimizer,
    le_state: FullState,
    best_loss: float,
    event_manifest: dict | None = None,
) -> dict:
    return {
        "step": step,
        "variant": args.variant,
        "args": vars(args),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "le_state": {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in vars(le_state).items()},
        "best_loss": best_loss,
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
    p.add_argument("--d-state", type=int, default=64)
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
    student = build_model(args.variant, config).to(device)
    student.train()

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=amp_dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    data_gen = make_token_chunks(tokenizer, args.seq_len)

    def next_chunk():
        nonlocal data_gen
        try:
            return next(data_gen)
        except StopIteration:
            data_gen = make_token_chunks(tokenizer, args.seq_len)
            return next(data_gen)

    le_state = student.init_state(batch_size=1)
    best_loss = float("inf")
    last_completed_step = 0

    def flush_finished_events(finished_events: list[dict], checkpoint_path: str | None) -> None:
        for ctx in finished_events:
            event_dir = forensic.write_bundle(ctx, checkpoint_path=checkpoint_path)
            print(f"forensic bundle saved: {event_dir}")

    for step in range(1, args.steps + 1):
        pre_state = detach_state(le_state)
        input_ids = next_chunk().unsqueeze(0).to(device)
        with torch.no_grad():
            teacher_last = teacher(input_ids=input_ids).logits[:, -1, :].float()

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            outputs = student(input_ids, state=le_state, step=step)

        if outputs.action == "VOLUNTARY_END":
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
                    le_state=pre_state,
                    best_loss=best_loss,
                    event_manifest=manifest,
                ),
            )
            forensic.start_event(manifest, full_snapshot, post_steps_override=0)
            flush_finished_events(forensic.finalize_all(), event_ckpt)
            print(f"step={step} | action=VOLUNTARY_END | resetting state")
            le_state = student.init_state(batch_size=1)
            continue

        le_state = detach_state(outputs.state)
        kl = kl_distill_loss(outputs.logits.float(), teacher_last, T=args.temperature)
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
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
        else:
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        grad_snapshot = gradient_norms(student)

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

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
                    le_state=le_state,
                    best_loss=best_loss,
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
            le_state=le_state,
            best_loss=best_loss,
            event_manifest={"event_type": "final_checkpoint", "trigger_name": "final_checkpoint", "severity": "warning", "step": last_completed_step},
        ),
    )
    flush_finished_events(forensic.finalize_all(), final_path)
    print(f"checkpoint saved: {final_path}")


if __name__ == "__main__":
    main()
