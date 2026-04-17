from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict

import torch

from .model import ForwardOutputs


def _tensor_to_serializable(value: torch.Tensor) -> dict:
    cpu = value.detach().cpu()
    flat = cpu.reshape(-1)
    stats = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "values": cpu.tolist(),
    }
    return stats


def _value_to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_to_serializable(value)
    if dataclasses.is_dataclass(value):
        return {k: _value_to_serializable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: _value_to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value_to_serializable(v) for v in value]
    return value


def build_telemetry_snapshot(
    *,
    step: int,
    variant_name: str,
    input_ids: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor | None,
    outputs: ForwardOutputs,
    total_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    grad_norms: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    return {
        "step": step,
        "variant_name": variant_name,
        "action": outputs.action,
        "phase_trace": list(outputs.phase_trace),
        "input_ids": _tensor_to_serializable(input_ids),
        "teacher_logits": _tensor_to_serializable(teacher_logits),
        "student_logits": None if student_logits is None else _tensor_to_serializable(student_logits),
        "loss_total_with_kl": float(total_loss.detach().item()),
        "loss_kl": float(kl_loss.detach().item()),
        "losses": _value_to_serializable(outputs.losses),
        "diagnostics": _value_to_serializable(outputs.diagnostics),
        "state": None if outputs.state is None else _value_to_serializable(vars(outputs.state)),
        "grad_norms": grad_norms or {},
    }


def write_snapshot_json(path: str | Path, snapshot: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def write_snapshot_pt(path: str | Path, snapshot: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, target)


def render_snapshot_summary(snapshot: Dict[str, Any]) -> str:
    losses = snapshot.get("losses", {})
    parts = [
        f"step={snapshot['step']}",
        f"variant={snapshot['variant_name']}",
        f"action={snapshot['action']}",
        f"kl={snapshot['loss_kl']:.6f}",
        f"total={snapshot['loss_total_with_kl']:.6f}",
    ]
    for key in ("L_pred", "L_id", "L_reg", "L_ctrl", "L_transition", "L_sde", "L_total"):
        if key in losses and isinstance(losses[key], dict) and "mean" in losses[key]:
            parts.append(f"{key}={losses[key]['mean']:.6f}")
    return " | ".join(parts)
