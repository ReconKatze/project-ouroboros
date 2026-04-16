from __future__ import annotations

import dataclasses
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import torch

from .model import ForwardOutputs
from .state import FullState


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value):
        return {k: _cpu_clone(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: _cpu_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(v) for v in value)
    return value


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    return float(value)


def state_to_cpu(state: Optional[FullState]) -> Optional[dict]:
    if state is None:
        return None
    return _cpu_clone(vars(state))


def summarize_outputs(outputs: ForwardOutputs) -> dict:
    return {
        "action": outputs.action,
        "phase_trace": list(outputs.phase_trace),
        "losses": {k: _scalar(v) for k, v in outputs.losses.items()},
        "diagnostics": {
            k: _scalar(v)
            for k, v in outputs.diagnostics.items()
            if isinstance(v, torch.Tensor) and v.numel() > 0
        },
    }


def build_lightweight_replay_entry(
    *,
    step: int,
    input_ids: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],   # accepted but not stored — [B, L, vocab] ~2 GB each
    student_logits: Optional[torch.Tensor],   # accepted but not stored — captured by kl/pred scalars
    pre_state: Optional[FullState],
    post_state: Optional[FullState],
    outputs: ForwardOutputs,
    total_loss: Optional[torch.Tensor],
    kl_loss: Optional[torch.Tensor],
) -> dict:
    return {
        "step": step,
        "input_ids": _cpu_clone(input_ids),
        "pre_state": state_to_cpu(pre_state),
        "post_state": state_to_cpu(post_state),
        "summary": summarize_outputs(outputs),
        "loss_total_with_kl": None if total_loss is None else _scalar(total_loss),
        "loss_kl": None if kl_loss is None else _scalar(kl_loss),
    }


def build_full_forensic_snapshot(
    *,
    step: int,
    variant_name: str,
    input_ids: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],   # accepted but not stored — [B, L, vocab] ~2 GB each
    student_logits: Optional[torch.Tensor],   # accepted but not stored — captured by kl/pred scalars
    pre_state: Optional[FullState],
    post_state: Optional[FullState],
    outputs: ForwardOutputs,
    total_loss: Optional[torch.Tensor],
    kl_loss: Optional[torch.Tensor],
    grad_norms: Optional[Dict[str, float]],
) -> dict:
    # Diagnostics: scalars only. Sequence traces ([B, L, d_model] per layer) and
    # embedded_sequence / raw_errors_last / *_seq_trace tensors are ~25 MB–600 MB each;
    # storing them in every snapshot and replay entry caused 200 GB+ output in 3 events.
    # The loss scalars already capture prediction and KL quality.
    scalar_diagnostics = {
        k: _scalar(v)
        for k, v in outputs.diagnostics.items()
        if isinstance(v, torch.Tensor) and v.numel() > 0
    }
    return {
        "step": step,
        "variant_name": variant_name,
        "timestamp": time.time(),
        "input_ids": _cpu_clone(input_ids),
        "pre_state": state_to_cpu(pre_state),
        "post_state": state_to_cpu(post_state),   # "state" key removed — was a duplicate of post_state
        "action": outputs.action,
        "phase_trace": list(outputs.phase_trace),
        "loss_total_with_kl": None if total_loss is None else _scalar(total_loss),
        "loss_kl": None if kl_loss is None else _scalar(kl_loss),
        "losses": {k: _scalar(v) for k, v in outputs.losses.items()},
        "diagnostics": scalar_diagnostics,
        "grad_norms": dict(grad_norms or {}),
    }


@dataclasses.dataclass
class ForensicConfig:
    pre_event_steps: int = 128
    post_event_steps_warn: int = 32
    post_event_steps_critical: int = 64
    baseline_window: int = 100
    cooldown_steps: int = 50
    loss_spike_z: float = 3.0
    kl_spike_z: float = 3.0
    pred_spike_z: float = 3.0
    reg_spike_z: float = 3.0
    coherence_floor: float = 0.15
    coherence_persist_steps: int = 8
    viability_warn: float = -1.0
    viability_critical: float = -3.0
    viability_persist_warn: int = 8
    viability_persist_critical: int = 4
    controller_action_limit: int = 6
    controller_window: int = 20
    load_state_low_utility_limit: int = 3
    low_utility_threshold: float = 0.0


class RollingStat:
    def __init__(self, maxlen: int):
        self.values: Deque[float] = deque(maxlen=maxlen)

    def update(self, value: float) -> None:
        if math.isfinite(value):
            self.values.append(value)

    def stats(self) -> tuple[float, float]:
        if not self.values:
            return 0.0, 0.0
        vals = list(self.values)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, math.sqrt(var)

    def enough(self, min_items: int = 10) -> bool:
        return len(self.values) >= min_items


class TriggerMonitor:
    def __init__(self, config: ForensicConfig):
        self.config = config
        self.loss_total = RollingStat(config.baseline_window)
        self.loss_kl = RollingStat(config.baseline_window)
        self.loss_pred = RollingStat(config.baseline_window)
        self.loss_reg = RollingStat(config.baseline_window)
        self.drift = RollingStat(config.baseline_window)
        self.actions: Deque[str] = deque(maxlen=config.controller_window)
        self.low_utility_loads: Deque[int] = deque(maxlen=config.controller_window)
        self.coherence_below = 0
        self.viability_warn_below = 0
        self.viability_critical_below = 0
        self.cooldowns: Dict[str, int] = {}

    def _in_cooldown(self, family: str, step: int) -> bool:
        return step < self.cooldowns.get(family, -1)

    def _set_cooldown(self, family: str, step: int) -> None:
        self.cooldowns[family] = step + self.config.cooldown_steps

    def _z_trigger(self, family: str, name: str, value: float, rolling: RollingStat, z_thresh: float, step: int) -> Optional[dict]:
        if self._in_cooldown(family, step) or not rolling.enough():
            return None
        mean, std = rolling.stats()
        if std <= 1e-8:
            return None
        z_score = (value - mean) / std
        if z_score <= z_thresh:
            return None
        self._set_cooldown(family, step)
        return {
            "event_type": "degradation_critical",
            "trigger_name": name,
            "severity": "critical",
            "current": value,
            "baseline_mean": mean,
            "baseline_std": std,
            "z_score": z_score,
        }

    def evaluate(self, *, step: int, outputs: ForwardOutputs, total_loss: float, kl_loss: float) -> List[dict]:
        events: List[dict] = []
        losses = outputs.losses
        diagnostics = outputs.diagnostics
        l_pred = _scalar(losses.get("L_pred", 0.0))
        l_reg = _scalar(losses.get("L_reg", 0.0))
        coherence = _scalar(diagnostics.get("coherence", 0.0))
        v_self = _scalar(diagnostics.get("v_self", 0.0))
        utility = _scalar(diagnostics.get("utility", 0.0))
        d_id = _scalar(losses.get("L_id", 0.0))

        for tensor_name in ("coherence", "v_self", "trigger", "controller_input"):
            tensor = diagnostics.get(tensor_name)
            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                events.append({
                    "event_type": "nonfinite_failure",
                    "trigger_name": f"nonfinite_{tensor_name}",
                    "severity": "fatal",
                })

        if not math.isfinite(total_loss) or not math.isfinite(kl_loss):
            events.append({
                "event_type": "nonfinite_failure",
                "trigger_name": "nonfinite_loss",
                "severity": "fatal",
            })

        self.actions.append(outputs.action)
        self.low_utility_loads.append(
            1 if outputs.action == "LOAD_STATE" and utility <= self.config.low_utility_threshold else 0
        )

        if coherence < self.config.coherence_floor:
            self.coherence_below += 1
        else:
            self.coherence_below = 0
        if v_self < self.config.viability_warn:
            self.viability_warn_below += 1
        else:
            self.viability_warn_below = 0
        if v_self < self.config.viability_critical:
            self.viability_critical_below += 1
        else:
            self.viability_critical_below = 0

        if (
            self.coherence_below >= self.config.coherence_persist_steps
            and not self._in_cooldown("coherence", step)
        ):
            self._set_cooldown("coherence", step)
            events.append({
                "event_type": "degradation_warning",
                "trigger_name": "coherence_collapse",
                "severity": "warning",
                "current": coherence,
                "persist_steps": self.coherence_below,
                "threshold": self.config.coherence_floor,
            })

        if (
            self.viability_critical_below >= self.config.viability_persist_critical
            and not self._in_cooldown("viability", step)
        ):
            self._set_cooldown("viability", step)
            events.append({
                "event_type": "degradation_critical",
                "trigger_name": "viability_collapse_critical",
                "severity": "critical",
                "current": v_self,
                "persist_steps": self.viability_critical_below,
                "threshold": self.config.viability_critical,
            })
        elif (
            self.viability_warn_below >= self.config.viability_persist_warn
            and not self._in_cooldown("viability", step)
        ):
            self._set_cooldown("viability", step)
            events.append({
                "event_type": "degradation_warning",
                "trigger_name": "viability_collapse_warning",
                "severity": "warning",
                "current": v_self,
                "persist_steps": self.viability_warn_below,
                "threshold": self.config.viability_warn,
            })

        non_continue = sum(1 for action in self.actions if action != "CONTINUE")
        if (
            non_continue > self.config.controller_action_limit
            and not self._in_cooldown("controller", step)
        ):
            self._set_cooldown("controller", step)
            events.append({
                "event_type": "controller_instability",
                "trigger_name": "controller_thrashing",
                "severity": "warning",
                "non_continue_actions": non_continue,
                "window": self.config.controller_window,
            })
        if (
            sum(self.low_utility_loads) >= self.config.load_state_low_utility_limit
            and not self._in_cooldown("controller", step)
        ):
            self._set_cooldown("controller", step)
            events.append({
                "event_type": "controller_instability",
                "trigger_name": "low_utility_load_state_loop",
                "severity": "warning",
                "count": sum(self.low_utility_loads),
                "window": self.config.controller_window,
            })

        for maybe in (
            self._z_trigger("loss_total", "loss_spike", total_loss, self.loss_total, self.config.loss_spike_z, step),
            self._z_trigger("kl", "kl_spike", kl_loss, self.loss_kl, self.config.kl_spike_z, step),
            self._z_trigger("pred", "prediction_spike", l_pred, self.loss_pred, self.config.pred_spike_z, step),
            self._z_trigger("reg", "regulation_spike", l_reg, self.loss_reg, self.config.reg_spike_z, step),
            self._z_trigger("drift", "identity_drift_spike", d_id, self.drift, self.config.pred_spike_z, step),
        ):
            if maybe is not None:
                events.append(maybe)

        self.loss_total.update(total_loss)
        self.loss_kl.update(kl_loss)
        self.loss_pred.update(l_pred)
        self.loss_reg.update(l_reg)
        self.drift.update(d_id)
        return events


class ForensicEventManager:
    def __init__(self, root_dir: str | Path, config: Optional[ForensicConfig] = None):
        self.root_dir = Path(root_dir)
        self.config = config or ForensicConfig()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = TriggerMonitor(self.config)
        self.replay_buffer: Deque[dict] = deque(maxlen=self.config.pre_event_steps)
        self.active_events: List[dict] = []
        self.event_counter = 0

    def append_replay_entry(self, entry: dict) -> None:
        self.replay_buffer.append(entry)

    def evaluate_triggers(self, *, step: int, outputs: ForwardOutputs, total_loss: float, kl_loss: float) -> List[dict]:
        return self.monitor.evaluate(step=step, outputs=outputs, total_loss=total_loss, kl_loss=kl_loss)

    def start_event(self, manifest: dict, full_snapshot: dict, post_steps_override: Optional[int] = None) -> None:
        self.event_counter += 1
        severity = manifest.get("severity", "warning")
        post_steps = post_steps_override
        if post_steps is None:
            post_steps = (
                self.config.post_event_steps_critical
                if severity in {"critical", "fatal"}
                else self.config.post_event_steps_warn
            )
        ctx = {
            "event_id": f"event_{self.event_counter:06d}",
            "manifest": {
                **manifest,
                "timestamp": time.time(),
                "pre_event_steps": self.config.pre_event_steps,
                "post_event_steps": post_steps,
            },
            "replay_buffer": list(self.replay_buffer),
            "full_snapshots": [full_snapshot],
            "remaining_post_steps": post_steps,
        }
        self.active_events.append(ctx)

    def append_post_step(self, full_snapshot: dict) -> List[dict]:
        finished: List[dict] = []
        still_active: List[dict] = []
        for ctx in self.active_events:
            if ctx["remaining_post_steps"] > 0 and full_snapshot["step"] != ctx["manifest"].get("step"):
                ctx["full_snapshots"].append(full_snapshot)
                ctx["remaining_post_steps"] -= 1
            if ctx["remaining_post_steps"] <= 0:
                finished.append(ctx)
            else:
                still_active.append(ctx)
        self.active_events = still_active
        return finished

    def finalize_all(self) -> List[dict]:
        finished = self.active_events
        self.active_events = []
        return finished

    def write_bundle(self, ctx: dict, checkpoint_path: Optional[str] = None) -> Path:
        event_dir = self.root_dir / ctx["event_id"]
        event_dir.mkdir(parents=True, exist_ok=True)
        manifest = dict(ctx["manifest"])
        manifest["checkpoint_path"] = checkpoint_path
        manifest_path = event_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        torch.save(
            {
                "manifest": manifest,
                "replay_buffer": ctx["replay_buffer"],
                "full_snapshots": ctx["full_snapshots"],
            },
            event_dir / "bundle.pt",
        )
        return event_dir
