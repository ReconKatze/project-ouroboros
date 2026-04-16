"""
EvalRunner — wires metric trackers into the training loop.

Responsibilities
----------------
1. record_train_step()   — called every training step (cheap: scalars only).
2. run_ab_eval()         — called on --ab-eval-every: multi-step rollout to measure
                           controller quality delta and memory retrieval improvement.
3. run_reload_test()     — called on --reload-test-every: clone state, run N steps,
                           record D_id with reload=True to verify convergence.
4. report()              — snapshot all trackers into MaturityReport + EvaluationReport.

A/B design note
---------------
Single-step A/B for controller quality is meaningless: the current forward pass's
logits don't change based on whether the controller fires (logits come from
layer_input + pred_seqs[-1] regardless).  The controller changes STATE for the
NEXT step.  Multi-step rollout is required:
  - Run K val chunks through the model with controller_mode=<live>
  - Run the SAME K chunks from the SAME starting state with controller_mode="disabled"
  - Compare mean KL over K steps → delta = kl_no_ctrl - kl_with_ctrl
    (positive = controller helped; lower KL = better quality)

Same logic applies to memory retrieval: the read only matters if subsequent
generation is better-grounded.
"""

from __future__ import annotations

import sys
import os
from contextlib import contextmanager
from dataclasses import replace as dc_replace
from typing import List, Optional

import torch
import torch.nn.functional as F

# Resolve imports whether runner is called from within the Amore tree or via
# train_distill_v3.py which inserts ROOT and ROOT/V3 onto sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_AMORE = os.path.dirname(os.path.dirname(_HERE))
if _AMORE not in sys.path:
    sys.path.insert(0, _AMORE)
_V3 = os.path.join(_AMORE, "V3")
if _V3 not in sys.path:
    sys.path.insert(0, _V3)

from life_eq_v3.maturity_gate import (
    MaturityGate,
    MaturityReport,
    GateThresholds,
    metrics_from_outputs,
)
from chimera.evaluation.metrics import (
    IdentityDriftTracker,
    ControllerMetrics,
    MemoryMetrics,
    CContMetrics,
    PerceptionCouplingMetrics,
    SelfModelMetrics,
    EvaluationReport,
    build_evaluation_report,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def _kl_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float = 2.0,
) -> torch.Tensor:
    vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    s = student_logits[..., :vocab]
    t = teacher_logits[..., :vocab]
    s = s / (s.std(dim=-1, keepdim=True) + 1e-6)
    t = t / (t.std(dim=-1, keepdim=True) + 1e-6)
    log_p_s = F.log_softmax(s / T, dim=-1)
    p_t = F.softmax(t / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)


def _clone_state(state):
    """Detached tensor clone of a FullState."""
    from life_eq_v3.state import FullState
    return FullState(**{
        k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
        for k, v in vars(state).items()
    })


def _compute_d_id(state) -> float:
    """Replicate IdentityModule.drift() as a plain scalar."""
    if state.I_0 is None or not torch.any(state.I_0 != 0):
        return 0.0
    diff = state.Z_id.detach().float() - state.I_0.detach().float()
    return float(diff.pow(2).mean(dim=(1, 2)).sqrt().mean().item())


@contextmanager
def _temp_profile(model, **overrides):
    """
    Temporarily override fields on model.profile for an eval pass.
    Restores the original profile on exit even if an exception is raised.

    Usage:
        with _temp_profile(model, controller_mode="disabled"):
            out = model(...)
    """
    original = model.profile
    try:
        object.__setattr__(model, "profile", dc_replace(original, **overrides))
        yield
    finally:
        object.__setattr__(model, "profile", original)


# ─────────────────────────────────────────────────────────────────────────── #
# EvalRunner                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class EvalRunner:
    """
    Coordinator for all evaluation metric tracking.

    Parameters
    ----------
    maturity_window : int
        Rolling window size for MaturityGate and all trackers.
    ab_rollout_steps : int
        Number of val chunks to run in each A/B rollout.
    temperature : float
        KL distillation temperature (must match training value).
    """

    def __init__(
        self,
        maturity_window: int = 500,
        ab_rollout_steps: int = 20,
        temperature: float = 2.0,
    ):
        self.maturity_window = maturity_window
        self.ab_rollout_steps = ab_rollout_steps
        self.temperature = temperature

        self.gate = MaturityGate(GateThresholds(eval_window=maturity_window))
        self.drift = IdentityDriftTracker(window=maturity_window)
        self.controller = ControllerMetrics(window=maturity_window)
        self.memory = MemoryMetrics(window=maturity_window)
        self.c_cont = CContMetrics(window=maturity_window // 2)
        self.perception = PerceptionCouplingMetrics(window=maturity_window)
        self.self_model = SelfModelMetrics(window=maturity_window)

    # ------------------------------------------------------------------ #
    # 1. Per-step recording (cheap — called every training step)          #
    # ------------------------------------------------------------------ #

    def record_train_step(
        self,
        step: int,
        outputs,             # ForwardOutputs
        pre_epi_index: int,  # state.epi_index BEFORE the forward pass
    ) -> None:
        """
        Record cheap scalar metrics from one training step.

        Call this immediately after model() returns, before detach_state().
        Pass the epi_index of le_state BEFORE the forward call so we can
        detect whether a memory write occurred.

        Example in train_distill_v3.py::

            pre_epi_index = le_state.epi_index
            outputs = student(input_ids, state=le_state, ...)
            runner.record_train_step(step, outputs, pre_epi_index)
        """
        if outputs.state is None:
            # VOLUNTARY_END — record NaN-clean state
            return

        state = outputs.state
        memory_wrote = state.epi_index > pre_epi_index

        m = metrics_from_outputs(
            step=step,
            outputs=outputs,
            state=state,
            memory_wrote=memory_wrote,
        )

        # MaturityGate (Track A)
        self.gate.record_step(m)

        # IdentityDrift
        self.drift.record(step=step, d_id=m.d_id, reload=False)

        # Controller (quality_delta requires A/B pass — None here)
        self.controller.record(
            fired=m.controller_fired,
            epsilon_pred=m.epsilon_pred,
            d_id=m.d_id,
            c_cont_pred=m.c_cont_pred,
        )

        # Memory (retrieval improvement requires A/B pass)
        self.memory.record_step(wrote=memory_wrote)

        # C_cont (prediction only; actual quality from A/B)
        self.c_cont.record_prediction(c_cont_pred=m.c_cont_pred)

        # SelfDynamicsModel: record L_self and per-dimension predictions if available
        diag = outputs.diagnostics or {}
        l_self = diag.get("sdm_l_self")
        if l_self is not None:
            self.self_model.record(l_self=float(l_self.item() if hasattr(l_self, "item") else l_self))

    # ------------------------------------------------------------------ #
    # 2. A/B rollout evaluation                                            #
    # ------------------------------------------------------------------ #

    def run_ab_eval(
        self,
        model,
        teacher,
        val_chunks: List[torch.Tensor],
        le_state,
        device: torch.device,
        amp_dtype,
    ) -> dict:
        """
        Multi-step A/B rollouts for controller quality and memory retrieval.

        Runs `ab_rollout_steps` val chunks from a cloned copy of le_state under
        four conditions:
          A) Normal (controller + memory active)
          B) Controller disabled (controller_mode="disabled")
          C) Memory read disabled (enable_memory_read=False)
        Then records:
          - CContMetrics A/B: (kl_A, kl_B) per step
          - MemoryMetrics retrieval: (score_A, score_C) per step
          - ControllerMetrics quality_delta: kl_B - kl_A per step

        Returns a summary dict for printing.
        """
        n = min(self.ab_rollout_steps, len(val_chunks))
        if n == 0:
            return {"ab_skipped": "no val_chunks"}

        model.eval()
        summary = {}

        # ── Condition A: normal ──────────────────────────────────────────
        kl_normal = self._run_rollout(
            model, teacher, val_chunks[:n],
            le_state, device, amp_dtype,
            profile_overrides={},
        )

        # ── Condition B: controller disabled ────────────────────────────
        kl_no_ctrl = self._run_rollout(
            model, teacher, val_chunks[:n],
            le_state, device, amp_dtype,
            profile_overrides={"controller_mode": "disabled"},
        )

        # ── Condition C: memory read disabled ───────────────────────────
        mem_read_active = getattr(model.profile, "enable_memory_read", False)
        if mem_read_active:
            kl_no_mem = self._run_rollout(
                model, teacher, val_chunks[:n],
                le_state, device, amp_dtype,
                profile_overrides={"enable_memory_read": False},
            )
        else:
            kl_no_mem = None

        # ── Record controller A/B ────────────────────────────────────────
        for kl_a, kl_b in zip(kl_normal, kl_no_ctrl):
            delta = kl_b - kl_a  # positive = controller helped (lower KL = better)
            self.c_cont.record_ab(with_ctrl=kl_a, without_ctrl=kl_b)
            # Re-record a controller step with quality delta attached
            self.controller.record(
                fired=True,
                epsilon_pred=0.5,   # dummy threshold-level signal for a "should-have-fired" step
                d_id=0.0,
                c_cont_pred=0.0,
                quality_delta=delta,
            )

        mean_kl_normal = sum(kl_normal) / len(kl_normal) if kl_normal else float("nan")
        mean_kl_no_ctrl = sum(kl_no_ctrl) / len(kl_no_ctrl) if kl_no_ctrl else float("nan")
        summary["ctrl_ab_delta"] = mean_kl_no_ctrl - mean_kl_normal
        summary["kl_normal"] = mean_kl_normal
        summary["kl_no_ctrl"] = mean_kl_no_ctrl

        # ── Record memory A/B ────────────────────────────────────────────
        if kl_no_mem is not None:
            for kl_m, kl_nm in zip(kl_normal, kl_no_mem):
                self.memory.record_retrieval(
                    retrieved=True,
                    with_memory_score=-kl_m,    # negate: higher = better
                    without_memory_score=-kl_nm,
                )
            mean_kl_no_mem = sum(kl_no_mem) / len(kl_no_mem)
            summary["mem_ab_delta"] = mean_kl_no_mem - mean_kl_normal
            summary["kl_no_mem"] = mean_kl_no_mem
        else:
            summary["mem_ab_skipped"] = "enable_memory_read=False in this variant"

        model.train()
        return summary

    def _run_rollout(
        self,
        model,
        teacher,
        chunks: List[torch.Tensor],
        le_state,
        device: torch.device,
        amp_dtype,
        profile_overrides: dict,
    ) -> List[float]:
        """Run K val chunks from a clone of le_state; return per-step KL list."""
        state = _clone_state(le_state)
        kl_values = []

        with _temp_profile(model, **profile_overrides):
            with torch.no_grad():
                for chunk in chunks:
                    input_ids = chunk.unsqueeze(0).to(device)
                    teacher_logits = teacher(input_ids=input_ids).logits[:, -1, :].float()
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        out = model(input_ids, state=state, step=0)
                    if out.logits is None or out.state is None:
                        break
                    kl = _kl_distill_loss(out.logits.float(), teacher_logits, T=self.temperature)
                    kl_values.append(float(kl.item()))
                    state = _clone_state(out.state)

        return kl_values

    # ------------------------------------------------------------------ #
    # 3. Reload convergence test                                           #
    # ------------------------------------------------------------------ #

    def run_reload_test(
        self,
        model,
        teacher,
        val_chunks: List[torch.Tensor],
        le_state,
        device: torch.device,
        amp_dtype,
        step: int,
        n_steps: int = 10,
    ) -> dict:
        """
        Reload convergence test.

        Clones the current LE state (simulating a session reload), runs
        n_steps from that clone, and records D_id with reload=True on the
        first step and reload=False on subsequent steps.

        A healthy system: D_id at reload ≈ D_id before reload; D_id does not
        diverge over the post-reload steps.

        Returns a summary dict for printing.
        """
        d_id_before = _compute_d_id(le_state)
        state = _clone_state(le_state)

        # Mark the reload entry
        self.drift.record(step=step, d_id=d_id_before, reload=True)

        n = min(n_steps, len(val_chunks))
        model.eval()
        d_id_trajectory = [d_id_before]

        with torch.no_grad():
            for i, chunk in enumerate(val_chunks[:n]):
                input_ids = chunk.unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(input_ids, state=state, step=step + i)
                if out.state is None:
                    break
                state = _clone_state(out.state)
                d_id = _compute_d_id(state)
                d_id_trajectory.append(d_id)
                self.drift.record(step=step + i + 1, d_id=d_id, reload=False)

        model.train()

        d_id_final = d_id_trajectory[-1] if d_id_trajectory else d_id_before
        converges = d_id_final <= d_id_before * 1.2   # within 20% of pre-reload value
        return {
            "d_id_at_reload": d_id_before,
            "d_id_final": d_id_final,
            "d_id_trajectory": d_id_trajectory,
            "converges": converges,
            "n_steps_run": len(d_id_trajectory) - 1,
        }

    # ------------------------------------------------------------------ #
    # 4. Aggregate report                                                  #
    # ------------------------------------------------------------------ #

    def report(self, step: int):
        """
        Snapshot all trackers.

        Returns
        -------
        maturity_report : MaturityReport
        eval_report : EvaluationReport
        """
        maturity_report = self.gate.evaluate()
        eval_report = build_evaluation_report(
            step=step,
            drift_tracker=self.drift,
            controller_metrics=self.controller,
            memory_metrics=self.memory,
            c_cont_metrics=self.c_cont,
            perception_metrics=self.perception,
            self_model_metrics=self.self_model,
        )
        return maturity_report, eval_report

    def print_report(self, step: int) -> None:
        maturity_report, eval_report = self.report(step)
        print("\n" + "=" * 70)
        print(maturity_report.summary())
        print()
        print(eval_report.summary())
        print("=" * 70 + "\n")
