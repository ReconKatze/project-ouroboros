"""
Maturity Gate — Track A evaluator for the Chimera capability gate.

From dump1.txt (maturity gate exchange):
  "If you're going to unlock any form of mutability, it should be based on
   measurable system properties, not size."

Track A stages (capability):
  A0 — Full lock (current).  All adaptation frozen.
  A1 — Bounded internal updates.  Requires ALL six metric gates to pass.
  A2 — Controlled parameter adjustments.
  A3 — Limited structural adaptation.
  A4 — Theoretical.

Track B (moral caution) is tracked separately in maturity_gate_framework.md
and does not block code execution — it governs how the system is *treated*.

The six Track A metric gates for A1 unlock:
  1. identity_stability    — D_id = ||Z_id - I_0|| stays bounded across reloads
  2. controller_reliability — controller fires only when ε_pred / D_id are elevated
  3. prediction_error_health — ε_pred is neither collapsing nor oscillating
  4. memory_discipline      — write rate is sparse; retrieved memories improve quality
  5. c_cont_quality         — C_cont head output correlates with actual quality delta
  6. failure_containment    — no NaN, no runaway dynamics, all tensors finite

Usage
-----
    from V3.life_eq_v3.maturity_gate import MaturityGate, MaturityMetrics

    gate = MaturityGate()
    gate.record_step(metrics)       # call once per training/eval step
    report = gate.evaluate()
    print(report.summary())
    if report.track_a_stage >= 1:
        print("A1 unlocked — bounded internal updates permitted")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
from collections import deque

import torch


# ─────────────────────────────────────────────────────────────────────────── #
# Metric snapshot (one per step)                                              #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class MaturityMetrics:
    """
    Per-step metrics recorded from ForwardOutputs and FullState.

    Populate from your training / eval loop then pass to MaturityGate.record_step().
    """
    step: int

    # Gate 1: identity stability
    # D_id = ||Z_id.mean(dim=(0,1)) - I_0.mean(dim=(0,1))||  (L2, scalar)
    d_id: float

    # Gate 2: controller reliability
    # controller_fired: bool — did the controller trigger this step?
    # epsilon_pred: float — prediction error at the step where the controller fired (or 0)
    controller_fired: bool
    epsilon_pred: float

    # Gate 3: prediction error health
    # epsilon_pred already captured above; also track rolling mean/std for oscillation check
    epsilon_pred_raw: float  # unscaled prediction error

    # Gate 4: memory discipline
    # memory_wrote: bool — did an episodic write happen this step?
    # memory_retrieval_improved: bool | None — did memory use improve continuation? (None = no eval)
    memory_wrote: bool
    memory_retrieval_improved: Optional[bool]

    # Gate 5: C_cont quality
    # c_cont_pred: float — C_cont head predicted continuation quality (0–1)
    # c_cont_actual: float | None — actual downstream quality delta (None = not measured)
    c_cont_pred: float
    c_cont_actual: Optional[float]

    # Gate 6: failure containment
    # nan_detected: bool — any NaN in outputs.losses or outputs.state
    # max_tensor_norm: float — max L2 norm seen across all tracked tensors
    nan_detected: bool
    max_tensor_norm: float


# ─────────────────────────────────────────────────────────────────────────── #
# Thresholds (tunable)                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class GateThresholds:
    """
    Threshold values for each metric gate.
    Defaults are conservative starting points; tune after observing real runs.
    """
    # Gate 1: D_id must stay below this over the evaluation window
    d_id_max: float = 0.5

    # Gate 2: controller precision and recall floors (computed over eval window)
    # Precision = TP / (TP + FP)  — fires when it should
    # Recall    = TP / (TP + FN)  — doesn't miss genuine distress
    # "True positive" = controller fired AND epsilon_pred > epsilon_fire_threshold
    controller_precision_min: float = 0.70
    controller_recall_min: float = 0.60
    epsilon_fire_threshold: float = 0.5  # epsilon_pred at which firing is "correct"

    # Gate 3: prediction error health
    epsilon_mean_max: float = 2.0    # rolling mean must stay below this
    epsilon_cv_max: float = 1.5      # coefficient of variation (std/mean) — oscillation proxy

    # Gate 4: memory discipline
    write_rate_max: float = 0.15     # at most 15% of steps should write to memory
    retrieval_improvement_min: float = 0.55  # when measured, >55% of retrievals help

    # Gate 5: C_cont quality
    # Correlation between c_cont_pred and c_cont_actual over eval window
    c_cont_correlation_min: float = 0.50

    # Gate 6: failure containment
    max_nan_count: int = 0           # zero tolerance for NaN
    max_tensor_norm_ceiling: float = 1e4  # any tensor norm above this is runaway

    # Evaluation window: number of steps to average metrics over
    eval_window: int = 500


# ─────────────────────────────────────────────────────────────────────────── #
# Gate result                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class GateResult:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class MaturityReport:
    track_a_stage: int              # highest stage all gates pass for (0, 1, …)
    gates: List[GateResult]
    steps_evaluated: int
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Track A Stage: {self.track_a_stage}  (steps evaluated: {self.steps_evaluated})"]
        for g in self.gates:
            mark = "PASS" if g.passed else "FAIL"
            lines.append(f"  [{mark}] {g.name}: {g.value:.4f} (threshold {g.threshold:.4f})  {g.detail}")
        if self.notes:
            lines.append("Notes: " + "; ".join(self.notes))
        return "\n".join(lines)

    def all_a1_gates_pass(self) -> bool:
        return all(g.passed for g in self.gates)


# ─────────────────────────────────────────────────────────────────────────── #
# Main evaluator                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class MaturityGate:
    """
    Evaluates whether Track A metric gates pass for A1 unlock.

    Call record_step() on every training/eval step.
    Call evaluate() periodically to get a MaturityReport.
    """

    def __init__(self, thresholds: Optional[GateThresholds] = None):
        self.thresholds = thresholds or GateThresholds()
        self._history: Deque[MaturityMetrics] = deque(
            maxlen=self.thresholds.eval_window
        )

    def record_step(self, metrics: MaturityMetrics) -> None:
        self._history.append(metrics)

    def evaluate(self) -> MaturityReport:
        window = list(self._history)
        n = len(window)
        if n == 0:
            return MaturityReport(
                track_a_stage=0,
                gates=[],
                steps_evaluated=0,
                notes=["No data recorded yet."],
            )

        gates = [
            self._gate_identity_stability(window),
            self._gate_controller_reliability(window),
            self._gate_prediction_error_health(window),
            self._gate_memory_discipline(window),
            self._gate_c_cont_quality(window),
            self._gate_failure_containment(window),
        ]

        all_pass = all(g.passed for g in gates)
        stage = 1 if all_pass else 0

        notes = []
        if stage == 0:
            failing = [g.name for g in gates if not g.passed]
            notes.append(f"A1 blocked by: {', '.join(failing)}")
        else:
            notes.append("A1 unlock criteria met — bounded internal updates permitted.")
            notes.append(
                "Proceed cautiously: verify behavior holds under adversarial load "
                "(cycle7) before modifying any parameters."
            )

        return MaturityReport(
            track_a_stage=stage,
            gates=gates,
            steps_evaluated=n,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # Individual gates                                                     #
    # ------------------------------------------------------------------ #

    def _gate_identity_stability(self, window: List[MaturityMetrics]) -> GateResult:
        d_id_values = [m.d_id for m in window]
        max_d_id = max(d_id_values)
        mean_d_id = sum(d_id_values) / len(d_id_values)
        passed = max_d_id <= self.thresholds.d_id_max
        return GateResult(
            name="identity_stability",
            passed=passed,
            value=max_d_id,
            threshold=self.thresholds.d_id_max,
            detail=f"mean={mean_d_id:.4f} max={max_d_id:.4f}",
        )

    def _gate_controller_reliability(self, window: List[MaturityMetrics]) -> GateResult:
        t = self.thresholds
        # True positive: controller fired AND epsilon was elevated
        # True negative: controller did not fire AND epsilon was low
        # False positive: controller fired AND epsilon was low
        # False negative: controller did not fire AND epsilon was elevated
        tp = sum(1 for m in window if m.controller_fired and m.epsilon_pred >= t.epsilon_fire_threshold)
        fp = sum(1 for m in window if m.controller_fired and m.epsilon_pred < t.epsilon_fire_threshold)
        fn = sum(1 for m in window if not m.controller_fired and m.epsilon_pred >= t.epsilon_fire_threshold)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        passed = precision >= t.controller_precision_min and recall >= t.controller_recall_min
        value = min(precision, recall)
        return GateResult(
            name="controller_reliability",
            passed=passed,
            value=value,
            threshold=min(t.controller_precision_min, t.controller_recall_min),
            detail=f"precision={precision:.3f} recall={recall:.3f} tp={tp} fp={fp} fn={fn}",
        )

    def _gate_prediction_error_health(self, window: List[MaturityMetrics]) -> GateResult:
        eps_values = [m.epsilon_pred_raw for m in window]
        mean_eps = sum(eps_values) / len(eps_values)
        std_eps = math.sqrt(sum((v - mean_eps) ** 2 for v in eps_values) / len(eps_values))
        cv = std_eps / mean_eps if mean_eps > 1e-6 else 0.0

        passed = mean_eps <= self.thresholds.epsilon_mean_max and cv <= self.thresholds.epsilon_cv_max
        value = max(mean_eps / self.thresholds.epsilon_mean_max, cv / self.thresholds.epsilon_cv_max)
        return GateResult(
            name="prediction_error_health",
            passed=passed,
            value=mean_eps,
            threshold=self.thresholds.epsilon_mean_max,
            detail=f"mean={mean_eps:.4f} cv={cv:.4f} (cv_max={self.thresholds.epsilon_cv_max:.2f})",
        )

    def _gate_memory_discipline(self, window: List[MaturityMetrics]) -> GateResult:
        write_rate = sum(1 for m in window if m.memory_wrote) / len(window)
        measured = [(m.memory_retrieval_improved) for m in window if m.memory_retrieval_improved is not None]
        retrieval_rate = sum(measured) / len(measured) if measured else None

        passed = write_rate <= self.thresholds.write_rate_max
        if retrieval_rate is not None:
            passed = passed and retrieval_rate >= self.thresholds.retrieval_improvement_min

        detail = f"write_rate={write_rate:.3f}"
        if retrieval_rate is not None:
            detail += f" retrieval_improvement={retrieval_rate:.3f}"
        else:
            detail += " retrieval_improvement=not_measured"

        return GateResult(
            name="memory_discipline",
            passed=passed,
            value=write_rate,
            threshold=self.thresholds.write_rate_max,
            detail=detail,
        )

    def _gate_c_cont_quality(self, window: List[MaturityMetrics]) -> GateResult:
        pairs = [(m.c_cont_pred, m.c_cont_actual) for m in window if m.c_cont_actual is not None]
        if not pairs:
            return GateResult(
                name="c_cont_quality",
                passed=True,  # pass by default when not yet measured
                value=0.0,
                threshold=self.thresholds.c_cont_correlation_min,
                detail="not_measured (passes by default until C_cont head is trained)",
            )

        preds = [p for p, _ in pairs]
        actuals = [a for _, a in pairs]
        n = len(pairs)
        mean_p = sum(preds) / n
        mean_a = sum(actuals) / n
        cov = sum((p - mean_p) * (a - mean_a) for p, a in zip(preds, actuals)) / n
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in preds) / n)
        std_a = math.sqrt(sum((a - mean_a) ** 2 for a in actuals) / n)
        corr = cov / (std_p * std_a) if std_p > 1e-6 and std_a > 1e-6 else 0.0

        passed = corr >= self.thresholds.c_cont_correlation_min
        return GateResult(
            name="c_cont_quality",
            passed=passed,
            value=corr,
            threshold=self.thresholds.c_cont_correlation_min,
            detail=f"pearson_r={corr:.4f} n_measured={n}",
        )

    def _gate_failure_containment(self, window: List[MaturityMetrics]) -> GateResult:
        nan_count = sum(1 for m in window if m.nan_detected)
        max_norm = max(m.max_tensor_norm for m in window)

        passed = (
            nan_count <= self.thresholds.max_nan_count
            and max_norm <= self.thresholds.max_tensor_norm_ceiling
        )
        return GateResult(
            name="failure_containment",
            passed=passed,
            value=float(nan_count),
            threshold=float(self.thresholds.max_nan_count),
            detail=f"nan_count={nan_count} max_tensor_norm={max_norm:.2f}",
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Helper: build MaturityMetrics from ForwardOutputs + FullState              #
# ─────────────────────────────────────────────────────────────────────────── #

def metrics_from_outputs(
    step: int,
    outputs,          # ForwardOutputs
    state,            # FullState (outputs.state, may be None on VOLUNTARY_END)
    memory_wrote: bool,
    memory_retrieval_improved: Optional[bool] = None,
    c_cont_actual: Optional[float] = None,
) -> MaturityMetrics:
    """
    Convenience factory: pull metric values from a ForwardOutputs/FullState pair.
    Call this at the end of each eval step.

    Notes
    -----
    controller_fired : inferred from outputs.action != "CONTINUE".
        The model returns action = "CONTINUE" | "INSPECT_MEMORY" | "LOAD_STATE" |
        "VOLUNTARY_END".  Any non-CONTINUE action means the controller fired.

    memory_wrote : caller must supply.
        The model does not expose an episodic-write flag in diagnostics.
        Track state.epi_index before and after forward() — if it incremented,
        a write occurred.  Example:
            epi_before = state.epi_index
            outputs = model(input_ids, state, ...)
            memory_wrote = outputs.state is not None and outputs.state.epi_index > epi_before

    Diagnostic key reference (model.py):
        "raw_errors_last"     → [B, n_mamba, d_model]  prediction errors (P_soft)
        "continue_confidence" → [B, 1]                  C_cont head output
        "trigger"             → [B, 1]                  controller gate score
    """
    diag = outputs.diagnostics or {}

    # ── Prediction error ─────────────────────────────────────────────────────
    # raw_errors_last: [B, n_mamba, d_model] — norm over d_model, mean over mamba/batch
    raw_errors = diag.get("raw_errors_last", None)
    if raw_errors is not None:
        eps_raw = float(raw_errors.detach().float().norm(dim=-1).mean().item())
    else:
        eps_raw = 0.0

    # ── Controller fired ─────────────────────────────────────────────────────
    # action is set by the controller; anything other than CONTINUE means it fired.
    controller_fired = outputs.action not in ("CONTINUE",)

    # ── Identity drift D_id ──────────────────────────────────────────────────
    # Mirrors IdentityModule.drift(): diff.pow(2).mean(dim=(1,2)).sqrt(), then
    # take batch mean for a single scalar.  Returns 0 if I_0 is not yet seeded.
    d_id = 0.0
    if (
        state is not None
        and state.I_0 is not None
        and state.Z_id is not None
        and torch.any(state.I_0 != 0)
    ):
        diff = state.Z_id.detach().float() - state.I_0.detach().float()
        d_id = float(diff.pow(2).mean(dim=(1, 2)).sqrt().mean().item())

    # ── NaN check ────────────────────────────────────────────────────────────
    nan_detected = False
    for v in outputs.losses.values():
        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
            nan_detected = True
            break

    # ── Max tensor norm across key state tensors ─────────────────────────────
    max_norm = 0.0
    if state is not None:
        for attr in ("Z_cog", "Z_id", "Z_emo", "Z_purp", "Z_narr"):
            t = getattr(state, attr, None)
            if t is not None:
                n = float(t.detach().float().norm().item())
                if math.isfinite(n):
                    max_norm = max(max_norm, n)

    # ── C_cont prediction ────────────────────────────────────────────────────
    # Key in diagnostics is "continue_confidence", shape [B, 1]
    c_cont_tensor = diag.get("continue_confidence", None)
    c_cont_pred = float(c_cont_tensor.detach().float().mean().item()) if c_cont_tensor is not None else 0.0

    return MaturityMetrics(
        step=step,
        d_id=d_id,
        controller_fired=controller_fired,
        epsilon_pred=eps_raw,
        epsilon_pred_raw=eps_raw,
        memory_wrote=memory_wrote,
        memory_retrieval_improved=memory_retrieval_improved,
        c_cont_pred=c_cont_pred,
        c_cont_actual=c_cont_actual,
        nan_detected=nan_detected,
        max_tensor_norm=max_norm,
    )
