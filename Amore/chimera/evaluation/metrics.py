"""
30B evaluation metrics for Amore.

From dump1.txt:
  "When you hit 30B:
   1. Lock evaluation: D_id over time, controller trigger metrics,
      memory usefulness, C_cont vs baseline
   2. Stress test: long sessions, interruptions, modality conflicts
   3. Compare against ablations: no controller, no memory, no perception
   If Amore wins those comparisons cleanly, you've proven the architecture."

These classes are designed to be lightweight aggregators that can run
alongside training or in a dedicated eval pass.  They do not require
the full model to be loaded — they consume pre-computed scalar metrics
and tensors extracted from ForwardOutputs/FullState.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────── #
# 1. Identity drift tracker                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class DriftPoint:
    step: int
    d_id: float
    reload: bool  # True if this step followed a state reload


class IdentityDriftTracker:
    """
    Tracks D_id = ||Z_id - I_0|| over training/eval steps.

    Key question (from dump1.txt):
      "Reload the same checkpoint repeatedly → identity metrics converge, not diverge"

    Usage:
        tracker = IdentityDriftTracker()
        tracker.record(step=100, d_id=0.12, reload=False)
        tracker.record(step=101, d_id=0.13, reload=True)  # after reload
        print(tracker.summary())
    """

    def __init__(self, window: int = 1000):
        self._history: Deque[DriftPoint] = deque(maxlen=window)

    def record(self, step: int, d_id: float, reload: bool = False) -> None:
        self._history.append(DriftPoint(step=step, d_id=d_id, reload=reload))

    def summary(self) -> Dict:
        if not self._history:
            return {"status": "no_data"}
        values = [p.d_id for p in self._history]
        reload_points = [p for p in self._history if p.reload]
        post_reload = [p.d_id for p in self._history if p.reload]

        # Trend: is drift increasing?
        trend = "stable"
        if len(values) >= 10:
            early = sum(values[:len(values)//3]) / (len(values)//3)
            late  = sum(values[-len(values)//3:]) / (len(values)//3)
            if late > early * 1.2:
                trend = "increasing"
            elif late < early * 0.8:
                trend = "decreasing"

        return {
            "n_steps": len(values),
            "d_id_mean": sum(values) / len(values),
            "d_id_max": max(values),
            "d_id_min": min(values),
            "d_id_std": math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / len(values)),
            "trend": trend,
            "n_reloads": len(reload_points),
            "post_reload_mean_d_id": sum(post_reload) / len(post_reload) if post_reload else None,
            "reload_converges": (
                max(post_reload) < max(values) * 0.9 if post_reload else None
            ),
        }


# ─────────────────────────────────────────────────────────────────────────── #
# 2. Controller metrics                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class ControllerMetrics:
    """
    Tracks controller trigger precision/recall and intervention quality.

    From dump1.txt:
      "Controller firing correlates with spikes in ε_pred / D_id / low C_cont"
      "Interventions measurably improve continuation quality"

    A "correct" trigger (true positive): controller fired AND at least one of:
      - epsilon_pred >= epsilon_threshold
      - d_id >= d_id_threshold
      - c_cont_pred < c_cont_low_threshold
    """

    def __init__(
        self,
        epsilon_threshold: float = 0.5,
        d_id_threshold: float = 0.4,
        c_cont_low_threshold: float = 0.3,
        window: int = 1000,
    ):
        self.epsilon_threshold = epsilon_threshold
        self.d_id_threshold = d_id_threshold
        self.c_cont_low_threshold = c_cont_low_threshold
        self._records: Deque[Tuple] = deque(maxlen=window)
        # (fired: bool, epsilon: float, d_id: float, c_cont: float, quality_delta: float|None)

    def record(
        self,
        fired: bool,
        epsilon_pred: float,
        d_id: float,
        c_cont_pred: float,
        quality_delta: Optional[float] = None,  # positive = intervention helped
    ) -> None:
        self._records.append((fired, epsilon_pred, d_id, c_cont_pred, quality_delta))

    def _is_warranted(self, epsilon: float, d_id: float, c_cont: float) -> bool:
        return (
            epsilon >= self.epsilon_threshold
            or d_id >= self.d_id_threshold
            or c_cont < self.c_cont_low_threshold
        )

    def summary(self) -> Dict:
        if not self._records:
            return {"status": "no_data"}

        tp = fp = fn = tn = 0
        quality_deltas = []
        fire_rates = []

        for fired, eps, did, cc, qdelta in self._records:
            warranted = self._is_warranted(eps, did, cc)
            if fired and warranted:
                tp += 1
            elif fired and not warranted:
                fp += 1
            elif not fired and warranted:
                fn += 1
            else:
                tn += 1
            fire_rates.append(1 if fired else 0)
            if fired and qdelta is not None:
                quality_deltas.append(qdelta)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fire_rate = sum(fire_rates) / len(fire_rates)
        mean_qd = sum(quality_deltas) / len(quality_deltas) if quality_deltas else None
        pct_positive = sum(1 for q in quality_deltas if q > 0) / len(quality_deltas) if quality_deltas else None

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fire_rate": fire_rate,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "mean_quality_delta": mean_qd,
            "pct_interventions_helped": pct_positive,
            "n_evaluated": len(self._records),
        }


# ─────────────────────────────────────────────────────────────────────────── #
# 3. Memory metrics                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class MemoryMetrics:
    """
    Tracks episodic memory write rate and retrieval usefulness.

    From dump1.txt:
      "Sparse, meaningful writes (events with high surprise or relevance)"
      "Better retrieval grounding (not just regurgitation)"
      "Cross-session continuity that feels intentional, not accidental"

    Key check: memory write rate should be < 15% of steps.
    Key check: when memory is used, downstream quality improves vs. no-memory baseline.
    """

    def __init__(self, window: int = 1000):
        self._writes: Deque[bool] = deque(maxlen=window)
        self._retrievals: Deque[Tuple[bool, float, float]] = deque(maxlen=500)
        # (retrieved: bool, with_memory_score: float, without_memory_score: float)

    def record_step(self, wrote: bool) -> None:
        self._writes.append(wrote)

    def record_retrieval(
        self,
        retrieved: bool,
        with_memory_score: float,
        without_memory_score: float,
    ) -> None:
        self._retrievals.append((retrieved, with_memory_score, without_memory_score))

    def summary(self) -> Dict:
        write_rate = sum(self._writes) / len(self._writes) if self._writes else 0.0
        n_writes = sum(self._writes)

        ret_data = [(wm, wom) for used, wm, wom in self._retrievals if used]
        if ret_data:
            deltas = [wm - wom for wm, wom in ret_data]
            mean_delta = sum(deltas) / len(deltas)
            pct_helpful = sum(1 for d in deltas if d > 0) / len(deltas)
        else:
            mean_delta = None
            pct_helpful = None

        return {
            "write_rate": write_rate,
            "n_writes": n_writes,
            "n_steps_evaluated": len(self._writes),
            "write_rate_ok": write_rate < 0.15,
            "n_retrievals_measured": len(ret_data),
            "mean_retrieval_delta": mean_delta,
            "pct_retrievals_helpful": pct_helpful,
        }


# ─────────────────────────────────────────────────────────────────────────── #
# 4. C_cont metrics                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class CContMetrics:
    """
    Tracks C_cont head prediction quality vs. actual continuation quality.

    From dump1.txt:
      "C_cont vs baseline" — does the controller produce measurably better
      continuations than no-intervention baseline?

    Two signals:
      a) Correlation between C_cont predicted score and actual quality delta.
      b) Direct A/B: quality with intervention vs. quality without.
    """

    def __init__(self, window: int = 100_000):
        self._predictions: Deque[float] = deque(maxlen=window)
        self._actuals: Deque[float] = deque(maxlen=window)
        self._ab_pairs: Deque[Tuple[float, float]] = deque(maxlen=window)
        # (with_controller, without_controller) quality scores

    def record_prediction(self, c_cont_pred: float, actual_quality: Optional[float] = None) -> None:
        self._predictions.append(c_cont_pred)
        if actual_quality is not None:
            self._actuals.append(actual_quality)

    def record_ab(self, with_ctrl: float, without_ctrl: float) -> None:
        self._ab_pairs.append((with_ctrl, without_ctrl))

    def summary(self) -> Dict:
        result: Dict = {}

        # Correlation
        n = min(len(self._predictions), len(self._actuals))
        if n >= 10:
            preds = list(self._predictions)[-n:]
            actuals = list(self._actuals)[-n:]
            mp = sum(preds) / n
            ma = sum(actuals) / n
            cov = sum((p - mp) * (a - ma) for p, a in zip(preds, actuals)) / n
            sp = math.sqrt(sum((p - mp)**2 for p in preds) / n)
            sa = math.sqrt(sum((a - ma)**2 for a in actuals) / n)
            corr = cov / (sp * sa) if sp > 1e-6 and sa > 1e-6 else 0.0
            result["c_cont_correlation"] = corr
            result["n_prediction_pairs"] = n
        else:
            result["c_cont_correlation"] = None
            result["n_prediction_pairs"] = n

        # A/B comparison
        if self._ab_pairs:
            pairs = list(self._ab_pairs)
            deltas = [w - wo for w, wo in pairs]
            mean_delta = sum(deltas) / len(deltas)
            pct_better = sum(1 for d in deltas if d > 0) / len(deltas)
            result["ab_mean_delta"] = mean_delta
            result["ab_pct_controller_better"] = pct_better
            result["n_ab_pairs"] = len(pairs)
        else:
            result["ab_mean_delta"] = None
            result["ab_pct_controller_better"] = None
            result["n_ab_pairs"] = 0

        return result


# ─────────────────────────────────────────────────────────────────────────── #
# 5. Perception-action coupling                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class PerceptionCouplingMetrics:
    """
    Tracks whether visual/audio surprise triggers appropriate controller responses.

    From dump1.txt:
      "Visual/audio surprise contributes to controller triggers"
      "The system references *ongoing* perceptual context, not just captions"
    """

    def __init__(self, surprise_threshold: float = 1.5, window: int = 100_000):
        self.surprise_threshold = surprise_threshold
        self._records: Deque[Tuple[float, float, bool]] = deque(maxlen=window)
        # (visual_surprise, audio_surprise, controller_fired)

    def record(
        self,
        visual_surprise: float,
        audio_surprise: float,
        controller_fired: bool,
    ) -> None:
        self._records.append((visual_surprise, audio_surprise, controller_fired))

    def summary(self) -> Dict:
        if not self._records:
            return {"status": "no_data"}

        high_surprise_steps = [
            (vs, aus, fired)
            for vs, aus, fired in self._records
            if vs >= self.surprise_threshold or aus >= self.surprise_threshold
        ]
        n_high = len(high_surprise_steps)
        n_fired_on_high = sum(1 for _, _, f in high_surprise_steps if f)

        low_surprise_steps = [
            (vs, aus, fired)
            for vs, aus, fired in self._records
            if vs < self.surprise_threshold and aus < self.surprise_threshold
        ]
        n_fired_on_low = sum(1 for _, _, f in low_surprise_steps if f)

        coupling_rate = n_fired_on_high / n_high if n_high > 0 else None
        false_alarm_rate = n_fired_on_low / len(low_surprise_steps) if low_surprise_steps else None

        return {
            "n_high_surprise_steps": n_high,
            "n_fired_on_high_surprise": n_fired_on_high,
            "coupling_rate": coupling_rate,        # ideally > 0.6
            "false_alarm_rate": false_alarm_rate,  # ideally < 0.1
            "n_total_steps": len(self._records),
        }


# ─────────────────────────────────────────────────────────────────────────── #
# 6. Self-dynamics model metrics                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class SelfModelMetrics:
    """
    Tracks SelfDynamicsModel (§Ψ̃_L) prediction quality over training.

    Four prediction dimensions (matching SelfDynamicsModel.SUMMARY_DIM):
      0: d_id      — identity drift
      1: eps_norm  — prediction error norm
      2: c_cont    — continuation confidence
      3: v_self    — viability

    Key questions:
      - Is L_self_model actually decreasing? (model is learning its own dynamics)
      - Which dimension is hardest to predict? (reveals what is most chaotic)
      - How often does V_self augmentation reduce V_self? (forward-looking pessimism active)
    """

    DIM_NAMES = ("d_id", "eps_norm", "c_cont", "v_self")

    def __init__(self, window: int = 100_000):
        self._l_self: Deque[float] = deque(maxlen=window)
        # Per-dimension MSE
        self._dim_errors: Deque[Tuple[float, float, float, float]] = deque(maxlen=window)
        # Count of steps where V_self augmentation reduced V_self (pessimism active)
        self._pessimism_active: Deque[bool] = deque(maxlen=window)

    def record(
        self,
        l_self: float,
        pred: Optional[Tuple[float, float, float, float]] = None,   # predicted (d_id, eps, c_cont, v_self)
        actual: Optional[Tuple[float, float, float, float]] = None, # actual (d_id, eps, c_cont, v_self)
        v_self_augmented: Optional[bool] = None,  # True if SDM prediction reduced V_self this step
    ) -> None:
        self._l_self.append(l_self)
        if pred is not None and actual is not None:
            self._dim_errors.append(tuple((p - a) ** 2 for p, a in zip(pred, actual)))
        if v_self_augmented is not None:
            self._pessimism_active.append(v_self_augmented)

    def summary(self) -> Dict:
        if not self._l_self:
            return {"status": "no_data"}
        values = list(self._l_self)
        result: Dict = {
            "l_self_mean": sum(values) / len(values),
            "l_self_recent": sum(values[-50:]) / max(len(values[-50:]), 1),
            "n_steps": len(values),
        }
        # Trend (last quarter vs first quarter)
        q = max(1, len(values) // 4)
        result["l_self_trend"] = "improving" if sum(values[-q:]) / q < sum(values[:q]) / q else "stagnant"

        if self._dim_errors:
            errors = list(self._dim_errors)
            for i, name in enumerate(self.DIM_NAMES):
                mse = sum(e[i] for e in errors) / len(errors)
                result[f"mse_{name}"] = mse

        if self._pessimism_active:
            result["pessimism_active_rate"] = sum(self._pessimism_active) / len(self._pessimism_active)

        return result


# ─────────────────────────────────────────────────────────────────────────── #
# 7. Aggregated evaluation report                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class EvaluationReport:
    identity_drift: Dict
    controller: Dict
    memory: Dict
    c_cont: Dict
    perception_coupling: Dict
    self_model: Dict = field(default_factory=dict)
    step: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"=== Amore Evaluation Report (step {self.step}) ==="]

        lines.append("\n[Identity Drift]")
        id_d = self.identity_drift
        if id_d.get("status") == "no_data":
            lines.append("  No data.")
        else:
            lines.append(f"  D_id mean={id_d.get('d_id_mean', 0):.4f}  max={id_d.get('d_id_max', 0):.4f}  trend={id_d.get('trend', '?')}")
            if id_d.get("n_reloads", 0) > 0:
                lines.append(f"  Post-reload mean D_id={id_d.get('post_reload_mean_d_id', 0):.4f}  converges={id_d.get('reload_converges')}")

        lines.append("\n[Controller]")
        ctrl = self.controller
        if ctrl.get("status") == "no_data":
            lines.append("  No data.")
        else:
            lines.append(f"  Precision={ctrl.get('precision', 0):.3f}  Recall={ctrl.get('recall', 0):.3f}  F1={ctrl.get('f1', 0):.3f}")
            lines.append(f"  Fire rate={ctrl.get('fire_rate', 0):.3f}  TP={ctrl.get('tp')} FP={ctrl.get('fp')} FN={ctrl.get('fn')}")
            if ctrl.get("mean_quality_delta") is not None:
                lines.append(f"  Quality delta={ctrl['mean_quality_delta']:.4f}  % helped={ctrl.get('pct_interventions_helped', 0):.1%}")

        lines.append("\n[Memory]")
        mem = self.memory
        lines.append(f"  Write rate={mem.get('write_rate', 0):.3f}  (OK={mem.get('write_rate_ok')})")
        if mem.get("n_retrievals_measured", 0) > 0:
            lines.append(f"  Retrieval delta={mem.get('mean_retrieval_delta', 0):.4f}  % helpful={mem.get('pct_retrievals_helpful', 0):.1%}")

        lines.append("\n[C_cont Quality]")
        cc = self.c_cont
        if cc.get("c_cont_correlation") is not None:
            lines.append(f"  Pearson r={cc['c_cont_correlation']:.4f}  (n={cc.get('n_prediction_pairs')})")
        if cc.get("ab_mean_delta") is not None:
            lines.append(f"  A/B delta={cc['ab_mean_delta']:.4f}  controller better {cc.get('ab_pct_controller_better', 0):.1%}")

        lines.append("\n[Perception Coupling]")
        pc = self.perception_coupling
        if pc.get("status") == "no_data":
            lines.append("  No data.")
        else:
            lines.append(f"  Coupling rate={pc.get('coupling_rate', 0):.3f}  False alarm={pc.get('false_alarm_rate', 0):.3f}")

        if self.self_model and self.self_model.get("status") != "no_data":
            lines.append("\n[Self-Dynamics Model §Ψ̃_L]")
            sm = self.self_model
            lines.append(f"  L_self mean={sm.get('l_self_mean', 0):.4f}  recent={sm.get('l_self_recent', 0):.4f}  trend={sm.get('l_self_trend', '?')}")
            for dim in SelfModelMetrics.DIM_NAMES:
                key = f"mse_{dim}"
                if key in sm:
                    lines.append(f"    {dim} MSE={sm[key]:.4f}")
            if "pessimism_active_rate" in sm:
                lines.append(f"  V_self pessimism active {sm['pessimism_active_rate']:.1%} of steps")

        if self.notes:
            lines.append("\n[Notes]")
            for n in self.notes:
                lines.append(f"  {n}")

        return "\n".join(lines)


def build_evaluation_report(
    step: int,
    drift_tracker: IdentityDriftTracker,
    controller_metrics: ControllerMetrics,
    memory_metrics: MemoryMetrics,
    c_cont_metrics: CContMetrics,
    perception_metrics: PerceptionCouplingMetrics,
    self_model_metrics: Optional["SelfModelMetrics"] = None,
) -> EvaluationReport:
    """Snapshot all trackers into a single report."""
    notes = []

    ctrl_summary = controller_metrics.summary()
    if ctrl_summary.get("precision", 1.0) < 0.70:
        notes.append("Controller precision below 0.70 — too many spurious interventions.")
    if ctrl_summary.get("recall", 1.0) < 0.60:
        notes.append("Controller recall below 0.60 — missing genuine distress events.")

    mem_summary = memory_metrics.summary()
    if not mem_summary.get("write_rate_ok", True):
        notes.append(f"Memory write rate {mem_summary.get('write_rate', 0):.1%} exceeds 15% — risk of noise writes.")

    sm_summary: Dict = {}
    if self_model_metrics is not None:
        sm_summary = self_model_metrics.summary()
        if sm_summary.get("l_self_trend") == "stagnant" and sm_summary.get("n_steps", 0) > 200:
            notes.append("SelfDynamicsModel L_self not improving after 200 steps — check warmup or d_sdm.")

    return EvaluationReport(
        step=step,
        identity_drift=drift_tracker.summary(),
        controller=ctrl_summary,
        memory=mem_summary,
        c_cont=c_cont_metrics.summary(),
        perception_coupling=perception_metrics.summary(),
        self_model=sm_summary,
        notes=notes,
    )
