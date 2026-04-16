"""
chimera.evaluation — 30B evaluation framework.

Tracks the metrics that determine whether the Amore architecture actually
produces a persistent, self-referential, memory-bearing agent under real
conditions (from dump1.txt: "Does this system behave like a persistent,
self-referential, memory-bearing agent under real-world conditions?")

Key metrics:
  - D_id over time: identity drift curves
  - Controller precision/recall: does it fire for the right reasons?
  - Memory usefulness: sparse writes + retrieval improvement
  - C_cont vs baseline: does controller intervention actually help?
  - Perception-action coupling: does visual/audio surprise trigger the controller?
  - Continuation quality: with vs. without memory

See chimera/evaluation/metrics.py for implementations.
"""

from .metrics import (
    IdentityDriftTracker,
    ControllerMetrics,
    MemoryMetrics,
    CContMetrics,
    PerceptionCouplingMetrics,
    SelfModelMetrics,
    EvaluationReport,
    build_evaluation_report,
)

__all__ = [
    "IdentityDriftTracker",
    "ControllerMetrics",
    "MemoryMetrics",
    "CContMetrics",
    "PerceptionCouplingMetrics",
    "SelfModelMetrics",
    "EvaluationReport",
    "build_evaluation_report",
]
