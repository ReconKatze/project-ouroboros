"""api_client.py — Provider-agnostic client for the data synthesis pipeline.

The core SynthesisClient is independent of any specific LLM SDK.
It takes a *backend callable* that does the actual API work.
Ready-made backends live in ``backends.py``.

Backend protocol:
    backend(system, messages, *, max_tokens, **kwargs) -> (text: str, cost_usd: float)

    text     : the model's response text
    cost_usd : estimated cost for this call (0.0 if unknown / not tracking)

SynthesisClient provides on top of any backend:
  - Budget tracking with hard cutoff
  - Resumability via a checkpoint file (JSON set of completed task IDs)
  - Atomic JSONL output

Usage with the built-in Anthropic backend::

    from chimera.data_synthesis.api_client import SynthesisClient
    from chimera.data_synthesis.backends import anthropic_backend

    client = SynthesisClient(
        backend=anthropic_backend(api_key="sk-ant-..."),
        output_dir="data/synthesis",
        budget_usd=20.0,
    )

Usage with a custom backend::

    def my_backend(system, messages, *, max_tokens=4096, **kwargs):
        text = call_my_llm(system, messages, max_tokens=max_tokens)
        return text, 0.0   # cost unknown

    client = SynthesisClient(backend=my_backend, output_dir="data/synthesis")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Backend type alias for documentation purposes
# backend(system, messages, *, max_tokens, **kwargs) -> (text, cost_usd)
Backend = Callable[..., Tuple[str, float]]


class BudgetExceeded(Exception):
    """Raised when cumulative API spend exceeds the configured dollar limit."""


class SynthesisClient:
    """Provider-agnostic synthesis client.

    Parameters
    ----------
    backend :
        Callable with signature
        ``(system: str, messages: list[dict], *, max_tokens: int, **kwargs) -> (str, float)``.
        The float is cost in USD for the call; use 0.0 if not tracking.
    output_dir :
        Directory for JSONL output and the checkpoint file.
    budget_usd :
        Raise ``BudgetExceeded`` when cumulative cost reaches this limit.
        Has no effect when all backends return 0.0 for cost.
    """

    def __init__(
        self,
        backend: Backend,
        output_dir: Union[str, Path],
        budget_usd: float = 20.0,
    ) -> None:
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.budget_usd = budget_usd

        self._checkpoint_path = self.output_dir / "checkpoint.json"
        self._completed: set[str] = self._load_checkpoint()

        self.total_cost_usd: float = 0.0
        self.total_calls: int = 0

    # ------------------------------------------------------------------
    # Checkpoint

    def _load_checkpoint(self) -> set[str]:
        if self._checkpoint_path.exists():
            try:
                data = json.loads(self._checkpoint_path.read_text())
                return set(data.get("completed", []))
            except (json.JSONDecodeError, KeyError):
                return set()
        return set()

    def _save_checkpoint(self) -> None:
        tmp = self._checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"completed": sorted(self._completed)}, indent=2))
        tmp.replace(self._checkpoint_path)

    def is_completed(self, task_id: str) -> bool:
        return task_id in self._completed

    def mark_completed(self, task_id: str) -> None:
        self._completed.add(task_id)
        self._save_checkpoint()

    # ------------------------------------------------------------------
    # Budget

    def _check_budget(self) -> None:
        if self.budget_usd > 0 and self.total_cost_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"Budget of ${self.budget_usd:.2f} exceeded "
                f"(spent ${self.total_cost_usd:.4f})"
            )

    # ------------------------------------------------------------------
    # Core call

    def call(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Call the backend and return the response text.

        Parameters
        ----------
        system :
            System prompt string.
        messages :
            List of ``{"role": ..., "content": ...}`` dicts.
        max_tokens :
            Maximum tokens for the completion.
        **kwargs :
            Passed through to the backend. Common keys:
            ``thinking=True`` (Anthropic adaptive thinking),
            ``temperature=0.7`` (OpenAI/generic).
        """
        self._check_budget()

        text, cost = self.backend(system, messages, max_tokens=max_tokens, **kwargs)

        self.total_cost_usd += cost
        self.total_calls += 1
        cost_str = f" cost=${cost:.4f} total=${self.total_cost_usd:.4f}" if cost > 0 else ""
        print(f"  [api] call #{self.total_calls}{cost_str}")

        return text

    # ------------------------------------------------------------------
    # JSONL output

    def append_jsonl(self, path: Union[str, Path], record: Dict[str, Any]) -> None:
        """Append a single record to a JSONL file."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_jsonl(self, path: Union[str, Path], records: Sequence[Dict[str, Any]]) -> None:
        """Atomically overwrite a JSONL file with the given records."""
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Stats

    def print_stats(self) -> None:
        cost_line = (
            f"  Total cost:      ${self.total_cost_usd:.4f} / ${self.budget_usd:.2f}\n"
            if self.budget_usd > 0
            else ""
        )
        print(
            f"\n=== Synthesis Stats ===\n"
            f"{cost_line}"
            f"  Total calls:     {self.total_calls}\n"
            f"  Tasks completed: {len(self._completed)}\n"
        )
