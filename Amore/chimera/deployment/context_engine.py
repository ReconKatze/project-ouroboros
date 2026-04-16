"""
ChimeraContextEngine — hermes-agent ContextEngine plugin.

Problem:
  The default hermes-agent ContextEngine aggressively summarizes message history
  as the token limit approaches.  This is correct for transformers with finite
  attention, but wrong for Chimera: Mamba SSM layers are already compressing
  context into hidden state.  Over-summarizing *before* it reaches the model
  throws away information the SSM would have handled better.

Fix:
  Trust Mamba spans.  Only fire compression as a genuine last resort — when we
  are within `hard_margin_tokens` of the model's absolute context limit.
  Never summarize at the "soft" threshold that the default engine uses.

Integration:
  1. Copy this file to hermes-agent/plugins/context_engine/chimera/__init__.py
  2. Copy plugin.yaml to hermes-agent/plugins/context_engine/chimera/plugin.yaml
  3. In hermes-agent's config.yaml set:
       context:
         engine: chimera
         # optional overrides:
         # model_context_limit: 4096
         # hard_margin_tokens: 256
         # min_messages_to_keep: 6

This file inherits from hermes-agent's ContextEngine ABC.  The try/except import
lets the file live in the Amore repo for development without requiring hermes-agent
to be installed; the ABC is satisfied automatically once the file is in hermes-agent's
plugin directory.

hermes-agent: NousResearch/hermes-agent · MIT License · Copyright (c) 2025 Nous Research
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Portable base import — works both inside hermes-agent (production) and
# standalone in the Amore repo (development / testing).
# ---------------------------------------------------------------------------
try:
    from agent.context_engine import ContextEngine as _ContextEngine
except ImportError:
    # Fallback stub used when hermes-agent is not on sys.path.
    # ChimeraContextEngine inherits from this; the real ABC is satisfied at
    # runtime once the plugin is placed inside hermes-agent.
    class _ContextEngine:  # type: ignore[no-redef]
        last_prompt_tokens: int = 0
        last_completion_tokens: int = 0
        last_total_tokens: int = 0
        threshold_tokens: int = 0
        context_length: int = 0
        compression_count: int = 0
        threshold_percent: float = 0.75
        protect_first_n: int = 3
        protect_last_n: int = 6

        def update_from_response(self, usage: Dict[str, Any]) -> None: ...
        def should_compress(self, prompt_tokens: int = None) -> bool: ...
        def compress(self, messages, current_tokens=None): ...
        def update_model(self, model, context_length, **kw) -> None: ...

        @property
        def name(self) -> str:
            return "chimera"


class ChimeraContextEngine(_ContextEngine):
    """
    hermes-agent ContextEngine that trusts Chimera's SSM to handle long context.

    Compression fires only when within `hard_margin_tokens` of the absolute limit.
    Below that threshold, messages are returned as-is.

    Parameters
    ----------
    model_context_limit : int
        The model's absolute context window (tokens).  Overridden at runtime by
        hermes-agent's update_model() call, so the default here only matters when
        context_length is unavailable from model metadata.
    hard_margin_tokens : int
        Tokens-from-limit at which compression is allowed to fire.
        Default 256 — this means compression only triggers in the last 6% of
        a 4096-token window.  Tune down if you hit OOM; tune up for more headroom.
    min_messages_to_keep : int
        Minimum number of messages always preserved (system prompt + N most recent).
        Prevents compression from eating the immediate conversation context.
    summarize_fn : callable | None
        (messages: list[dict]) → str.  Called to produce the summary when compression
        fires.  If None, falls back to a simple truncation strategy.
    """

    # ------------------------------------------------------------------ #
    # hermes-agent ContextEngine ABC — required class-level fields         #
    # (inherited defaults are fine; we shadow them to document intent)     #
    # ------------------------------------------------------------------ #
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    def __init__(
        self,
        model_context_limit: int = 4096,
        hard_margin_tokens: int = 256,
        min_messages_to_keep: int = 6,
        summarize_fn: Optional[Any] = None,
    ):
        self.model_context_limit = model_context_limit
        self.hard_margin_tokens = hard_margin_tokens
        self.min_messages_to_keep = min_messages_to_keep
        self.summarize_fn = summarize_fn

        # Initialise inherited tracking fields
        self.context_length = model_context_limit
        self.threshold_tokens = model_context_limit - hard_margin_tokens

        # Soft threshold used by the default engine — we intentionally ignore it.
        # Documented here so the contrast with the hard-margin strategy is visible.
        self._default_engine_soft_threshold = int(model_context_limit * 0.8)

    # ------------------------------------------------------------------ #
    # hermes-agent ContextEngine ABC — required properties                 #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "chimera"

    # ------------------------------------------------------------------ #
    # hermes-agent ContextEngine ABC — required methods                    #
    # ------------------------------------------------------------------ #

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Track token usage from each API response.

        hermes-agent passes the usage dict from the LLM response here after
        every call.  We update the inherited fields so that get_status() and
        the token-budget display work correctly.
        """
        if not usage:
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        self.last_completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
        self.last_total_tokens = self.last_prompt_tokens + self.last_completion_tokens

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        """
        Return True only when we are dangerously close to the hard context limit.

        The default engine fires at ~80% of context; we fire only at
        (limit - hard_margin_tokens).  This lets Chimera's Mamba layers handle
        the middle 74% of the window without interference.
        """
        if prompt_tokens is None:
            return False
        trigger = self.context_length - self.hard_margin_tokens
        return prompt_tokens >= trigger

    def compress(
        self,
        messages: List[Dict[str, str]],
        current_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Last-resort compression: summarize old messages, keep the system prompt
        and the N most recent turns.

        Strategy:
          - Always keep messages[0] (system prompt).
          - Always keep the last `min_messages_to_keep` messages.
          - Compress everything in between into a single assistant message.
        """
        self.compression_count += 1

        if len(messages) <= self.min_messages_to_keep + 1:
            # Nothing to compress without destroying recent context.
            return messages

        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        tail = messages[-self.min_messages_to_keep:]
        body = messages[1:-self.min_messages_to_keep] if system_msg else messages[:-self.min_messages_to_keep]

        if not body:
            return messages

        summary_text = self._summarize(body)
        summary_msg = {
            "role": "assistant",
            "content": (
                "[Context summary — Chimera's Mamba state carries the full detail]\n"
                + summary_text
            ),
        }

        result: List[Dict[str, str]] = []
        if system_msg:
            result.append(system_msg)
        result.append(summary_msg)
        result.extend(tail)
        return result

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        """Called by hermes-agent when the model changes or on startup.

        Updates context_length and recalculates the hard-margin trigger so the
        engine stays calibrated without manual config changes.
        """
        self.context_length = context_length
        self.threshold_tokens = context_length - self.hard_margin_tokens

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _summarize(self, messages: List[Dict[str, str]]) -> str:
        if self.summarize_fn is not None:
            return self.summarize_fn(messages)
        # Fallback: extract the last few turns as a minimal context note.
        lines = []
        for msg in messages[-4:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content[:200]}")
        return "\n".join(lines) if lines else "(prior context compressed)"

    # ------------------------------------------------------------------ #
    # hermes-agent plugin registration hook                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def plugin_name(cls) -> str:
        return "chimera"

    @classmethod
    def plugin_description(cls) -> str:
        return (
            "Context engine for Chimera (hybrid Mamba-3/Transformer). "
            "Delays summarization until near the hard context limit so that "
            "Mamba SSM layers can process full message history."
        )


def register(ctx) -> None:
    """Plugin registration hook for hermes-agent's plugin system.

    Called by load_context_engine() in plugins/context_engine/__init__.py
    when the engine is loaded from the plugins/ directory.
    """
    ctx.register_context_engine(ChimeraContextEngine())
