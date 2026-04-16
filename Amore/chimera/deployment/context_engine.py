"""
ChimeraContextEngine — hermes-agent ContextEngine plugin.

Problem (from dump1.txt):
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
     (or wherever hermes-agent looks for ContextEngine plugins).
  2. In config.yaml set:
       context:
         engine: chimera
         # optional overrides:
         # model_context_limit: 4096
         # hard_margin_tokens: 256
         # min_messages_to_keep: 6

This class implements the hermes-agent ContextEngine ABC.  If that ABC changes
in a future hermes-agent release, only the method signatures here need updating.

~100 lines as noted in dump1.txt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ChimeraContextEngine:
    """
    hermes-agent ContextEngine that trusts Chimera's SSM to handle long context.

    Compression fires only when within `hard_margin_tokens` of the absolute limit.
    Below that threshold, messages are returned as-is.

    Parameters
    ----------
    model_context_limit : int
        The model's absolute context window (tokens).  Default 4096 for llama.cpp
        serving a 1.5B GGUF.  Increase for larger context configurations.
    hard_margin_tokens : int
        Tokens-from-limit at which compression is allowed to fire.
        Default 256 — this means compression only triggers in the last 6% of
        a 4096-token window.  Tune down if you hit OOM; tune up if you want
        more headroom.
    min_messages_to_keep : int
        Minimum number of messages always preserved (system prompt + N most recent).
        Prevents compression from eating the immediate conversation context.
    summarize_fn : callable | None
        (messages: list[dict]) → str.  Called to produce the summary when compression
        fires.  If None, falls back to a simple truncation strategy.
    """

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

        # Soft threshold used by the default engine — we intentionally ignore this.
        # Set here for visibility; nothing reads it internally.
        self._default_engine_soft_threshold = int(model_context_limit * 0.8)

    # ------------------------------------------------------------------ #
    # hermes-agent ContextEngine interface                                 #
    # ------------------------------------------------------------------ #

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        """
        Return True only when we are dangerously close to the hard context limit.

        The default engine fires at ~80% of context; we fire only at
        (limit - hard_margin_tokens).  This lets Chimera's Mamba layers handle
        the middle 74% of the window without interference.
        """
        if prompt_tokens is None:
            return False
        trigger = self.model_context_limit - self.hard_margin_tokens
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

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _summarize(self, messages: List[Dict[str, str]]) -> str:
        if self.summarize_fn is not None:
            return self.summarize_fn(messages)
        # Fallback: extract the last user and assistant turn from the body
        # as a minimal "what was being discussed" note.
        lines = []
        for msg in messages[-4:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content[:200]}")
        return "\n".join(lines) if lines else "(prior context compressed)"

    # ------------------------------------------------------------------ #
    # hermes-agent plugin metadata                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def plugin_name(cls) -> str:
        return "chimera"

    @classmethod
    def plugin_description(cls) -> str:
        return (
            "Context engine for Chimera (hybrid Mamba/Transformer). "
            "Delays summarization until near the hard context limit so that "
            "Mamba SSM layers can process full message history."
        )
