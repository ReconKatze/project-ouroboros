"""
ThinkBridge — [THINK] token injection for hermes-agent.

From dump1.txt:
  "Your codex_memory notes: 'The minimal 9B test is think-before-responding
   with hidden [THINK] tokens.'  Hermes-agent's agent/prompt_builder.py is
   exactly where you'd inject this.  The prompt builder constructs the system
   prompt before each API call — that's where you'd prepend [THINK] framing
   and strip it from the final displayed response.  No model changes needed;
   pure shell behavior."

What this does:
  1. `wrap_messages(messages)` — adds a [THINK] framing turn before the
     final user message so the model generates an inner monologue first.
  2. `strip_think(response_text)` — removes everything inside [THINK]…[/THINK]
     from the model's output before displaying it to the user.

Inner tokens are never shown to the user.  Training signal: response quality
with vs. without the think pass.  Testable at 9B with no model changes.

Integration in hermes-agent:
  In agent/prompt_builder.py, before calling the API:
    from chimera.deployment.think_bridge import ThinkBridge
    bridge = ThinkBridge(n_think_tokens=64)
    messages = bridge.wrap_messages(messages)

  After receiving the response:
    visible_text = bridge.strip_think(raw_response)

See inner_stream.md for the full architecture; this is the "minimal 9B test"
version that requires no new infrastructure.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

THINK_OPEN = "[THINK]"
THINK_CLOSE = "[/THINK]"

# Regex to match a think block (greedy within a single response)
_THINK_RE = re.compile(
    r"\[THINK\].*?\[/THINK\]",
    re.DOTALL | re.IGNORECASE,
)


class ThinkBridge:
    """
    Injects [THINK] framing into hermes-agent message sequences and strips
    the inner monologue from the displayed response.

    Parameters
    ----------
    n_think_tokens : int
        Rough target for the inner monologue length (passed as hint in the
        system prompt injection; the model may produce more or less).
    think_instruction : str | None
        Custom instruction prepended to the think block.  If None, uses the
        default Chimera inner-stream instruction.
    strip_from_display : bool
        Whether to strip think blocks from the final displayed text.
        Set False during training/debugging to inspect the inner monologue.
    """

    def __init__(
        self,
        n_think_tokens: int = 64,
        think_instruction: Optional[str] = None,
        strip_from_display: bool = True,
    ):
        self.n_think_tokens = n_think_tokens
        self.strip_from_display = strip_from_display
        self._instruction = think_instruction or (
            "Before responding, use a brief internal monologue "
            f"(roughly {n_think_tokens} tokens) inside {THINK_OPEN}...{THINK_CLOSE} "
            "to consider the request, your persistent state, and any relevant context. "
            "Your inner monologue will never be shown to the user."
        )

    def wrap_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Prepend [THINK] framing to the message sequence.

        Strategy:
          - If there is a system message, append the think instruction to it.
          - Otherwise, inject a system message at position 0.
          - Append a brief assistant prefix "[THINK]\n" to prime the model
            to start its inner monologue immediately.

        Returns a new list; the input is not modified.
        """
        result = list(messages)

        # Inject instruction into system prompt
        if result and result[0].get("role") == "system":
            result[0] = dict(result[0])
            result[0]["content"] = result[0]["content"].rstrip() + "\n\n" + self._instruction
        else:
            result.insert(0, {"role": "system", "content": self._instruction})

        # Prime the assistant to begin with [THINK]
        result.append({"role": "assistant", "content": f"{THINK_OPEN}\n"})

        return result

    def strip_think(self, response_text: str) -> str:
        """
        Remove all [THINK]...[/THINK] blocks from a model response.
        Returns the cleaned text for display to the user.
        """
        if not self.strip_from_display:
            return response_text
        cleaned = _THINK_RE.sub("", response_text)
        return cleaned.strip()

    def extract_think(self, response_text: str) -> Tuple[str, str]:
        """
        Split a response into (inner_monologue, visible_text).
        Useful for logging/training signal collection.

        Returns
        -------
        inner_monologue : str
            Everything inside [THINK]...[/THINK] (concatenated if multiple blocks).
        visible_text : str
            The response with all think blocks removed.
        """
        inner_parts = _THINK_RE.findall(response_text)
        # Strip the THINK tags themselves from the captured blocks
        inner_clean = []
        for block in inner_parts:
            text = re.sub(r"^\[THINK\]\s*", "", block, flags=re.IGNORECASE)
            text = re.sub(r"\s*\[/THINK\]$", "", text, flags=re.IGNORECASE)
            inner_clean.append(text.strip())

        inner_monologue = "\n\n".join(inner_clean)
        visible_text = _THINK_RE.sub("", response_text).strip()
        return inner_monologue, visible_text
