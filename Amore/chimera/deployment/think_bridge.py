"""
ThinkBridge — [THINK] token injection for hermes-agent.

hermes-agent's agent/prompt_builder.py constructs the system prompt before every
API call.  That is exactly where the [THINK] instruction is injected: as a suffix
appended to the assembled system prompt.  No model changes needed; pure shell
behaviour.

The response stripping (hiding inner monologue from the user) happens after the
API call — in run_agent.py wherever the response text is displayed.

Two integration points, both minimal:

  1. In agent/prompt_builder.py — add [THINK] instruction to system prompt:

       from chimera.deployment.think_bridge import ThinkBridge
       _bridge = ThinkBridge(n_think_tokens=64)

       # At the end of build_system_prompt() (or wherever system_prompt is assembled):
       system_prompt += _bridge.build_system_prompt_suffix()

  2. In run_agent.py — strip before displaying to user:

       from chimera.deployment.think_bridge import ThinkBridge
       _bridge = ThinkBridge(n_think_tokens=64)

       # After receiving response text:
       visible_text = _bridge.strip_think(raw_response_text)

       # Optional — to capture inner monologue for training signal:
       inner, visible_text = _bridge.extract_think(raw_response_text)

The complete wrap_messages() method is also provided for cases where you want
stronger priming (forcing an assistant-prefix [THINK] turn in the messages list),
but the system-prompt approach is simpler and requires no messages-list surgery.

This is the "minimal 9B test" of think-before-responding from inner_stream.md.
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
    Injects [THINK] framing into hermes-agent and strips the inner monologue
    from the displayed response.

    Typical usage:
      - build_system_prompt_suffix() → appended in agent/prompt_builder.py
      - strip_think(response_text) → called in run_agent.py after API response
      - extract_think(response_text) → optional training-signal capture

    Parameters
    ----------
    n_think_tokens : int
        Rough target for the inner monologue length (hint in the instruction;
        the model may produce more or less).
    think_instruction : str | None
        Custom instruction text.  If None, uses the default Chimera inner-stream
        instruction.
    strip_from_display : bool
        Whether strip_think() removes think blocks from the displayed text.
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

    # ------------------------------------------------------------------ #
    # Primary integration surface — system prompt injection               #
    # Used in agent/prompt_builder.py                                     #
    # ------------------------------------------------------------------ #

    def build_system_prompt_suffix(self) -> str:
        """Return the [THINK] instruction text to append to the system prompt.

        Usage in agent/prompt_builder.py:
            system_prompt += bridge.build_system_prompt_suffix()

        The model will then open every response with a [THINK]...[/THINK] block
        containing its inner monologue, followed by the visible response.
        """
        return "\n\n" + self._instruction

    # ------------------------------------------------------------------ #
    # Response handling — called in run_agent.py after the API call       #
    # ------------------------------------------------------------------ #

    def strip_think(self, response_text: str) -> str:
        """Remove all [THINK]...[/THINK] blocks from a model response.

        Returns the cleaned text for display to the user.
        The inner monologue is silently discarded (use extract_think to keep it).
        """
        if not self.strip_from_display:
            return response_text
        cleaned = _THINK_RE.sub("", response_text)
        return cleaned.strip()

    def extract_think(self, response_text: str) -> Tuple[str, str]:
        """Split a response into (inner_monologue, visible_text).

        Useful for logging and training signal collection: inner_monologue
        captures the model's reasoning; visible_text is what the user sees.

        Returns
        -------
        inner_monologue : str
            Everything inside [THINK]...[/THINK] (concatenated if multiple blocks).
        visible_text : str
            The response with all think blocks removed.
        """
        inner_parts = _THINK_RE.findall(response_text)
        inner_clean = []
        for block in inner_parts:
            text = re.sub(r"^\[THINK\]\s*", "", block, flags=re.IGNORECASE)
            text = re.sub(r"\s*\[/THINK\]$", "", text, flags=re.IGNORECASE)
            inner_clean.append(text.strip())

        inner_monologue = "\n\n".join(inner_clean)
        visible_text = _THINK_RE.sub("", response_text).strip()
        return inner_monologue, visible_text

    # ------------------------------------------------------------------ #
    # Alternative — stronger priming via messages-list surgery            #
    # Use this if system-prompt injection alone is insufficient.          #
    # ------------------------------------------------------------------ #

    def wrap_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Inject [THINK] framing directly into the messages list.

        Adds the instruction to the system message AND appends a forced
        assistant-prefix "[THINK]\\n" to guarantee the model opens with its
        inner monologue.  More aggressive than build_system_prompt_suffix()
        but requires touching the messages list rather than just the system
        prompt string.

        Returns a new list; the input is not modified.
        """
        result = list(messages)

        # Append instruction to existing system message (or create one)
        if result and result[0].get("role") == "system":
            result[0] = dict(result[0])
            result[0]["content"] = result[0]["content"].rstrip() + "\n\n" + self._instruction
        else:
            result.insert(0, {"role": "system", "content": self._instruction})

        # Force-prime the assistant to begin with [THINK]
        result.append({"role": "assistant", "content": f"{THINK_OPEN}\n"})

        return result
