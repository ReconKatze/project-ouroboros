"""backends.py — Ready-made backend callables for SynthesisClient.

Each function returns a backend: a callable with signature
    (system: str, messages: list[dict], *, max_tokens: int, **kwargs) -> (str, float)

The float return is cost in USD (0.0 if not tracked).

Available backends
------------------
anthropic_backend   — Anthropic SDK (claude-opus-4-6 default, prompt caching, adaptive thinking)
openai_backend      — OpenAI SDK (gpt-4o default)
litellm_backend     — LiteLLM proxy (any model string litellm supports)
http_backend        — Raw HTTP to any OpenAI-compatible endpoint (Ollama, Together, etc.)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Anthropic

# Pricing reference for cost estimation
_ANTHROPIC_PRICES: Dict[str, Dict[str, float]] = {
    "claude-opus-4-6":   {"input": 5.00,  "output": 25.00, "cache_write": 6.25,  "cache_read": 0.50},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00, "cache_write": 3.75,  "cache_read": 0.30},
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00,  "cache_write": 1.25,  "cache_read": 0.10},
}


def anthropic_backend(
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-6",
    cache_system: bool = True,
    max_retries: int = 8,
):
    """Backend that calls the Anthropic SDK.

    Parameters
    ----------
    api_key :
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    model :
        Model string. Default is ``claude-opus-4-6``.
    cache_system :
        Attach ``cache_control: {type: ephemeral}`` to the system prompt.
        Saves cost when the same system prompt is reused across many calls.
    max_retries :
        Passed to ``anthropic.Anthropic`` — handles 429/529 with backoff.
    """
    import anthropic as _anthropic

    client = _anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        max_retries=max_retries,
    )
    prices = _ANTHROPIC_PRICES.get(model, _ANTHROPIC_PRICES["claude-opus-4-6"])

    def _call(
        system: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 4096,
        thinking: bool = False,
        **_ignored: Any,
    ) -> Tuple[str, float]:
        # Build system block — optionally cached
        if cache_system:
            system_content = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            system_content = [{"type": "text", "text": system}]

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_content,
            "messages": messages,
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        with client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        # Extract text blocks (skip thinking blocks)
        parts = [b.text for b in response.content if hasattr(b, "type") and b.type == "text"]
        text = "\n\n".join(parts).strip()

        # Cost estimate
        u = response.usage
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        uncached = max(0, (getattr(u, "input_tokens", 0) or 0) - cache_write - cache_read)
        output = getattr(u, "output_tokens", 0) or 0
        cost = (
            uncached      * prices["input"]       / 1_000_000
            + cache_write * prices["cache_write"] / 1_000_000
            + cache_read  * prices["cache_read"]  / 1_000_000
            + output      * prices["output"]      / 1_000_000
        )
        print(
            f"    [anthropic] in={getattr(u,'input_tokens',0):,} "
            f"out={output:,} "
            f"cr={cache_read:,} cw={cache_write:,}"
        )
        return text, cost

    return _call


# ---------------------------------------------------------------------------
# OpenAI

def openai_backend(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
):
    """Backend that calls the OpenAI SDK.

    Parameters
    ----------
    api_key :
        OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
    model :
        Model string (e.g. ``gpt-4o``, ``gpt-4o-mini``).
    base_url :
        Override the API base URL (useful for Azure, Together, etc.).
    """
    import openai as _openai

    client_kwargs: Dict[str, Any] = {
        "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    client = _openai.OpenAI(**client_kwargs)

    def _call(
        system: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **_ignored: Any,
    ) -> Tuple[str, float]:
        full_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        return text.strip(), 0.0  # cost not tracked for OpenAI by default

    return _call


# ---------------------------------------------------------------------------
# LiteLLM (any model string it supports)

def litellm_backend(
    model: str,
    api_key: Optional[str] = None,
    **litellm_kwargs: Any,
):
    """Backend that calls LiteLLM — supports 100+ model providers.

    model string examples:
        "anthropic/claude-opus-4-6"
        "openai/gpt-4o"
        "ollama/llama3"
        "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"
        "gemini/gemini-1.5-pro"

    Requires:  pip install litellm
    """
    import litellm as _litellm

    def _call(
        system: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **_ignored: Any,
    ) -> Tuple[str, float]:
        full_messages = [{"role": "system", "content": system}] + messages
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **litellm_kwargs,
        }
        if api_key:
            call_kwargs["api_key"] = api_key

        response = _litellm.completion(**call_kwargs)
        text = response.choices[0].message.content or ""

        # LiteLLM exposes cost via response._hidden_params when available
        cost = 0.0
        try:
            cost = getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
        except Exception:
            pass

        return text.strip(), cost

    return _call


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible HTTP (Ollama, vLLM, Together, etc.)

def http_backend(
    base_url: str,
    model: str,
    api_key: str = "local",
):
    """Backend for any OpenAI-compatible REST endpoint.

    Works with: Ollama (http://localhost:11434/v1), vLLM, Together AI,
    Anyscale, Fireworks, Perplexity, etc.

    Requires:  pip install openai
    """
    import openai as _openai

    client = _openai.OpenAI(api_key=api_key, base_url=base_url)

    def _call(
        system: str,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **_ignored: Any,
    ) -> Tuple[str, float]:
        full_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        return text.strip(), 0.0

    return _call
