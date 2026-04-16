"""data_synthesis — Frontier model knowledge transfer pipeline for Chimera.

Generates training data by querying Claude (Opus 4.6) to produce:
  - Chain-of-thought coding traces (highest leverage: teaches reasoning)
  - Constitutional alignment examples (anchor responses, adversarial robustness, cultural)
  - DPO preference pairs (sycophancy, hallucination, rigidity contrasts)
  - Bug/fix error pairs (correctness-focused coding examples)

All generators share a SynthesisClient that handles:
  - Prompt caching (system prompts cached — significant cost savings on repeated calls)
  - Budget tracking with hard cutoff
  - Resumability via checkpoint file (safe to interrupt and resume)
  - Atomic JSONL output compatible with HuggingFace load_dataset

Usage:
    from chimera.data_synthesis.api_client import SynthesisClient
    from chimera.data_synthesis.generate_cot import generate_cot_traces
    from chimera.data_synthesis.generate_constitutional import (
        generate_anchor_responses, generate_adversarial_responses, generate_cultural_scenarios
    )
    from chimera.data_synthesis.generate_preferences import generate_preference_pairs
    from chimera.data_synthesis.generate_error_pairs import generate_error_pairs
    from chimera.data_synthesis.build_corpus import build_corpus

Or use the CLI runner (safe to interrupt and resume):
    python -m chimera.data_synthesis.run_pipeline --output-dir data/synthesis --budget 10.0
"""

from .api_client import BudgetExceeded, SynthesisClient
from .backends import anthropic_backend, http_backend, litellm_backend, openai_backend
from .build_corpus import build_corpus
from .generate_constitutional import (
    generate_adversarial_responses,
    generate_anchor_responses,
    generate_cultural_scenarios,
)
from .generate_cot import generate_cot_traces
from .generate_error_pairs import generate_error_pairs
from .generate_preferences import generate_preference_pairs

__all__ = [
    "SynthesisClient",
    "BudgetExceeded",
    "anthropic_backend",
    "openai_backend",
    "litellm_backend",
    "http_backend",
    "generate_cot_traces",
    "generate_anchor_responses",
    "generate_adversarial_responses",
    "generate_cultural_scenarios",
    "generate_preference_pairs",
    "generate_error_pairs",
    "build_corpus",
]
