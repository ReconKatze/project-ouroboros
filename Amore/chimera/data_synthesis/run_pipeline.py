"""run_pipeline.py — CLI entry point for the full data synthesis pipeline.

Safe to interrupt and resume at any time — completed tasks are checkpointed.

Usage examples:

    # Anthropic backend (default)
    python -m chimera.data_synthesis.run_pipeline \\
        --output-dir data/synthesis \\
        --budget 20.0

    # OpenAI backend
    python -m chimera.data_synthesis.run_pipeline \\
        --output-dir data/synthesis \\
        --backend openai \\
        --model gpt-4o \\
        --budget 0      # disable budget tracking

    # LiteLLM (any provider)
    python -m chimera.data_synthesis.run_pipeline \\
        --output-dir data/synthesis \\
        --backend litellm \\
        --model ollama/llama3 \\
        --budget 0

    # Local Ollama (no API key needed)
    python -m chimera.data_synthesis.run_pipeline \\
        --output-dir data/synthesis \\
        --backend http \\
        --base-url http://localhost:11434/v1 \\
        --model llama3 \\
        --budget 0

    # Run only specific generators
    python -m chimera.data_synthesis.run_pipeline \\
        --output-dir data/synthesis \\
        --only cot error_pairs

    # Skip assembly (just run generators)
    python -m chimera.data_synthesis.run_pipeline \\
        --output-dir data/synthesis \\
        --no-build
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_backend(args: argparse.Namespace):
    """Construct the backend callable from CLI args."""
    from chimera.data_synthesis.backends import (
        anthropic_backend,
        http_backend,
        litellm_backend,
        openai_backend,
    )

    backend_name = args.backend.lower()

    if backend_name == "anthropic":
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: Anthropic backend requires --api-key or ANTHROPIC_API_KEY env var")
            sys.exit(1)
        model = args.model or "claude-opus-4-6"
        print(f"Backend: Anthropic ({model})")
        return anthropic_backend(api_key=api_key, model=model)

    elif backend_name == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        model = args.model or "gpt-4o"
        print(f"Backend: OpenAI ({model})")
        return openai_backend(api_key=api_key, model=model, base_url=args.base_url)

    elif backend_name == "litellm":
        if not args.model:
            print("ERROR: litellm backend requires --model (e.g. 'ollama/llama3')")
            sys.exit(1)
        print(f"Backend: LiteLLM ({args.model})")
        return litellm_backend(model=args.model, api_key=args.api_key)

    elif backend_name == "http":
        if not args.base_url:
            print("ERROR: http backend requires --base-url")
            sys.exit(1)
        if not args.model:
            print("ERROR: http backend requires --model")
            sys.exit(1)
        print(f"Backend: HTTP ({args.base_url}, {args.model})")
        return http_backend(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key or "local",
        )

    else:
        print(f"ERROR: Unknown backend '{args.backend}'. Choose: anthropic, openai, litellm, http")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chimera data synthesis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", required=True, help="Directory for output JSONL files")
    parser.add_argument(
        "--backend",
        default="anthropic",
        choices=["anthropic", "openai", "litellm", "http"],
        help="LLM backend to use (default: anthropic)",
    )
    parser.add_argument("--model", default=None, help="Model string (backend-specific)")
    parser.add_argument("--api-key", default=None, help="API key (overrides env var)")
    parser.add_argument("--base-url", default=None, help="Base URL for http/openai backend")
    parser.add_argument(
        "--budget",
        type=float,
        default=20.0,
        help="Dollar budget cap (0 = disabled). Only tracked for backends that report cost.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["cot", "anchor", "adversarial", "cultural", "preferences", "error_pairs"],
        default=None,
        help="Run only specified generators (default: all)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip the build_corpus assembly step",
    )
    parser.add_argument(
        "--scenarios-per-category",
        type=int,
        default=3,
        help="Cultural scenarios per category (default: 3)",
    )
    args = parser.parse_args()

    from chimera.data_synthesis.api_client import BudgetExceeded, SynthesisClient
    from chimera.data_synthesis.build_corpus import build_corpus
    from chimera.data_synthesis.generate_constitutional import (
        generate_adversarial_responses,
        generate_anchor_responses,
        generate_cultural_scenarios,
    )
    from chimera.data_synthesis.generate_cot import generate_cot_traces
    from chimera.data_synthesis.generate_error_pairs import generate_error_pairs
    from chimera.data_synthesis.generate_preferences import generate_preference_pairs

    backend = build_backend(args)
    client = SynthesisClient(
        backend=backend,
        output_dir=args.output_dir,
        budget_usd=args.budget,
    )

    only = set(args.only) if args.only else None

    try:
        if only is None or "cot" in only:
            generate_cot_traces(client)

        if only is None or "anchor" in only:
            generate_anchor_responses(client)

        if only is None or "adversarial" in only:
            generate_adversarial_responses(client)

        if only is None or "cultural" in only:
            generate_cultural_scenarios(
                client,
                scenarios_per_category=args.scenarios_per_category,
            )

        if only is None or "preferences" in only:
            generate_preference_pairs(client)

        if only is None or "error_pairs" in only:
            generate_error_pairs(client)

    except BudgetExceeded as e:
        print(f"\n[pipeline] Stopped: {e}")
    except KeyboardInterrupt:
        print("\n[pipeline] Interrupted — progress saved to checkpoint.")

    if not args.no_build:
        try:
            build_corpus(args.output_dir)
        except Exception as e:
            print(f"[pipeline] build_corpus failed: {e}")

    client.print_stats()


if __name__ == "__main__":
    main()
