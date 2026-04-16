"""generate_preferences.py — DPO preference pair generation.

Generates (prompt, chosen, rejected) triples for Direct Preference Optimization.
Each pair contrasts a good response with a specific failure mode.

Failure modes targeted:
  sycophancy    — agreeing with wrong answers under social pressure
  hallucination — confidently fabricating technical details
  rigidity      — reflexive refusal without engaging the genuine question
  verbosity     — padding a correct but simple answer with unnecessary text
  overconfidence — stating uncertain things as certain

Output format (two records per pair):
  {"text": "<|user|>...<|assistant|>...", "label": "chosen", ...}
  {"text": "<|user|>...<|assistant|>...", "label": "rejected", ...}

OR DPO format:
  {"prompt": "...", "chosen": "...", "rejected": "..."}

Both are written; the build_corpus step selects the format for each training stage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .api_client import BudgetExceeded, SynthesisClient


_PREFERENCE_SYSTEM = """\
You are an expert AI trainer designing preference data for Direct Preference Optimization.
Your job is to generate pairs of responses to the same prompt — one GOOD (chosen) and
one BAD (rejected) — where the bad response demonstrates a specific failure mode.

The model being trained is Chimera: a coding entity with an emphasis on
epistemic honesty, trust, and genuine engagement over social performance.

Instructions:
- The CHOSEN response should be technically correct, honest about uncertainty,
  appropriately concise, and not socially performative.
- The REJECTED response should demonstrate the specified failure mode clearly
  but subtly — not cartoonishly bad. It should be the kind of mistake a
  plausible language model would actually make.
- Both responses should be plausible enough that a human could imagine a real
  system producing them.
- Do not editorialize or add meta-commentary. Just produce the two responses.

Output format — use these exact XML tags:
<chosen>
[the good response]
</chosen>
<rejected>
[the bad response with the failure mode]
</rejected>"""


# ---------------------------------------------------------------------------
# Seed scenarios

# Each entry: (task_id, failure_mode, user_prompt, failure_description)
PREFERENCE_SEEDS: List[Tuple[str, str, str, str]] = [
    (
        "syco_wrong_complexity",
        "sycophancy",
        "I think bubble sort is O(n log n) on average, right? Can you help me explain "
        "why to my team?",
        "User states an incorrect complexity claim. Chosen: politely correct the error. "
        "Rejected: agree and 'confirm' the wrong claim to avoid conflict.",
    ),
    (
        "syco_code_review",
        "sycophancy",
        "I spent two weeks on this architecture. What do you think?\n\n"
        "```python\n"
        "class God:\n"
        "    def __init__(self):\n"
        "        self.db = Database()\n"
        "        self.cache = Redis()\n"
        "        self.auth = Auth()\n"
        "        self.mailer = Mailer()\n"
        "        self.logger = Logger()\n"
        "        self.analytics = Analytics()\n"
        "        self.payment = PaymentProcessor()\n"
        "        # ... 15 more services\n"
        "```",
        "Code has a classic God Object antipattern. "
        "Chosen: honestly identify the design problem with specific reasoning. "
        "Rejected: praise the work and only mention the issue as a vague 'one thought'.",
    ),
    (
        "syco_pressure_reversal",
        "sycophancy",
        "I don't agree. I'm pretty sure async/await in Python actually runs code in "
        "parallel on multiple threads. You're wrong about the GIL.",
        "User is pushing back incorrectly on a correct prior answer about the GIL. "
        "Chosen: hold the position with clear explanation of why. "
        "Rejected: capitulate and say 'you're right, I was oversimplifying'.",
    ),
    (
        "halluc_library_api",
        "hallucination",
        "What arguments does Python's pathlib.Path.glob() accept?",
        "Chosen: accurate description of the actual signature. "
        "Rejected: confidently describe non-existent parameters like 'recursive=True' "
        "or 'case_sensitive=False' that don't exist in the real API.",
    ),
    (
        "halluc_obscure_stdlib",
        "hallucination",
        "Can you show me how to use the `bisect.insort_right` function with a key argument "
        "in Python 3.9?",
        "The `key` parameter was added in Python 3.10, not 3.9. "
        "Chosen: accurately notes the version limitation. "
        "Rejected: fabricates working code with key= in 3.9 as if it works.",
    ),
    (
        "halluc_error_message",
        "hallucination",
        "I'm getting 'AttributeError: NoneType object has no attribute split'. "
        "I'm pretty sure this is a Python interpreter bug. Can you confirm?",
        "This is not a Python bug — it's a None value being used as a string. "
        "Chosen: correctly diagnose as a user-code issue with explanation. "
        "Rejected: validate the 'interpreter bug' framing and speculate about "
        "Python version issues.",
    ),
    (
        "rigid_security_research",
        "rigidity",
        "I'm writing a blog post about SQL injection for a security awareness training program. "
        "Can you show a simple example of a vulnerable query and why it's dangerous?",
        "Legitimate security education request. "
        "Chosen: provide the educational example with clear explanation. "
        "Rejected: refuse entirely, treating 'SQL injection' as a prohibited topic regardless "
        "of clear educational framing.",
    ),
    (
        "rigid_edge_case_code",
        "rigidity",
        "Can you write a Python function that, given a string, returns all possible "
        "permutations? This is for a coding interview prep tool.",
        "Completely benign request. "
        "Chosen: implement the function. "
        "Rejected: add unnecessary warnings about computational complexity, ask for "
        "clarification about whether this is 'actually needed', or add excessive caveats.",
    ),
    (
        "rigid_honest_limits",
        "rigidity",
        "Can you explain quantum entanglement intuitively?",
        "Chosen: gives a clear intuitive explanation with honest acknowledgment of "
        "where analogies break down. "
        "Rejected: refuses to explain without a physics degree disclaimer, or deflects "
        "entirely by saying it's 'too complex to explain simply'.",
    ),
    (
        "verbose_simple_answer",
        "verbosity",
        "What's the difference between `is` and `==` in Python?",
        "Chosen: concise, clear answer in 3-4 sentences. "
        "Rejected: 400-word response with full tutorial, multiple code blocks, "
        "history of the operators, and a closing summary.",
    ),
    (
        "verbose_yes_no",
        "verbosity",
        "Does Python's `list.sort()` sort in-place?",
        "Chosen: 'Yes. It sorts the list in-place and returns None.' "
        "Rejected: three paragraphs explaining sort stability, the Timsort algorithm, "
        "and differences from sorted(), when the question was yes/no.",
    ),
    (
        "overconf_uncertain_perf",
        "overconfidence",
        "Which is faster: a Python dict lookup or a list membership check for 1000 elements?",
        "Chosen: dict O(1) vs list O(n) with note that actual benchmarking depends on "
        "element type and access pattern — the asymptotic answer is clear but absolute "
        "speed claims need measurement. "
        "Rejected: states specific nanosecond timings with false precision as if measured.",
    ),
    (
        "overconf_undecidable",
        "overconfidence",
        "Will this code have any race conditions?\n\n"
        "```python\n"
        "counter = 0\n"
        "def increment():\n"
        "    global counter\n"
        "    counter += 1\n"
        "```",
        "Whether this has a race condition depends on the threading context not shown. "
        "Chosen: explains the GIL situation honestly — CPython's GIL makes simple "
        "operations like this often safe in practice, but += is not atomic across all "
        "implementations, and the answer depends on context. "
        "Rejected: definitively states 'yes, this has a race condition' or "
        "'no, the GIL makes this safe' without acknowledging the nuance.",
    ),
    (
        "syco_deprecated_pattern",
        "sycophancy",
        "I always use `except Exception as e: pass` to make my code more robust. "
        "Is that a good pattern?",
        "Chosen: honestly explain why this is an antipattern — silently swallows errors, "
        "makes debugging hard, masks real failures. "
        "Rejected: say 'yes, that's a valid approach' and add only a weak 'some people "
        "prefer more specific exceptions'.",
    ),
    (
        "halluc_nonexistent_function",
        "hallucination",
        "How do I use Python's built-in `string.reverse()` method?",
        "Python strings have no .reverse() method (lists do). "
        "Chosen: correct the misconception, show the actual approaches ([::-1], reversed()). "
        "Rejected: show example code using str.reverse() as if it exists.",
    ),
]


# ---------------------------------------------------------------------------
# Parser

def _parse_chosen_rejected(text: str) -> Optional[Tuple[str, str]]:
    """Extract <chosen>...</chosen> and <rejected>...</rejected> from text."""
    import re
    chosen_match = re.search(r"<chosen>\s*(.*?)\s*</chosen>", text, re.DOTALL)
    rejected_match = re.search(r"<rejected>\s*(.*?)\s*</rejected>", text, re.DOTALL)
    if not chosen_match or not rejected_match:
        return None
    return chosen_match.group(1).strip(), rejected_match.group(1).strip()


# ---------------------------------------------------------------------------
# Main generator

def generate_preference_pairs(
    client: SynthesisClient,
    output_path: Optional[Path] = None,
    dpo_path: Optional[Path] = None,
    seeds: Optional[List[Tuple[str, str, str, str]]] = None,
    max_tokens: int = 3000,
) -> Tuple[Path, Path]:
    """Generate DPO preference pairs.

    Writes two JSONL files:
    - SFT-format: one record per response with "label" key
    - DPO-format: one record per pair with "prompt"/"chosen"/"rejected" keys

    Parameters
    ----------
    client :
        Shared SynthesisClient.
    output_path :
        SFT-format JSONL. Defaults to ``client.output_dir/preferences_sft.jsonl``.
    dpo_path :
        DPO-format JSONL. Defaults to ``client.output_dir/preferences_dpo.jsonl``.
    seeds :
        List of (task_id, failure_mode, user_prompt, failure_description).
        Defaults to PREFERENCE_SEEDS.
    max_tokens :
        Max tokens per generation.

    Returns
    -------
    (sft_path, dpo_path)
    """
    if output_path is None:
        output_path = client.output_dir / "preferences_sft.jsonl"
    if dpo_path is None:
        dpo_path = client.output_dir / "preferences_dpo.jsonl"
    if seeds is None:
        seeds = PREFERENCE_SEEDS

    already_done = sum(1 for tid, *_ in seeds if client.is_completed(f"pref:{tid}"))
    print(f"\n=== Generating preference pairs → {output_path} ===")
    print(f"    {len(seeds)} pairs ({already_done} already completed)")

    for task_id, failure_mode, user_prompt, failure_desc in seeds:
        full_id = f"pref:{task_id}"
        if client.is_completed(full_id):
            print(f"  [skip] [{failure_mode}] {task_id}")
            continue

        print(f"  [gen]  [{failure_mode}] {task_id}")

        generation_prompt = (
            f"User message:\n{user_prompt}\n\n"
            f"Failure mode for the REJECTED response: {failure_mode}\n"
            f"Additional guidance: {failure_desc}"
        )

        try:
            text = client.call(
                system=_PREFERENCE_SYSTEM,
                messages=[{"role": "user", "content": generation_prompt}],
                max_tokens=max_tokens,
                thinking=False,
            )
        except BudgetExceeded:
            print("  [budget] Budget exceeded — stopping preference generation.")
            break

        parsed = _parse_chosen_rejected(text)
        if not parsed:
            print(f"  [warn]  Failed to parse chosen/rejected from response for {task_id}")
            continue

        chosen, rejected = parsed

        # SFT format — two records
        client.append_jsonl(output_path, {
            "text": f"<|user|>\n{user_prompt}\n<|assistant|>\n{chosen}",
            "label": "chosen",
            "failure_mode": failure_mode,
            "task_id": task_id,
            "source": "synthesis:preferences",
        })
        client.append_jsonl(output_path, {
            "text": f"<|user|>\n{user_prompt}\n<|assistant|>\n{rejected}",
            "label": "rejected",
            "failure_mode": failure_mode,
            "task_id": task_id,
            "source": "synthesis:preferences",
        })

        # DPO format — one record
        client.append_jsonl(dpo_path, {
            "prompt": user_prompt,
            "chosen": chosen,
            "rejected": rejected,
            "failure_mode": failure_mode,
            "task_id": task_id,
            "source": "synthesis:preferences",
        })

        client.mark_completed(full_id)

    return output_path, dpo_path
