"""generate_cot.py — Chain-of-thought coding trace generation.

Generates step-by-step reasoning traces for software engineering tasks.
These are the highest-leverage training examples: they teach Chimera not
just *what* to produce but *how to reason* about code problems.

The system prompt is cached across all calls — one cache write, many cache reads.
Output format: {"text": "<|user|>\n...\n<|assistant|>\n..."} in JSONL.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .api_client import BudgetExceeded, SynthesisClient

# ---------------------------------------------------------------------------
# System prompt — stable across all CoT calls, cached

_COT_SYSTEM = """\
You are an expert software engineer with deep knowledge across systems programming, \
algorithms, data structures, and software design. When given a coding problem, you \
reason carefully through it before producing your answer.

Your reasoning process:
1. Restate the problem in your own words to confirm understanding
2. Identify the problem type (debugging, implementation, refactoring, design, explanation)
3. Work through the problem step by step — show intermediate reasoning, not just conclusions
4. For bugs: reproduce the issue mentally, trace through execution, identify root cause
5. For implementations: think about edge cases, complexity, and correctness before writing code
6. For explanations: build understanding from first principles before stating the explanation
7. After reasoning, produce the final clean answer

Response format:
<reasoning>
[Your step-by-step working — this is the most important part]
</reasoning>
<answer>
[Final clean answer, code, or explanation]
</answer>

Be thorough in reasoning, concise in the final answer. Correct is more important than fast."""


# ---------------------------------------------------------------------------
# Task builder helpers

def _debug_prompt(language: str, buggy_code: str, symptom: str) -> str:
    return (
        f"Debug this {language} code. Symptom: {symptom}\n\n"
        f"```{language}\n{buggy_code}\n```"
    )


def _implement_prompt(language: str, signature: str, docstring: str) -> str:
    return (
        f"Implement the following {language} function:\n\n"
        f"```{language}\n{signature}\n    \"\"\"{docstring}\"\"\"\n    pass\n```"
    )


def _explain_prompt(language: str, code: str) -> str:
    return (
        f"Explain this {language} code in detail. What does it do, how does it work, "
        f"and what are the important design decisions?\n\n"
        f"```{language}\n{code}\n```"
    )


def _refactor_prompt(language: str, code: str, goal: str) -> str:
    return (
        f"Refactor this {language} code to {goal}. Preserve all existing behavior.\n\n"
        f"```{language}\n{code}\n```"
    )


def _design_prompt(problem_statement: str) -> str:
    return (
        f"Design a solution for this problem:\n\n{problem_statement}\n\n"
        "Include: data structures, algorithm, time/space complexity, key trade-offs."
    )


# ---------------------------------------------------------------------------
# Seed tasks — a diverse corpus of starting prompts.
# In production these can be expanded by sourcing from real codebases.

SEED_TASKS: List[Tuple[str, str]] = [
    # (task_id, prompt)
    (
        "debug_off_by_one_binary_search",
        _debug_prompt(
            "Python",
            """\
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid - 1
    return -1""",
            "Sometimes loops forever on arrays with two elements",
        ),
    ),
    (
        "debug_integer_overflow_c",
        _debug_prompt(
            "C",
            """\
int sum_array(int *arr, int n) {
    int sum = 0;
    for (int i = 0; i <= n; i++) {
        sum += arr[i];
    }
    return sum;
}""",
            "Segfault on some inputs, wrong answer on others",
        ),
    ),
    (
        "implement_lru_cache",
        _implement_prompt(
            "Python",
            "class LRUCache:",
            "An LRU (Least Recently Used) cache with O(1) get and put.\n"
            "    Args:\n"
            "        capacity: Maximum number of items to store.\n"
            "    Methods:\n"
            "        get(key) -> int: Return value, or -1 if absent. Marks as recently used.\n"
            "        put(key, value): Insert or update. If at capacity, evict LRU item.",
        ),
    ),
    (
        "implement_topological_sort",
        _implement_prompt(
            "Python",
            "def topological_sort(graph: dict[str, list[str]]) -> list[str]:",
            "Return a topological ordering of the DAG (adjacency list). "
            "Raise ValueError if a cycle is detected.",
        ),
    ),
    (
        "explain_python_descriptor_protocol",
        _explain_prompt(
            "Python",
            """\
class ValidatedAttribute:
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f'_{name}'

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f'{self.name} must be numeric')
        setattr(obj, self.private_name, value)

class Point:
    x = ValidatedAttribute()
    y = ValidatedAttribute()

    def __init__(self, x, y):
        self.x = x
        self.y = y""",
        ),
    ),
    (
        "explain_rust_lifetime_elision",
        _explain_prompt(
            "Rust",
            """\
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    &s[..]
}""",
        ),
    ),
    (
        "refactor_nested_conditionals",
        _refactor_prompt(
            "Python",
            """\
def process_user(user):
    if user is not None:
        if user.is_active:
            if user.has_permission('write'):
                if user.quota_remaining > 0:
                    return do_write(user)
                else:
                    return 'quota_exceeded'
            else:
                return 'permission_denied'
        else:
            return 'inactive'
    else:
        return 'no_user'""",
            "reduce nesting and improve readability",
        ),
    ),
    (
        "refactor_extract_duplication",
        _refactor_prompt(
            "Python",
            """\
def send_welcome_email(user):
    msg = f'Subject: Welcome {user.name}\\n\\n'
    msg += f'Dear {user.name},\\nWelcome! Your email: {user.email}.\\n'
    smtp = smtplib.SMTP('smtp.example.com', 587)
    smtp.starttls()
    smtp.login('noreply@example.com', 'password123')
    smtp.sendmail('noreply@example.com', user.email, msg)
    smtp.quit()

def send_password_reset_email(user, token):
    msg = f'Subject: Password Reset\\n\\n'
    msg += f'Dear {user.name},\\nReset: https://example.com/reset/{token}\\n'
    smtp = smtplib.SMTP('smtp.example.com', 587)
    smtp.starttls()
    smtp.login('noreply@example.com', 'password123')
    smtp.sendmail('noreply@example.com', user.email, msg)
    smtp.quit()""",
            "eliminate duplication and separate concerns",
        ),
    ),
    (
        "design_rate_limiter",
        _design_prompt(
            "Design a rate limiter that enforces N requests per second per user. "
            "It must work correctly under concurrent access and handle burst traffic gracefully. "
            "Multiple server instances will share state via Redis."
        ),
    ),
    (
        "design_event_sourcing",
        _design_prompt(
            "Design an event-sourcing system for a bank account. "
            "Support: deposit, withdraw, transfer (atomic across accounts), "
            "point-in-time balance query, and audit log. "
            "Discuss consistency guarantees and failure modes."
        ),
    ),
    (
        "debug_race_condition_go",
        _debug_prompt(
            "Go",
            """\
var cache = make(map[string]string)

func getOrFetch(key string) string {
    if val, ok := cache[key]; ok {
        return val
    }
    val := expensiveFetch(key)
    cache[key] = val
    return val
}""",
            "Random panics: 'concurrent map read and map write'",
        ),
    ),
    (
        "implement_parser_combinator",
        _implement_prompt(
            "Python",
            "def parse_expression(tokens: list[str]) -> tuple[int, list[str]]:",
            "Parse an arithmetic expression using recursive descent. "
            "Supports +, -, *, / with correct precedence and parentheses. "
            "Returns (result_value, remaining_tokens). "
            "Tokens are integers or single-char operators: '3', '+', '(', etc.",
        ),
    ),
    (
        "explain_async_generators_python",
        _explain_prompt(
            "Python",
            """\
async def paginated_fetch(url: str, page_size: int = 100):
    session = aiohttp.ClientSession()
    page = 0
    try:
        while True:
            async with session.get(url, params={'page': page, 'size': page_size}) as resp:
                data = await resp.json()
                if not data['items']:
                    return
                for item in data['items']:
                    yield item
                page += 1
    finally:
        await session.close()

async def process_all():
    async for item in paginated_fetch('https://api.example.com/data'):
        await process_item(item)""",
        ),
    ),
    (
        "design_distributed_lock",
        _design_prompt(
            "Design a distributed lock service. "
            "Requirements: mutual exclusion across N servers, automatic expiry on holder crash, "
            "no split-brain, minimal latency for the common case (lock acquired successfully). "
            "Compare Redlock vs. single-master vs. consensus-based approaches."
        ),
    ),
    (
        "debug_memory_leak_cpp",
        _debug_prompt(
            "C++",
            """\
class EventBus {
    std::vector<std::function<void(Event*)>> listeners;
public:
    void subscribe(std::function<void(Event*)> listener) {
        listeners.push_back(listener);
    }
    void publish(Event* event) {
        for (auto& listener : listeners) { listener(event); }
        delete event;
    }
};

// In a long-running service:
EventBus bus;
void handle_request(Request* req) {
    bus.subscribe([req](Event* e) { req->notify(e); });
    process(req);
}""",
            "Memory usage grows linearly with request count until OOM",
        ),
    ),
    (
        "implement_persistent_segment_tree",
        _implement_prompt(
            "Python",
            "class PersistentSegmentTree:",
            "A persistent segment tree supporting range sum queries. "
            "Each update creates a new version sharing structure with the previous version. "
            "Methods: update(version, idx, val) -> new_version_root; "
            "query(version_root, l, r) -> sum. "
            "Values are non-negative integers, array length fixed at construction.",
        ),
    ),
    (
        "debug_signed_unsigned_c",
        _debug_prompt(
            "C",
            """\
#include <string.h>
void process_buffer(char *buf, int len) {
    for (int i = 0; i < len; i++) {
        if (i < strlen(buf) - 1) {
            process_pair(buf[i], buf[i+1]);
        }
    }
}""",
            "When buf is empty, causes buffer overread instead of doing nothing",
        ),
    ),
    (
        "refactor_command_pattern",
        _refactor_prompt(
            "Python",
            """\
class Editor:
    def __init__(self):
        self.text = ''

    def insert(self, pos, chars):
        self.text = self.text[:pos] + chars + self.text[pos:]

    def delete(self, pos, count):
        self.text = self.text[:pos] + self.text[pos + count:]

    def replace(self, pos, count, chars):
        self.text = self.text[:pos] + chars + self.text[pos + count:]
    # No undo support""",
            "add undo/redo support without changing the public interface",
        ),
    ),
    (
        "design_write_ahead_log",
        _design_prompt(
            "Design a write-ahead log (WAL) for a simple key-value store. "
            "The store must survive crashes without data loss. "
            "Cover: log format, recovery on startup, log compaction, "
            "and the trade-off between durability and write throughput."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Main generator

def generate_cot_traces(
    client: SynthesisClient,
    output_path: Optional[Path] = None,
    tasks: Optional[Sequence[Tuple[str, str]]] = None,
    max_tokens: int = 8192,
) -> Path:
    """Generate chain-of-thought coding traces.

    Parameters
    ----------
    client :
        Shared SynthesisClient.
    output_path :
        Where to write the JSONL output. Defaults to ``client.output_dir/cot_traces.jsonl``.
    tasks :
        List of (task_id, prompt) tuples. Defaults to ``SEED_TASKS``.
    max_tokens :
        Max tokens per response. 8192 gives enough room for deep reasoning traces.

    Returns
    -------
    Path to the written JSONL file.
    """
    if output_path is None:
        output_path = client.output_dir / "cot_traces.jsonl"
    if tasks is None:
        tasks = SEED_TASKS

    already_done = sum(1 for tid, _ in tasks if client.is_completed(f"cot:{tid}"))
    print(f"\n=== Generating CoT traces → {output_path} ===")
    print(f"    {len(tasks)} tasks ({already_done} already completed)")

    for task_id, prompt in tasks:
        full_id = f"cot:{task_id}"
        if client.is_completed(full_id):
            print(f"  [skip] {task_id}")
            continue

        print(f"  [gen]  {task_id}")
        try:
            text = client.call(
                system=_COT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                thinking=True,
            )
        except BudgetExceeded:
            print("  [budget] Budget exceeded — stopping CoT generation.")
            break

        if text:
            record = {
                "text": f"<|user|>\n{prompt}\n<|assistant|>\n{text}",
                "task_type": "cot",
                "task_id": task_id,
                "source": "synthesis:cot",
            }
            client.append_jsonl(output_path, record)
            client.mark_completed(full_id)

    return output_path
