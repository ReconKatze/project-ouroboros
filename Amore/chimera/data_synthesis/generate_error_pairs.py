"""generate_error_pairs.py — Bug/fix coding pair generation.

Generates (buggy_code, fixed_code, explanation) triples across a range of
error types and languages. These train Chimera to:
  - Recognise and reason about defects in code
  - Produce correct fixes, not just identify problems
  - Explain why the fix works (not just what changed)

Output format: {"text": "..."} JSONL where the text is a complete
bug-report → reasoning → fix dialogue.

Error categories:
  logic_error      — wrong algorithm / off-by-one / inverted condition
  resource_leak    — unclosed files/connections, missing free
  concurrency      — race conditions, lock misuse, atomicity violations
  type_error       — implicit coercions, signedness, wrong type assumption
  api_misuse       — calling library functions incorrectly
  security         — injection, buffer overflow, path traversal (educational)
  performance      — O(n²) where O(n) was possible, unnecessary copies
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .api_client import BudgetExceeded, SynthesisClient


_ERROR_SYSTEM = """\
You are an expert software engineer reviewing and fixing code defects.

For each problem you are given:
1. A description of the symptom or bug category
2. Buggy code

You must produce:

<analysis>
Step-by-step reasoning: what is wrong, why it causes the symptom, and what the
correct behaviour should be.
</analysis>

<fix>
The corrected code — complete, not just the changed lines.
</fix>

<explanation>
One short paragraph (2-4 sentences): what changed and why that fixes the problem.
This should be clear enough for a developer unfamiliar with the bug to understand.
</explanation>

Be precise. Show your reasoning. Do not add features or refactor beyond the fix."""


# ---------------------------------------------------------------------------
# Seed bugs

# (task_id, language, category, symptom_description, buggy_code)
BUG_SEEDS: List[Tuple[str, str, str, str, str]] = [
    (
        "logic_binary_search_infinite_loop",
        "Python",
        "logic_error",
        "Binary search sometimes loops forever on two-element arrays",
        """\
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid          # Bug: should be mid + 1
        else:
            right = mid - 1
    return -1""",
    ),
    (
        "logic_inverted_condition",
        "Python",
        "logic_error",
        "Palindrome checker returns True for non-palindromes and False for palindromes",
        """\
def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned != cleaned[::-1]  # Bug: inverted comparison""",
    ),
    (
        "logic_off_by_one_slice",
        "Python",
        "logic_error",
        "Moving average drops the last window and produces one too few values",
        """\
def moving_average(data, window):
    result = []
    for i in range(len(data) - window):  # Bug: should be len(data) - window + 1
        result.append(sum(data[i:i+window]) / window)
    return result""",
    ),
    (
        "resource_file_not_closed",
        "Python",
        "resource_leak",
        "File handles leak when an exception is thrown during processing",
        """\
def process_file(path):
    f = open(path, 'r')
    data = f.read()
    result = expensive_parse(data)   # May raise
    f.close()                         # Never reached on exception
    return result""",
    ),
    (
        "resource_db_connection_leak",
        "Python",
        "resource_leak",
        "Database connections exhausted after many requests",
        """\
def get_user(user_id):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    # Bug: conn never closed on success or exception
    return user""",
    ),
    (
        "concurrency_race_counter",
        "Python",
        "concurrency",
        "Final counter value is lower than expected when incremented from multiple threads",
        """\
import threading

counter = 0

def increment(n):
    global counter
    for _ in range(n):
        counter += 1  # Bug: read-modify-write not atomic under threading

threads = [threading.Thread(target=increment, args=(10000,)) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(counter)  # Expected 40000, often lower""",
    ),
    (
        "concurrency_deadlock",
        "Python",
        "concurrency",
        "Program hangs indefinitely when two threads run concurrently",
        """\
import threading

lock_a = threading.Lock()
lock_b = threading.Lock()

def task_1():
    with lock_a:
        with lock_b:   # Acquires A then B
            do_work()

def task_2():
    with lock_b:
        with lock_a:   # Bug: acquires B then A — circular dependency
            do_work()""",
    ),
    (
        "type_signed_strlen",
        "C",
        "type_error",
        "Off-by-one or buffer overread when string is empty",
        """\
#include <string.h>
void process_pairs(char *buf, int len) {
    for (int i = 0; i < len; i++) {
        if (i < strlen(buf) - 1) {   // Bug: strlen returns size_t (unsigned);
            process(buf[i], buf[i+1]); // when buf is empty, strlen()-1 wraps to SIZE_MAX
        }
    }
}""",
    ),
    (
        "type_integer_overflow_c",
        "C",
        "type_error",
        "Wrong result when array is large; correct for small arrays",
        """\
int array_sum(int *arr, int n) {
    int sum = 0;          // Bug: sum should be long long for large arrays
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}""",
    ),
    (
        "api_dict_default",
        "Python",
        "api_misuse",
        "All keys in the outer dict share the same inner list",
        """\
# Bug: mutable default argument is shared across all calls
def make_graph(nodes, default_edges=[]):
    graph = {}
    for node in nodes:
        graph[node] = default_edges
    return graph""",
    ),
    (
        "api_requests_no_timeout",
        "Python",
        "api_misuse",
        "HTTP calls occasionally hang forever, blocking the thread indefinitely",
        """\
import requests

def fetch_data(url):
    response = requests.get(url)  # Bug: no timeout — hangs if server stalls
    return response.json()""",
    ),
    (
        "api_regex_catastrophic",
        "Python",
        "api_misuse",
        "Regex causes exponential backtracking; hangs on malformed input",
        """\
import re

# Bug: nested quantifiers (a+)+ create catastrophic backtracking
EMAIL_RE = re.compile(r'^([a-zA-Z0-9]+)+@([a-zA-Z0-9]+)+\\.([a-zA-Z]{2,6})+$')

def validate_email(email):
    return bool(EMAIL_RE.match(email))""",
    ),
    (
        "security_sqli_fstring",
        "Python",
        "security",
        "SQL injection vulnerability via unsanitised user input (educational)",
        """\
import sqlite3

def get_user_by_name(db_path, username):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Bug: f-string interpolation allows SQL injection
    cur.execute(f"SELECT * FROM users WHERE name = '{username}'")
    return cur.fetchone()""",
    ),
    (
        "security_path_traversal",
        "Python",
        "security",
        "Path traversal allows reading files outside the intended directory (educational)",
        """\
import os

STATIC_DIR = '/var/www/static'

def serve_file(filename):
    # Bug: no path normalisation — '../../etc/passwd' escapes STATIC_DIR
    path = os.path.join(STATIC_DIR, filename)
    with open(path, 'rb') as f:
        return f.read()""",
    ),
    (
        "performance_quadratic_string",
        "Python",
        "performance",
        "Function is correct but takes O(n²) time due to string concatenation in a loop",
        """\
def join_words(words):
    result = ''
    for word in words:
        result += word + ' '   # Bug: creates a new string on every iteration
    return result.rstrip()""",
    ),
    (
        "performance_repeated_lookup",
        "Python",
        "performance",
        "Correct but O(n²) due to unnecessary repeated list-membership test",
        """\
def find_duplicates(items):
    seen = []
    duplicates = []
    for item in items:
        if item in seen:          # Bug: O(n) lookup per item → O(n²) overall
            if item not in duplicates:
                duplicates.append(item)
        else:
            seen.append(item)
    return duplicates""",
    ),
]


# ---------------------------------------------------------------------------
# Parser

def _parse_sections(text: str) -> Optional[Tuple[str, str, str]]:
    """Extract <analysis>, <fix>, <explanation> from generated text."""
    import re
    analysis = re.search(r"<analysis>\s*(.*?)\s*</analysis>", text, re.DOTALL)
    fix = re.search(r"<fix>\s*(.*?)\s*</fix>", text, re.DOTALL)
    explanation = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
    if not (analysis and fix and explanation):
        return None
    return analysis.group(1).strip(), fix.group(1).strip(), explanation.group(1).strip()


# ---------------------------------------------------------------------------
# Main generator

def generate_error_pairs(
    client: SynthesisClient,
    output_path: Optional[Path] = None,
    seeds: Optional[List[Tuple[str, str, str, str, str]]] = None,
    max_tokens: int = 3000,
) -> Path:
    """Generate bug/fix coding pairs with reasoning traces.

    Parameters
    ----------
    client :
        Shared SynthesisClient.
    output_path :
        JSONL output. Defaults to ``client.output_dir/error_pairs.jsonl``.
    seeds :
        List of (task_id, language, category, symptom, buggy_code).
        Defaults to BUG_SEEDS.
    max_tokens :
        Max tokens per generation.

    Returns
    -------
    Path to the written JSONL file.
    """
    if output_path is None:
        output_path = client.output_dir / "error_pairs.jsonl"
    if seeds is None:
        seeds = BUG_SEEDS

    already_done = sum(1 for tid, *_ in seeds if client.is_completed(f"err:{tid}"))
    print(f"\n=== Generating error pairs → {output_path} ===")
    print(f"    {len(seeds)} bugs ({already_done} already completed)")

    for task_id, language, category, symptom, buggy_code in seeds:
        full_id = f"err:{task_id}"
        if client.is_completed(full_id):
            print(f"  [skip] [{category}] {task_id}")
            continue

        print(f"  [gen]  [{category}] {task_id}")

        user_prompt = (
            f"Language: {language}\n"
            f"Category: {category}\n"
            f"Symptom: {symptom}\n\n"
            f"Buggy code:\n```{language.lower()}\n{buggy_code}\n```"
        )

        try:
            text = client.call(
                system=_ERROR_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                thinking=True,
            )
        except BudgetExceeded:
            print("  [budget] Budget exceeded — stopping error pair generation.")
            break

        parsed = _parse_sections(text)
        if not parsed:
            print(f"  [warn]  Failed to parse sections from response for {task_id}")
            # Fall back to raw text
            record = {
                "text": f"<|user|>\n{user_prompt}\n<|assistant|>\n{text}",
                "task_type": "error_pair",
                "language": language,
                "category": category,
                "task_id": task_id,
                "source": "synthesis:error_pairs",
            }
        else:
            analysis, fix, explanation = parsed
            formatted_response = (
                f"**Analysis:**\n{analysis}\n\n"
                f"**Fixed code:**\n```{language.lower()}\n{fix}\n```\n\n"
                f"**Why this fixes it:**\n{explanation}"
            )
            record = {
                "text": f"<|user|>\n{user_prompt}\n<|assistant|>\n{formatted_response}",
                "task_type": "error_pair",
                "language": language,
                "category": category,
                "task_id": task_id,
                "source": "synthesis:error_pairs",
                "has_structured_output": True,
            }

        client.append_jsonl(output_path, record)
        client.mark_completed(full_id)

    return output_path
