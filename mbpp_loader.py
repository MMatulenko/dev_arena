"""
MBPP Task Loader
================
Loads Python coding tasks from Google's MBPP dataset (HuggingFace).
Each task has a natural-language prompt and 3 assert-based test cases.

The 427 sanitized problems are ordered by task_id, which loosely correlates
with increasing difficulty — making them a natural skill ladder equivalent to
Stockfish level escalation in the chess arena.

Usage:
    from mbpp_loader import load_mbpp_tasks
    tasks = load_mbpp_tasks(start=0, count=50)   # first 50 tasks
    tasks = load_mbpp_tasks(start=50, count=50)  # next 50 (harder)
"""

import logging
import re
import traceback
from functools import lru_cache
from typing import Callable

from tasks import Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loader (cached — only downloads once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_dataset() -> list[dict]:
    """Download and cache the MBPP sanitized split."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info("Downloading MBPP sanitized dataset from HuggingFace...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    rows = [dict(row) for row in ds]
    # Sort by task_id so difficulty escalates predictably
    rows.sort(key=lambda r: r["task_id"])
    logger.info("Loaded %d MBPP tasks", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Test function factory
# ---------------------------------------------------------------------------

def _make_test_fn(task_id: int, test_list: list[str],
                  test_imports: list[str]) -> Callable[[str], tuple[bool, str]]:
    """
    Build a run_tests() closure from MBPP assert statements.
    test_imports handles tasks that need stdlib modules (e.g. math, re).
    """
    def run_tests(code_str: str) -> tuple[bool, str]:
        local_env: dict = {}

        # Execute any required imports into the local environment first
        for imp in test_imports:
            try:
                exec(compile(imp, "<import>", "exec"), local_env)  # noqa: S102
            except Exception:
                pass

        try:
            exec(compile(code_str, "<generated>", "exec"), local_env)  # noqa: S102
        except Exception:
            return False, "Compilation error:\n" + traceback.format_exc()

        passed = 0
        for assertion in test_list:
            try:
                exec(compile(assertion, "<test>", "exec"), dict(local_env))  # noqa: S102
                passed += 1
            except AssertionError:
                # Re-evaluate both sides to give the Coach concrete values
                detail = _eval_assertion_detail(assertion, local_env)
                return (
                    False,
                    "Test failed [task %d]:\n  %s\n%s\n%s"
                    % (task_id, assertion, detail, traceback.format_exc()),
                )
            except Exception:
                return (
                    False,
                    "Runtime error [task %d] on:\n  %s\n\n%s"
                    % (task_id, assertion, traceback.format_exc()),
                )

        return True, "All %d tests passed." % passed

    return run_tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_assertion_detail(assertion: str, env: dict) -> str:
    """
    For 'assert f(x) == expected', evaluate both sides and return a one-liner
    like '  got: [1,2,3]  expected: [1,2,4]' to help the Coach diagnose.
    """
    import re as _re
    m = _re.match(r"assert\s+(.+?)\s*==\s*(.+)$", assertion.strip())
    if not m:
        return ""
    lhs, rhs = m.group(1).strip(), m.group(2).strip()
    try:
        got = eval(lhs, dict(env))  # noqa: S307
        expected = eval(rhs, dict(env))  # noqa: S307
        return "  got: %r\n  expected: %r" % (got, expected)
    except Exception:
        return ""


def _extract_fn_name(test_list: list[str]) -> str | None:
    """
    Pull the function name that tests expect from the first assert statement.
    e.g. 'assert first_repeated_char("abc") == "a"'  →  'first_repeated_char'
    """
    for assertion in test_list:
        m = re.match(r"assert\s+(\w+)\s*\(", assertion.strip())
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mbpp_tasks(start: int = 0, count: int = 50) -> list[Task]:
    """
    Return `count` MBPP tasks starting at index `start` (0-based within
    the sorted sanitized split).

    Args:
        start: Index into the sorted task list (0 = easiest).
        count: Number of tasks to return.
    """
    rows = _load_dataset()
    selected = rows[start: start + count]
    if not selected:
        raise ValueError(
            "No MBPP tasks at start=%d (dataset has %d tasks)" % (start, len(rows))
        )

    tasks: list[Task] = []
    for row in selected:
        task_id      = row["task_id"]
        prompt       = row["prompt"].strip()
        test_list    = row["test_list"]
        test_imports = row.get("test_imports", [])

        # Extract the exact function name from the first test assertion so the
        # Junior Coder doesn't invent its own descriptive name.
        fn_name = _extract_fn_name(test_list)
        if fn_name:
            description = (
                "%s\n\nIMPORTANT: You MUST name your function exactly `%s` "
                "(copy from the test assertions, even if the spelling looks unusual)."
                % (prompt, fn_name)
            )
        else:
            description = prompt

        tasks.append(Task(
            name="mbpp_%d" % task_id,
            description=description,
            run_tests=_make_test_fn(task_id, test_list, test_imports),
        ))

    logger.info(
        "Loaded %d MBPP tasks (indices %d-%d, task_ids %d-%d)",
        len(tasks), start, start + len(tasks) - 1,
        selected[0]["task_id"], selected[-1]["task_id"],
    )
    return tasks


def mbpp_total() -> int:
    """Total number of tasks in the sanitized MBPP split."""
    return len(_load_dataset())
