"""
LeetCode Task Loader
====================
Loads Python coding tasks from the newfacade/LeetCodeDataset on HuggingFace.
2,641 problems (Easy/Medium/Hard) with pre-generated test suites.

Each task's test is a `def check(candidate)` function — the Junior writes a
plain `def method_name(...):` function and we pass it as the candidate.

Usage:
    from leetcode_loader import load_leetcode_tasks, leetcode_total
    tasks = load_leetcode_tasks(difficulty="Easy", start=0, count=50)
    tasks = load_leetcode_tasks(difficulty="Medium", start=0, count=50)
"""

import logging
import re
import traceback
from functools import lru_cache
from typing import Callable

from tasks import Task

logger = logging.getLogger(__name__)

DIFFICULTIES = ("Easy", "Medium", "Hard")

# ---------------------------------------------------------------------------
# Standard LeetCode helper classes/functions injected into every test env
# ---------------------------------------------------------------------------

_LC_HELPERS = """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    def __repr__(self):
        vals, cur = [], self
        while cur and len(vals) < 20:
            vals.append(str(cur.val))
            cur = cur.next
        return "ListNode([" + "->".join(vals) + "])"

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def list_to_linked(lst):
    if not lst:
        return None
    head = ListNode(lst[0])
    cur = head
    for v in lst[1:]:
        cur.next = ListNode(v)
        cur = cur.next
    return head

def linked_to_list(node):
    result, seen = [], set()
    while node and id(node) not in seen:
        seen.add(id(node))
        result.append(node.val)
        node = node.next
    return result

def is_same_list(l1, l2):
    while l1 and l2:
        if l1.val != l2.val:
            return False
        l1, l2 = l1.next, l2.next
    return l1 is None and l2 is None

def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None or p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

# Common lowercase aliases used in some LC test harnesses
def list_node(val=0, next=None):
    return ListNode(val, next)

def tree_node(val=0, left=None, right=None):
    return TreeNode(val, left, right)
"""


@lru_cache(maxsize=1)
def _load_dataset() -> list[dict]:
    """Download and cache the LeetCode dataset, sorted by difficulty then question_id."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info("Downloading LeetCodeDataset from HuggingFace...")
    ds = load_dataset("newfacade/LeetCodeDataset", split="train")
    rows = [dict(row) for row in ds]
    diff_order = {d: i for i, d in enumerate(DIFFICULTIES)}
    rows.sort(key=lambda r: (diff_order.get(r.get("difficulty", "Hard"), 99), r.get("question_id", 0)))
    logger.info("Loaded %d LeetCode tasks", len(rows))
    return rows


def _extract_method_name(entry_point: str) -> str:
    """'Solution().twoSum' → 'twoSum'"""
    m = re.search(r"\.(\w+)$", entry_point)
    return m.group(1) if m else entry_point


def _make_test_fn(task_id: str, check_src: str,
                  method_name: str) -> Callable[[str], tuple[bool, str]]:
    """
    Build a run_tests() closure from LeetCode's check(candidate) function.
    The generated code must define a plain `def method_name(...)` function.
    LC helper classes (ListNode, TreeNode, is_same_list, etc.) are pre-injected.
    """
    def run_tests(code_str: str) -> tuple[bool, str]:
        # Bootstrap env with LC helpers
        env: dict = {}
        exec(compile(_LC_HELPERS, "<lc_helpers>", "exec"), env)  # noqa: S102

        # Compile & exec generated code
        try:
            exec(compile(code_str, "<generated>", "exec"), env)  # noqa: S102
        except Exception:
            return False, "Compilation error:\n" + traceback.format_exc()

        # Find the function by name
        candidate = env.get(method_name)
        if candidate is None:
            return False, (
                "NameError: function `%s` not found in generated code.\n"
                "Make sure the function is defined at module level with that exact name."
                % method_name
            )

        # Load and run the check function
        try:
            exec(compile(check_src, "<check>", "exec"), env)  # noqa: S102
            check_fn = env["check"]
            check_fn(candidate)
        except AssertionError:
            return False, "Test assertion failed [%s]:\n%s" % (task_id, traceback.format_exc())
        except Exception:
            return False, "Runtime error [%s]:\n%s" % (task_id, traceback.format_exc())

        return True, "All tests passed."

    return run_tests


# Names injected via _LC_HELPERS — we can handle any test referencing these
_KNOWN_HELPERS = frozenset({
    "ListNode", "TreeNode", "list_to_linked", "linked_to_list",
    "is_same_list", "is_same_tree", "list_node", "tree_node",
})

# Regex to detect references to unknown undefined helpers in test code
_HELPER_REF = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _test_is_runnable(check_src: str) -> bool:
    """Return False if the test references helpers we cannot provide."""
    defined = set(_HELPER_REF.findall(_LC_HELPERS)) | _KNOWN_HELPERS | {"check", "candidate"}
    referenced = set(_HELPER_REF.findall(check_src))
    unknown = referenced - defined - {"print", "len", "range", "zip", "enumerate",
                                       "sorted", "list", "dict", "set", "tuple",
                                       "int", "str", "float", "bool", "type",
                                       "min", "max", "sum", "abs", "round",
                                       "isinstance", "hasattr", "getattr"}
    # Only block on clearly external custom helpers (ALL_CAPS or snake_case with 'is_' / '_to_')
    bad = {u for u in unknown if re.match(r"is_\w+|to_\w+|\w+_to_\w+", u)}
    return len(bad) == 0


def load_leetcode_tasks(difficulty: str = "Easy",
                        start: int = 0,
                        count: int = 50) -> list[Task]:
    """
    Return `count` LeetCode tasks of the given difficulty starting at `start`.

    Args:
        difficulty: "Easy", "Medium", or "Hard"
        start: 0-based index within the filtered difficulty slice
        count: number of tasks to return
    """
    if difficulty not in DIFFICULTIES:
        raise ValueError("difficulty must be one of: %s" % ", ".join(DIFFICULTIES))

    rows = _load_dataset()
    filtered = [r for r in rows if r.get("difficulty") == difficulty]
    selected = filtered[start: start + count]

    if not selected:
        raise ValueError(
            "No LeetCode %s tasks at start=%d (have %d total)"
            % (difficulty, start, len(filtered))
        )

    tasks: list[Task] = []
    skipped = 0
    for row in selected:
        qid = row.get("question_id", 0)
        check_src = row.get("test", "")
        if not _test_is_runnable(check_src):
            skipped += 1
            continue

        method = _extract_method_name(row.get("entry_point", "solve"))
        description = (
            "%s\n\nIMPORTANT: Write a plain function named exactly `%s` "
            "(not inside a class). The function will be called directly by the tests.\n"
            "Note: ListNode and TreeNode classes are available if needed."
            % (row.get("problem_description", row.get("prompt", "")).strip(), method)
        )
        tasks.append(Task(
            name="lc_%s_%d" % (difficulty[0].lower(), qid),
            description=description,
            run_tests=_make_test_fn("lc_%d" % qid, check_src, method),
        ))

    logger.info(
        "Loaded %d LeetCode %s tasks (indices %d-%d, skipped %d broken)",
        len(tasks), difficulty, start, start + len(tasks) - 1, skipped,
    )
    return tasks


def leetcode_total(difficulty: str = "Easy") -> int:
    """Total number of tasks for a given difficulty level."""
    rows = _load_dataset()
    return sum(1 for r in rows if r.get("difficulty") == difficulty)
