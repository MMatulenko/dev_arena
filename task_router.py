"""
Failure-Targeted Task Router
=============================
After a Coach patch, extracts the concept keywords from the diagnosis and
scores all remaining MBPP tasks by relevance. The highest-scoring tasks are
injected next in the curriculum queue so the AI immediately reinforces what
it just learned, rather than advancing blindly through task_ids.

Usage:
    from task_router import TaskRouter
    router = TaskRouter(all_tasks)           # pass full list of Task objects
    router.inject_from_diagnosis(diagnosis)  # called after each patch
    next_task = router.next()               # returns Task (priority or linear)
"""

import logging
import re
from collections import defaultdict

from tasks import Task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concept → keyword mapping
# ---------------------------------------------------------------------------

_CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "sort":         ["sort", "order", "ascending", "descending", "rank", "arrange"],
    "tuple":        ["tuple", "immutable", "unpack", "pack"],
    "string":       ["string", "str", "substring", "character", "replace", "split", "join",
                     "strip", "upper", "lower", "palindrome", "anagram"],
    "recursion":    ["recursive", "recursion", "fibonacci", "factorial", "base case"],
    "list":         ["list", "array", "element", "index", "append", "extend", "flatten"],
    "dict":         ["dictionary", "dict", "hash", "key", "value", "map", "lookup"],
    "math":         ["sum", "product", "average", "mean", "prime", "factorial", "gcd",
                     "lcm", "modulo", "remainder", "divisible", "digit"],
    "loop":         ["loop", "iterate", "sieve", "counter", "while", "increment"],
    "matrix":       ["matrix", "column", "row", "transpose", "grid", "2d"],
    "type_conv":    ["convert", "coerce", "cast", "int(", "float(", "str(", "type"],
    "signature":    ["parameter", "argument", "signature", "function name", "wrong order"],
    "index":        ["index", "0-based", "1-based", "position", "offset", "k-th", "n-th"],
    "return_type":  ["return type", "returns", "list vs tuple", "int vs float", "string vs int"],
    "regex":        ["regex", "pattern", "match", "search", "re."],
    "set":          ["set", "unique", "distinct", "duplicate", "intersection", "union"],
    "geometry":     ["coordinate", "distance", "point", "vector", "magnitude", "euclidean"],
}


def _score_task(task: Task, keywords: set[str]) -> int:
    text = (task.description + " " + task.name).lower()
    return sum(1 for kw in keywords if kw in text)


def _extract_keywords(diagnosis: str) -> set[str]:
    """Pull concept keywords from a Coach diagnosis string."""
    diag_lower = diagnosis.lower()
    found: set[str] = set()
    for concept, kws in _CONCEPT_KEYWORDS.items():
        for kw in kws:
            if kw in diag_lower:
                found.update(kws)   # inject all siblings so scoring is richer
                break
    return found


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TaskRouter:
    """
    Maintains a linear curriculum (indexed list) plus a priority injection queue.
    Pops from the priority queue first; falls back to the linear index.
    """

    def __init__(self, tasks: list[Task]):
        self._tasks = tasks
        self._linear_idx = 0          # next task in linear order
        self._priority: list[int] = []  # injected task indices (high priority)
        self._seen: set[int] = set()   # indices already returned

    # ------------------------------------------------------------------ #

    def inject_from_diagnosis(self, diagnosis: str, top_n: int = 3):
        """
        Score all not-yet-seen tasks by relevance to the diagnosis and
        inject the top_n highest-scoring tasks into the priority queue.
        """
        keywords = _extract_keywords(diagnosis)
        if not keywords:
            logger.debug("Router: no keywords extracted from diagnosis")
            return

        scores: list[tuple[int, int]] = []  # (score, idx)
        for idx, task in enumerate(self._tasks):
            if idx in self._seen or idx in self._priority:
                continue
            if idx < self._linear_idx:
                continue  # already passed linearly
            s = _score_task(task, keywords)
            if s > 0:
                scores.append((s, idx))

        scores.sort(key=lambda x: -x[0])
        injected = [idx for _, idx in scores[:top_n]]

        if injected:
            logger.info("Router: injecting %d related tasks after diagnosis: %s",
                        len(injected), [self._tasks[i].name for i in injected])
            # Prepend to priority queue (highest-score first)
            self._priority = injected + self._priority

    def next(self) -> Task | None:
        """Return next Task (priority first, then linear). None if exhausted."""
        # Drain stale priority entries
        while self._priority and self._priority[0] in self._seen:
            self._priority.pop(0)

        if self._priority:
            idx = self._priority.pop(0)
            self._seen.add(idx)
            logger.info("Router: priority task → %s (idx %d)", self._tasks[idx].name, idx)
            return self._tasks[idx]

        # Linear advance (skip already-seen)
        while self._linear_idx < len(self._tasks):
            idx = self._linear_idx
            self._linear_idx += 1
            if idx not in self._seen:
                self._seen.add(idx)
                return self._tasks[idx]

        return None  # curriculum complete

    @property
    def remaining(self) -> int:
        return len(self._tasks) - len(self._seen)

    @property
    def current_linear_idx(self) -> int:
        return self._linear_idx
