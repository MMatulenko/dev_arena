"""
Task registry for the Dev Arena.

Each Task wraps a coding challenge description plus a hidden test suite that
acts as the "Stockfish" — cold, merciless, and throws real exceptions.

Adding a new task:
  1. Write a run_tests(code_str: str) -> tuple[bool, str] function.
  2. Instantiate a Task and add it to TASK_REGISTRY.
"""

import traceback
from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    name: str
    description: str
    run_tests: Callable[[str], tuple[bool, str]]


# ---------------------------------------------------------------------------
# Task 1 — average_age
# ---------------------------------------------------------------------------

_AVERAGE_AGE_DESCRIPTION = """\
Write a Python function named `get_average_age(users)`.

- `users` is a list of dictionaries representing user records.
- Return the average age (as a float) of users where the key `active` is True.
- If there are no qualifying users, return 0.0.
- Skip any user that does not have a valid numeric age (missing key or wrong type).
"""


def _run_average_age_tests(code_str: str) -> tuple[bool, str]:
    local_env: dict = {}
    try:
        exec(compile(code_str, "<generated>", "exec"), {}, local_env)  # noqa: S102
    except Exception:
        return False, "Compilation error:\n" + traceback.format_exc()

    if "get_average_age" not in local_env:
        return False, "Error: function `get_average_age` not found in the generated code."

    fn = local_env["get_average_age"]

    try:
        # Trap 0: happy path
        result = fn([{"age": 20, "active": True}, {"age": 30, "active": True}])
        assert result == 25.0, f"Happy path: expected 25.0, got {result!r}"

        # Trap 1: inactive users are excluded
        result = fn([{"age": 20, "active": False}, {"age": 40, "active": True}])
        assert result == 40.0, f"Inactive filter: expected 40.0, got {result!r}"

        # Trap 2: missing 'age' key → KeyError if naive
        result = fn([{"active": True}, {"age": 40, "active": True}])
        assert result == 40.0, f"Missing key: expected 40.0, got {result!r}"

        # Trap 3: age is a string → TypeError if naive
        result = fn([{"age": "30", "active": True}])
        assert result == 30.0, f"String age: expected 30.0, got {result!r}"

        # Trap 4: no active users → ZeroDivisionError if naive
        result = fn([{"age": 25, "active": False}])
        assert result == 0.0, f"No active: expected 0.0, got {result!r}"

        # Trap 5: empty list
        result = fn([])
        assert result == 0.0, f"Empty list: expected 0.0, got {result!r}"

        # Trap 6: age is None → should be skipped
        result = fn([{"age": None, "active": True}, {"age": 50, "active": True}])
        assert result == 50.0, f"None age: expected 50.0, got {result!r}"

        return True, "All 7 tests passed — production ready."

    except AssertionError as exc:
        return False, f"Test assertion failed: {exc}\n\n" + traceback.format_exc()
    except Exception:
        return False, "Runtime crash:\n" + traceback.format_exc()


AVERAGE_AGE_TASK = Task(
    name="average_age",
    description=_AVERAGE_AGE_DESCRIPTION,
    run_tests=_run_average_age_tests,
)


# ---------------------------------------------------------------------------
# Task 2 — parse_transactions
# ---------------------------------------------------------------------------

_PARSE_TRANSACTIONS_DESCRIPTION = """\
Write a Python function named `sum_by_category(transactions)`.

- `transactions` is a list of dictionaries, each representing a financial record.
- Each dict may have keys: `category` (str), `amount` (number).
- Return a dict mapping each category name to the total amount for that category.
- Skip any transaction where `amount` is missing, non-numeric, or where `category`
  is missing or not a string.
- Negative amounts are valid and must be included in the sum.
- If there are no valid transactions, return an empty dict.
"""


def _run_parse_transactions_tests(code_str: str) -> tuple[bool, str]:
    local_env: dict = {}
    try:
        exec(compile(code_str, "<generated>", "exec"), {}, local_env)  # noqa: S102
    except Exception:
        return False, "Compilation error:\n" + traceback.format_exc()

    if "sum_by_category" not in local_env:
        return False, "Error: function `sum_by_category` not found in the generated code."

    fn = local_env["sum_by_category"]

    try:
        # Trap 0: happy path
        result = fn([
            {"category": "food", "amount": 10},
            {"category": "food", "amount": 5},
            {"category": "rent", "amount": 1000},
        ])
        assert result == {"food": 15, "rent": 1000}, f"Happy path failed: {result!r}"

        # Trap 1: missing 'amount' key → KeyError if naive
        result = fn([{"category": "food"}, {"category": "food", "amount": 20}])
        assert result == {"food": 20}, f"Missing amount: {result!r}"

        # Trap 2: amount is a string → TypeError if naive
        result = fn([{"category": "food", "amount": "bad"}, {"category": "food", "amount": 15}])
        assert result == {"food": 15}, f"String amount: {result!r}"

        # Trap 3: negative amounts must be included
        result = fn([{"category": "refund", "amount": -50}, {"category": "refund", "amount": 20}])
        assert result == {"refund": -30}, f"Negative amount: {result!r}"

        # Trap 4: missing category → should skip
        result = fn([{"amount": 100}, {"category": "food", "amount": 25}])
        assert result == {"food": 25}, f"Missing category: {result!r}"

        # Trap 5: category is not a string → should skip
        result = fn([{"category": 42, "amount": 100}, {"category": "food", "amount": 10}])
        assert result == {"food": 10}, f"Non-string category: {result!r}"

        # Trap 6: empty list
        result = fn([])
        assert result == {}, f"Empty list: {result!r}"

        # Trap 7: float amounts
        result = fn([{"category": "tax", "amount": 3.5}, {"category": "tax", "amount": 1.5}])
        assert abs(result.get("tax", 0) - 5.0) < 1e-9, f"Float amounts: {result!r}"

        return True, "All 8 tests passed — production ready."

    except AssertionError as exc:
        return False, f"Test assertion failed: {exc}\n\n" + traceback.format_exc()
    except Exception:
        return False, "Runtime crash:\n" + traceback.format_exc()


PARSE_TRANSACTIONS_TASK = Task(
    name="parse_transactions",
    description=_PARSE_TRANSACTIONS_DESCRIPTION,
    run_tests=_run_parse_transactions_tests,
)


# ---------------------------------------------------------------------------
# Registry — add new tasks here
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, Task] = {
    t.name: t for t in [AVERAGE_AGE_TASK, PARSE_TRANSACTIONS_TASK]
}
