# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## What This Project Is

Dev Arena is an AI agent training/evaluation framework. A **Junior Coder** LLM generates code for coding tasks, a **CI Test Runner** executes hidden test suites, and a **Senior Dev Coach** (stronger model) analyzes failures and surgically patches an evolving playbook (`skills/CORE_PRINCIPLES.md`). The playbook is git-versioned inside `skills/` and the Coach uses forced tool use (Anthropic tool_choice="any") to guarantee structured JSON output. A **Librarian** periodically consolidates the playbook to remove duplicates and contradictions.

## Running the System

All modes go through `dev_loop.py`. Key commands:

```zsh path=null start=null
# Single hand-authored task
python dev_loop.py --task average_age --budget 1

# MBPP curriculum (427 tasks, difficulty-escalating)
python dev_loop.py --mbpp --budget 3 --resume

# LeetCode curriculum
python dev_loop.py --leetcode Easy --budget 2

# Evaluate current playbook on held-out tasks (no coaching)
python dev_loop.py --eval --budget 0.20

# Baseline comparison (uses seed playbook, no learning)
python dev_loop.py --mbpp --budget 3 --no-coach --baseline

# Real-world local workspace task
python dev_loop.py --real

# SWE-bench task via Docker container
python dev_loop.py --swe-bench <INSTANCE_ID>

# Consolidate playbook (Librarian)
python dev_loop.py --consolidate

# Duel mode (Agent A vs Agent B with ELO tracking)
python duel_loop.py --mbpp --budget 2
```

Common flags: `--resume` (continue from current playbook), `--sprints N` (max sprints per task), `--budget USD` (dollar spend cap), `--wins N` (consecutive passes before advancing), `--no-coach` (eval-only, one attempt), `--baseline` (use seed playbook).

View playbook evolution history:
```zsh path=null start=null
git -C skills log --oneline
```

## Dependencies

Requires `anthropic`, `openai`, `datasets` (HuggingFace), `python-dotenv`. No `requirements.txt` exists — install manually.

## Configuration

All config lives in `config.py`. Values come from environment variables with sensible defaults. The `.env` file in this directory sets `LLM_PROVIDER`. API keys are loaded from `../chess_skill_arena/.env` (sibling project). Key env vars:

- `LLM_PROVIDER` — `"anthropic"` or `"openai"`
- `ANTHROPIC_MODEL` / `OPENAI_MODEL` — model for the Junior Coder
- `ANTHROPIC_COACH_MODEL` — model for the Senior Dev Coach (always Anthropic)
- `DEV_MAX_SPRINTS`, `DEV_TEST_TIMEOUT`, `DEV_TEMPERATURE`, `DEV_MAX_RETRIES`
- `MBPP_WINS_TO_ADVANCE`, `MBPP_TASK_BATCH`, `MBPP_EVAL_TASK_COUNT`

## Architecture

### Core Loop (`dev_loop.py`)

Orchestrates the sprint cycle: Junior generates code → CI tests → on failure, Coach patches playbook → commit to git → repeat. Supports multiple modes (single task, MBPP, LeetCode, eval, real-world, SWE-bench). The playbook is stored at `skills/CORE_PRINCIPLES.md` and every Coach patch creates a git commit in the `skills/` sub-repo.

### Agents (`dev_arena.py`)

- **`generate_code()`** — Junior Coder reads CORE_PRINCIPLES.md + task description, outputs code in a ```python block.
- **`run_sprint()`** — Generates code then runs `task.run_tests(code)` with a SIGALRM timeout.
- **`reflect_and_learn()`** — Senior Dev Coach (Sonnet) uses forced Anthropic tool use to call one of three tools: `no_update_required`, `tool_upgrade_ticket`, or `playbook_patch`. The `playbook_patch` tool returns structured `ops` (replace/delete) applied via `_apply_ops()`.

### Task System

- **`tasks.py`** — `Task` dataclass with `name`, `description`, `run_tests(code_str) -> (bool, str)`. Hand-authored tasks (average_age, parse_transactions) live here. `TASK_REGISTRY` is the dict of all hand-authored tasks.
- **`mbpp_loader.py`** — Loads from HuggingFace `google-research-datasets/mbpp` sanitized split. Tasks are sorted by `task_id` (difficulty escalation).
- **`leetcode_loader.py`** — Loads from `newfacade/LeetCodeDataset`. Injects helper classes (ListNode, TreeNode) into the test environment. Filters out tasks with unrunnable test harnesses.

### Real-World / SWE-bench Mode

- **`real_world_agent.py`** — Agentic loop where the Junior has tools (file_read, file_write, file_patch, shell, grep, glob_search, run_tests). Supports both Anthropic and OpenAI providers. Uses prompt caching on both.
- **`tools.py`** — Local filesystem tool implementations (the `Tools` class). Includes workspace sandboxing, syntax checking before writes, and partial-match hints on failed patches.
- **`docker_tools.py`** — `DockerTools` class with the same API as `Tools` but executes everything via `docker exec` inside a SWE-bench container. Also has `run_tests()` which runs FAIL_TO_PASS and PASS_TO_PASS pytest suites.
- **`swe_bench_task.py`** — Pulls pre-built SWE-bench Lite Docker images, starts containers, applies test patches, returns `(Task, DockerTools, container_name)`.

### Supporting Components

- **`task_router.py`** — After a Coach patch, extracts concept keywords from the diagnosis and scores remaining tasks by relevance, injecting the most related tasks next into the curriculum queue (failure-targeted routing).
- **`librarian.py`** — Asks an LLM to consolidate the playbook (merge duplicates, resolve contradictions, regroup). Triggered every `LIBRARIAN_EVERY_N` patches.
- **`duel_loop.py`** — Two agents (A and B) with separate playbooks compete on the same tasks. ELO tracking, cross-referencing winner's code in the loser's Coach feedback. State stored in `skills/duel/`.
- **`TOOL_TICKETS.md`** — When the Coach identifies a tool deficiency (not a cognitive error), it issues a `tool_upgrade_ticket` which is auto-appended here for the platform engineer to review.

### Cost Tracking

`CostLedger` in `dev_arena.py` accumulates token usage and dollar cost. `SESSION_LEDGER` is a global singleton. Pricing is defined per-model in `config.py`. Budget enforcement happens in `dev_loop.py` before each sprint and Coach call.

## Adding a New Task

1. Write a `run_tests(code_str: str) -> tuple[bool, str]` function that compiles and tests the generated code.
2. Create a `Task(name=..., description=..., run_tests=...)` instance.
3. Add it to `TASK_REGISTRY` in `tasks.py`.

## Key Patterns

- **LLM calls** go through `_call_llm()` in `dev_arena.py` with automatic retry on 500/529/overloaded errors.
- **Coach always uses forced tool use** (`tool_choice={"type": "any"}`) — never free-text JSON parsing.
- **Playbook patches** are surgical find-and-replace ops, not full rewrites. If `old_text` isn't found, the new text is appended as a fallback.
- **The `skills/` directory is its own git repo**, separate from the parent project. Don't confuse the two.
- **Test timeouts** use `signal.SIGALRM` (Unix only) to kill runaway code.

## Cursor Cloud specific instructions

- **Dependencies**: `requirements.txt` exists. Run `pip install -r requirements.txt` to install all Python deps (the existing AGENTS.md note about no requirements.txt is outdated).
- **API keys**: `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` must be set as environment variables (injected via Cursor Secrets). `LLM_PROVIDER` env var selects which provider to use (`openai` or `anthropic`).
- **Config fix**: `config.py` was missing `OPENAI_MODEL` and `OPENAI_COACH_MODEL` exports that `dev_arena.py` imports. These were added to `config.py` (deriving `OPENAI_MODEL` from `OPENAI_JUNIOR_MODEL` and defaulting `OPENAI_COACH_MODEL` to `gpt-5.2`). Without this fix, the entire codebase fails to import.
- **No formal test suite**: There is no pytest/unittest suite for the framework itself. The project IS a test harness — validation is done by running tasks (`python dev_loop.py --task average_age --budget 0.10`).
- **No linter config**: No `.flake8`, `pyproject.toml`, or `ruff.toml` exists. `ruff` can be used for ad-hoc linting (`python3 -m ruff check *.py`); existing warnings are pre-existing unused imports.
- **The `skills/` directory**: This is a separate git repo from the workspace root. The system auto-initializes it on first run via `_init_git()`. Do not manually modify its git state.
- **Quick smoke test**: `python dev_loop.py --task average_age --budget 0.05 --sprints 1` runs one sprint (~20s, ~$0.001) to verify the full loop works.
