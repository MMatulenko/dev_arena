"""
Dev Arena — Duel Mode (Agent A vs Agent B)
==========================================
Two Junior Coders with separate playbooks (DEV_SKILLS_A.md / DEV_SKILLS_B.md)
compete on the same MBPP tasks simultaneously.

Rules:
  - Both generate code for the same task.
  - Both are tested against the same hidden trap suite.
  - First to pass = winner of that sprint. Loser's Coach patches their playbook,
    optionally cross-referencing the winner's code.
  - Both fail → the one with fewer test errors "wins" (most tests passed conceptually).
    Both coaches are called with the opponent's code as a reference.
  - ELO tracks performance across all sprints.
  - Librarian runs every LIBRARIAN_EVERY_N patches on each agent's playbook.

Run:
  python duel_loop.py --mbpp --budget 2
  python duel_loop.py --mbpp --budget 2 --resume
  python duel_loop.py --task average_age --budget 0.5
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from config import (
    BASE_DIR,
    DEV_MAX_SPRINTS,
    DUEL_ELO_K,
    DUEL_ELO_START,
    DUEL_SKILLS_DIR,
    LIBRARIAN_EVERY_N,
    MBPP_WINS_TO_ADVANCE,
    SKILLS_DIR,
)
from dev_arena import (
    CostLedger,
    SESSION_LEDGER,
    generate_code,
    reflect_and_learn,
)
from librarian import consolidate, should_run
from tasks import TASK_REGISTRY, Task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("duel_loop")

# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------

_ELO_FILE = DUEL_SKILLS_DIR / "duel_elo.json"


def _load_elo() -> dict:
    if _ELO_FILE.exists():
        return json.loads(_ELO_FILE.read_text())
    return {"A": DUEL_ELO_START, "B": DUEL_ELO_START, "history": []}


def _save_elo(data: dict):
    _ELO_FILE.write_text(json.dumps(data, indent=2))


def _update_elo(data: dict, winner: str, sprint: int, task: str) -> tuple[int, int]:
    """Update ELO; winner='A', 'B', or 'draw'. Returns (new_elo_a, new_elo_b)."""
    ea, eb = data["A"], data["B"]
    exp_a = 1 / (1 + 10 ** ((eb - ea) / 400))
    exp_b = 1 - exp_a

    if winner == "A":
        sa, sb = 1.0, 0.0
    elif winner == "B":
        sa, sb = 0.0, 1.0
    else:
        sa, sb = 0.5, 0.5

    data["A"] = round(ea + DUEL_ELO_K * (sa - exp_a))
    data["B"] = round(eb + DUEL_ELO_K * (sb - exp_b))
    data["history"].append({
        "sprint": sprint, "task": task, "winner": winner,
        "elo_a": data["A"], "elo_b": data["B"],
    })
    return data["A"], data["B"]


# ---------------------------------------------------------------------------
# Git helpers (shared repo, two files)
# ---------------------------------------------------------------------------

def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(DUEL_SKILLS_DIR), *args],
        capture_output=True, text=True, check=check,
    )


def _init_git():
    DUEL_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    result = _git("rev-parse", "--git-dir", check=False)
    if result.returncode != 0:
        _git("init")
        _git("config", "user.email", "duel-arena@local")
        _git("config", "user.name", "Duel Arena")
        logger.info("Initialized duel git repo in %s", DUEL_SKILLS_DIR)


def _git_commit(agent: str, message: str):
    filename = "DEV_SKILLS_%s.md" % agent.upper()
    _git("add", filename, "duel_elo.json", check=False)
    status = _git("status", "--porcelain")
    if not status.stdout.strip():
        return
    result = _git("commit", "-m", "[%s] %s" % (agent.upper(), message), check=False)
    if result.returncode == 0:
        sha = _git("rev-parse", "--short", "HEAD").stdout.strip()
        logger.info("git [%s] %s — %s", agent.upper(), sha, message)


def _git_count(agent: str) -> int:
    filename = "DEV_SKILLS_%s.md" % agent.upper()
    result = _git("log", "--oneline", "--", filename, check=False)
    return len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0


def _git_sha(agent: str) -> str:
    result = _git("rev-parse", "--short", "HEAD", check=False)
    return result.stdout.strip() or "init"


# ---------------------------------------------------------------------------
# Playbook helpers
# ---------------------------------------------------------------------------

def _skill_file(agent: str) -> Path:
    return DUEL_SKILLS_DIR / ("DEV_SKILLS_%s.md" % agent.upper())


def _seed_file(agent: str) -> Path:
    return DUEL_SKILLS_DIR / ("DEV_SKILLS_%s_seed.md" % agent.upper())


def _load(agent: str) -> str:
    return _skill_file(agent).read_text(encoding="utf-8")


def _save(agent: str, content: str, sprint: int, task_name: str, diagnosis: str):
    _skill_file(agent).write_text(content, encoding="utf-8")
    short_diag = diagnosis[:72].replace("\n", " ") if diagnosis else "updated"
    _git_commit(agent, "sprint %d [%s]: %s" % (sprint, task_name, short_diag))


def _bootstrap_agent(agent: str, resume: bool) -> str:
    _init_git()
    skill = _skill_file(agent)
    seed = _seed_file(agent)

    if resume and skill.exists():
        logger.info("Agent %s: resuming from %s", agent.upper(), skill.name)
        return _load(agent)

    if not seed.exists():
        # Auto-generate a divergent seed from the shared seed
        shared_seed = SKILLS_DIR / "DEV_SKILLS_seed.md"
        if not shared_seed.exists():
            logger.error("No seed file for agent %s and no shared seed", agent.upper())
            sys.exit(1)
        base = shared_seed.read_text(encoding="utf-8")
        if agent.upper() == "A":
            header = "# Agent A Playbook — Aggressive / Try-First\n\n"
            style = ("- When uncertain about edge cases, write the simplest working "
                     "solution first and let tests reveal the gaps.\n")
        else:
            header = "# Agent B Playbook — Defensive / Guard-First\n\n"
            style = ("- Before writing the main logic, identify all edge cases "
                     "(None, empty, wrong type, boundary values) and handle them first.\n")
        content = header + base.replace("# My Developer Playbook — Generation 0\n", "") + style
        seed.write_text(content, encoding="utf-8")

    content = seed.read_text(encoding="utf-8")
    skill.write_text(content, encoding="utf-8")
    _git_commit(agent, "init: seed playbook")
    logger.info("Agent %s: fresh start from seed", agent.upper())
    return content


# ---------------------------------------------------------------------------
# Duel sprint
# ---------------------------------------------------------------------------

def _budget_exhausted(budget_usd: float | None) -> bool:
    return budget_usd is not None and SESSION_LEDGER.total_cost >= budget_usd


def _budget_left(budget_usd: float | None) -> str:
    if budget_usd is None:
        return ""
    return "  |  Budget $%.2f left" % (budget_usd - SESSION_LEDGER.total_cost)


def _run_duel_sprint(
    task: Task,
    playbook_a: str,
    playbook_b: str,
    sprint: int,
    budget_usd: float | None,
) -> tuple[str, str, str, str, str]:
    """
    Both agents generate code and are tested.
    Returns (new_playbook_a, new_playbook_b, winner, diag_a, diag_b).
    """
    logger.info("=" * 70)
    sha_a = _git_sha("a")
    logger.info("DUEL SPRINT %d | Task: %s%s", sprint, task.name, _budget_left(budget_usd))
    logger.info("  Agent A @%s vs Agent B", sha_a)
    logger.info("=" * 70)

    ledger_a, ledger_b = CostLedger(), CostLedger()

    # Both generate code
    code_a, reasoning_a = generate_code(task, playbook_a, sprint_ledger=ledger_a)
    if _budget_exhausted(budget_usd):
        return playbook_a, playbook_b, "draw", "", ""
    code_b, reasoning_b = generate_code(task, playbook_b, sprint_ledger=ledger_b)

    logger.info("Agent A code:\n%s", code_a)
    logger.info("Agent B code:\n%s", code_b)

    # Test both (with timeout inherited from run_sprint → SIGALRM is set per call)
    import signal
    from dev_arena import _alarm_handler, _CodeTimeout, DEV_TEST_TIMEOUT

    def _safe_test(code: str) -> tuple[bool, str]:
        old = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(DEV_TEST_TIMEOUT)
        try:
            ok, fb = task.run_tests(code)
        except _CodeTimeout:
            ok, fb = False, "TIMEOUT: infinite loop detected"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        return ok, fb

    ok_a, fb_a = _safe_test(code_a)
    ok_b, fb_b = _safe_test(code_b)

    logger.info("A: %s  |  B: %s", "PASS" if ok_a else "FAIL", "PASS" if ok_b else "FAIL")

    # Determine winner
    if ok_a and not ok_b:
        winner = "A"
    elif ok_b and not ok_a:
        winner = "B"
    elif ok_a and ok_b:
        winner = "draw"
    else:
        winner = "draw"   # both fail — both get coached

    new_a, new_b = playbook_a, playbook_b
    diag_a = diag_b = ""

    # Coach the loser(s), using the winner's code as reference
    if _budget_exhausted(budget_usd):
        return new_a, new_b, winner, diag_a, diag_b

    if winner == "A" or winner == "draw":
        # Coach B — reference A's code in the error feedback
        b_feedback = fb_b + "\n\n[Reference: Agent A passed with this code]\n" + code_a
        new_b, diag_b = reflect_and_learn(
            task=task, bad_code=code_b, error_traceback=b_feedback,
            skills_text=playbook_b, generation=sprint,
            junior_reasoning=reasoning_b, sprint_ledger=ledger_b,
        )

    if winner == "B" or winner == "draw":
        # Coach A — reference B's code
        a_feedback = fb_a + "\n\n[Reference: Agent B passed with this code]\n" + code_b
        new_a, diag_a = reflect_and_learn(
            task=task, bad_code=code_a, error_traceback=a_feedback,
            skills_text=playbook_a, generation=sprint,
            junior_reasoning=reasoning_a, sprint_ledger=ledger_a,
        )

    return new_a, new_b, winner, diag_a, diag_b


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_duel(task_names: list[str] | None = None,
             use_mbpp: bool = False,
             resume: bool = False,
             max_sprints_per_task: int = DEV_MAX_SPRINTS,
             budget_usd: float | None = None,
             wins_to_advance: int = MBPP_WINS_TO_ADVANCE):

    playbook_a = _bootstrap_agent("a", resume)
    playbook_b = _bootstrap_agent("b", resume)
    elo_data = _load_elo()

    if budget_usd:
        logger.info("Duel budget: $%.2f | ELO A=%d B=%d",
                    budget_usd, elo_data["A"], elo_data["B"])

    # Build task list
    if use_mbpp:
        from mbpp_loader import load_mbpp_tasks, mbpp_total
        from task_router import TaskRouter
        from config import MBPP_EVAL_TASK_COUNT
        train_count = mbpp_total() - MBPP_EVAL_TASK_COUNT
        all_tasks = load_mbpp_tasks(start=0, count=train_count)
        logger.info("Duel training on %d tasks | %d held out for eval", train_count, MBPP_EVAL_TASK_COUNT)
        router = TaskRouter(all_tasks)
    else:
        tasks = [TASK_REGISTRY[n] for n in (task_names or list(TASK_REGISTRY.keys()))]
        router = None

    sprint = 0
    wins_a = wins_b = 0
    consecutive_wins: dict[str, int] = {"A": 0, "B": 0, "draw": 0}

    all_records: list[dict] = []

    # Patch commit counters (for librarian trigger)
    patches_a = patches_b = 0

    while not _budget_exhausted(budget_usd):
        # Pick next task
        if use_mbpp and router is not None:
            task = router.next()
            if task is None:
                logger.info("Curriculum complete.")
                break
        else:
            # Cycle through task list
            task_idx = sprint % len(tasks)
            task = tasks[task_idx]

        sprint += 1
        new_a, new_b, winner, diag_a, diag_b = _run_duel_sprint(
            task, playbook_a, playbook_b, sprint, budget_usd
        )

        # Persist changes
        if new_a != playbook_a:
            playbook_a = new_a
            _save("a", playbook_a, sprint, task.name, diag_a)
            patches_a += 1
            if use_mbpp and router is not None and diag_a:
                router.inject_from_diagnosis(diag_a)
            if should_run(patches_a, LIBRARIAN_EVERY_N):
                logger.info("Librarian running for Agent A...")
                cleaned, summary = consolidate(playbook_a)
                if cleaned != playbook_a:
                    playbook_a = cleaned
                    _save("a", playbook_a, sprint, task.name,
                          "librarian: %s" % summary)

        if new_b != playbook_b:
            playbook_b = new_b
            _save("b", playbook_b, sprint, task.name, diag_b)
            patches_b += 1
            if use_mbpp and router is not None and diag_b:
                router.inject_from_diagnosis(diag_b)
            if should_run(patches_b, LIBRARIAN_EVERY_N):
                logger.info("Librarian running for Agent B...")
                cleaned, summary = consolidate(playbook_b)
                if cleaned != playbook_b:
                    playbook_b = cleaned
                    _save("b", playbook_b, sprint, task.name,
                          "librarian: %s" % summary)

        # ELO
        elo_a, elo_b = _update_elo(elo_data, winner, sprint, task.name)
        _save_elo(elo_data)

        # Consecutive wins tracking (for task advancement in mbpp mode)
        if winner in ("A", "B"):
            consecutive_wins[winner] += 1
            consecutive_wins["draw"] = 0
        else:
            consecutive_wins = {"A": 0, "B": 0, "draw": 0}

        all_records.append({
            "sprint": sprint, "task": task.name, "winner": winner,
            "elo_a": elo_a, "elo_b": elo_b,
        })

        logger.info("Sprint %d | Winner: %-4s | ELO A=%d B=%d | $%.4f spent",
                    sprint, winner.upper(), elo_a, elo_b, SESSION_LEDGER.total_cost)
        _print_scoreboard(all_records)

    _print_summary(all_records, elo_data, budget_usd, patches_a, patches_b)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_scoreboard(records: list[dict]):
    wins_a = sum(1 for r in records if r["winner"] == "A")
    wins_b = sum(1 for r in records if r["winner"] == "B")
    draws   = sum(1 for r in records if r["winner"] == "draw")
    if records:
        latest = records[-1]
        logger.info("  Scoreboard: A=%d  B=%d  Draw=%d  |  ELO A=%d B=%d",
                    wins_a, wins_b, draws, latest["elo_a"], latest["elo_b"])


def _print_summary(records: list[dict], elo_data: dict, budget_usd: float | None,
                   patches_a: int, patches_b: int):
    wins_a = sum(1 for r in records if r["winner"] == "A")
    wins_b = sum(1 for r in records if r["winner"] == "B")
    draws   = sum(1 for r in records if r["winner"] == "draw")
    logger.info("")
    logger.info("╔═══════════════════════════════════════════════════════╗")
    logger.info("║            DUEL ARENA — FINAL SUMMARY                ║")
    logger.info("╠═══════════════════════════════════════════════════════╣")
    logger.info("║  Sprints      : %d", len(records))
    logger.info("║  Agent A wins : %d  |  patches: %d", wins_a, patches_a)
    logger.info("║  Agent B wins : %d  |  patches: %d", wins_b, patches_b)
    logger.info("║  Draws        : %d", draws)
    logger.info("║  Final ELO    : A=%d  B=%d", elo_data["A"], elo_data["B"])
    logger.info("║  Total $      : $%.4f%s",
                SESSION_LEDGER.total_cost,
                "  (budget $%.2f)" % budget_usd if budget_usd else "")
    logger.info("║  Tokens       : %s", SESSION_LEDGER.summary())
    logger.info("╚═══════════════════════════════════════════════════════╝")
    logger.info("  git -C skills/duel log --oneline -- DEV_SKILLS_A.md  # A history")
    logger.info("  git -C skills/duel log --oneline -- DEV_SKILLS_B.md  # B history")
    logger.info("")
    for r in records[-20:]:
        logger.info("  Sprint %03d | %-24s | %-4s | A=%d B=%d",
                    r["sprint"], r["task"][-24:], r["winner"].upper(),
                    r["elo_a"], r["elo_b"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dev Arena — Duel Mode A vs B")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mbpp", action="store_true",
                      help="Run MBPP curriculum with failure-targeted routing")
    mode.add_argument("--task", nargs="+", choices=list(TASK_REGISTRY.keys()),
                      help="Run specific hand-authored task(s)")
    mode.add_argument("--all-tasks", action="store_true",
                      help="Cycle all hand-authored tasks")

    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing DEV_SKILLS_A/B.md")
    parser.add_argument("--budget", type=float, default=None, metavar="USD")
    parser.add_argument("--sprints", type=int, default=DEV_MAX_SPRINTS,
                        help="Max sprints per task in hand-authored mode")
    parser.add_argument("--wins", type=int, default=MBPP_WINS_TO_ADVANCE,
                        help="[MBPP] consecutive passes before advancing")

    args = parser.parse_args()

    task_names = None
    if args.all_tasks:
        task_names = list(TASK_REGISTRY.keys())
    elif hasattr(args, "task") and args.task:
        task_names = args.task

    run_duel(
        task_names=task_names,
        use_mbpp=args.mbpp,
        resume=args.resume,
        max_sprints_per_task=args.sprints,
        budget_usd=args.budget,
        wins_to_advance=args.wins,
    )
