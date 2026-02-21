"""
Dev Arena — Sprint Loop
=======================
Orchestrates sprints: generate code -> CI tests -> Coach patches DEV_SKILLS.md.

The playbook lives in a single `skills/DEV_SKILLS.md` file whose full history is
tracked automatically via git commits inside `skills/`.  Every time the Coach
patches the playbook a commit is made:

    sprint 7 [mbpp_610]: assumed 0-based index; k is 1-based

Modes:
  --task <name>   Run one hand-authored task
  --all-tasks     Cycle through all hand-authored tasks
  --mbpp          Run MBPP curriculum with difficulty escalation

Run:
  python dev_loop.py --task average_age --budget 1
  python dev_loop.py --mbpp --budget 3
  python dev_loop.py --mbpp --budget 3 --resume
  git -C skills log --oneline          # view playbook history
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from config import (
    BASE_DIR, DEV_MAX_SPRINTS,
    MBPP_EVAL_TASK_COUNT, MBPP_TASK_BATCH, MBPP_WINS_TO_ADVANCE,
    SKILLS_DIR, SKILLS_FILE,
)
from dev_arena import CostLedger, SESSION_LEDGER, reflect_and_learn, run_sprint
from tasks import TASK_REGISTRY, Task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dev_loop")

# ---------------------------------------------------------------------------
# Git helpers — playbook versioning lives in skills/ as a git repo
# ---------------------------------------------------------------------------

def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command inside the skills directory."""
    return subprocess.run(
        ["git", "-C", str(SKILLS_DIR), *args],
        capture_output=True, text=True, check=check,
    )


def _init_git():
    """Ensure skills/ is a git repo with at least one commit."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    result = _git("rev-parse", "--git-dir", check=False)
    if result.returncode != 0:
        _git("init")
        _git("config", "user.email", "dev-arena@local")
        _git("config", "user.name", "Dev Arena")
        logger.info("Initialized git repo in %s", SKILLS_DIR)


def _git_commit(message: str):
    """Stage DEV_SKILLS.md and commit if there are changes."""
    _git("add", "CORE_PRINCIPLES.md")
    # --porcelain is empty when there's nothing to commit
    status = _git("status", "--porcelain")
    if not status.stdout.strip():
        logger.debug("No changes to commit — playbook unchanged")
        return
    result = _git("commit", "-m", message, check=False)
    if result.returncode != 0:
        # e.g. "nothing to commit" even after status showed changes (rare race)
        logger.debug("git commit skipped: %s", result.stderr.strip())
        return
    sha = _git("rev-parse", "--short", "HEAD").stdout.strip()
    logger.info("git commit %s: %s", sha, message)


def _git_log_count() -> int:
    """Number of commits in the playbook history (= number of evolutions)."""
    result = _git("rev-list", "--count", "HEAD", check=False)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def _git_short_sha() -> str:
    result = _git("rev-parse", "--short", "HEAD", check=False)
    return result.stdout.strip() or "init"


# ---------------------------------------------------------------------------
# Playbook helpers
# ---------------------------------------------------------------------------

def _load_playbook() -> str:
    return SKILLS_FILE.read_text(encoding="utf-8")


def _save_playbook(content: str, sprint: int, task_name: str, diagnosis: str):
    """Write CORE_PRINCIPLES.md and auto-commit with a meaningful message."""
    SKILLS_FILE.write_text(content, encoding="utf-8")
    # Keep commit message short: sprint + task + trimmed diagnosis
    short_diag = diagnosis[:80].replace("\n", " ") if diagnosis else "playbook updated"
    message = "sprint %d [%s]: %s" % (sprint, task_name, short_diag)
    _git_commit(message)


def _bootstrap(resume: bool) -> str:
    """Return initial playbook text; set up git repo if needed."""
    _init_git()

    if resume and SKILLS_FILE.exists():
        count = _git_log_count()
        sha = _git_short_sha()
        logger.info("Resuming playbook @ %s (%d evolution(s))", sha, count)
        return _load_playbook()

    # Fresh start — copy the seed playbook
    seed = SKILLS_DIR / "CORE_PRINCIPLES_seed.md"
    if not seed.exists():
        logger.error("Seed file not found: %s", seed)
        sys.exit(1)

    content = seed.read_text(encoding="utf-8")
    SKILLS_FILE.write_text(content, encoding="utf-8")
    _git_commit("init: seed playbook")
    logger.info("Fresh start — seeded from %s", seed.name)
    return content


def _budget_left(budget_usd: float | None) -> str:
    if budget_usd is None:
        return ""
    return "  |  Budget left: $%.2f" % (budget_usd - SESSION_LEDGER.total_cost)


def _budget_exhausted(budget_usd: float | None) -> bool:
    return budget_usd is not None and SESSION_LEDGER.total_cost >= budget_usd


# ---------------------------------------------------------------------------
# Core sprint runner
# ---------------------------------------------------------------------------

def _run_task(task: Task, playbook: str,
              max_sprints: int, budget_usd: float | None,
              sprint_offset: int = 0,
              patch_counter: list[int] | None = None,
              router=None) -> tuple[str, list[dict], bool]:
    """
    Run sprints for one task until it passes, budget runs out, or max_sprints reached.
    Returns (playbook, sprint_records, passed).

    patch_counter: mutable [int] — incremented on each playbook patch so the
                   caller can trigger the Librarian every LIBRARIAN_EVERY_N patches.
    router: optional TaskRouter — receives diagnosis after each patch for
            failure-targeted task injection.
    """
    if patch_counter is None:
        patch_counter = [0]

    records: list[dict] = []

    for sprint in range(1, max_sprints + 1):
        global_sprint = sprint_offset + sprint

        if _budget_exhausted(budget_usd):
            logger.info("Budget of $%.2f reached ($%.4f spent) — stopping.",
                        budget_usd, SESSION_LEDGER.total_cost)
            return playbook, records, False

        sha = _git_short_sha()
        logger.info("=" * 65)
        logger.info("SPRINT %d  |  Playbook @%s  |  Task: %s%s",
                    global_sprint, sha, task.name, _budget_left(budget_usd))
        logger.info("=" * 65)

        sprint_ledger = CostLedger()
        cost_before = SESSION_LEDGER.total_cost

        try:
            success, code, feedback, junior_reasoning = run_sprint(task, playbook, sprint_ledger=sprint_ledger)
        except Exception as exc:
            logger.error("Sprint %d crashed: %s", global_sprint, exc)
            return playbook, records, False

        sprint_cost = SESSION_LEDGER.total_cost - cost_before
        records.append({
            "sprint": global_sprint,
            "task": task.name,
            "playbook_sha": sha,
            "success": success,
            "cost": sprint_cost,
        })

        logger.info("Sprint %d cost: $%.4f | session $%.4f (%s)",
                    global_sprint, sprint_cost,
                    SESSION_LEDGER.total_cost, sprint_ledger.summary())
        _log_scoreboard(records)

        if success:
            logger.info("PASS! '%s' solved on sprint %d.", task.name, global_sprint)
            return playbook, records, True

        tail = "\n".join(feedback.split("\n")[-8:])
        logger.info("CRASH:\n%s", tail)

        if _budget_exhausted(budget_usd):
            logger.info("Budget exhausted before Coach call — stopping.")
            return playbook, records, False

        new_playbook, diagnosis = reflect_and_learn(
            task=task,
            bad_code=code,
            error_traceback=feedback,
            skills_text=playbook,
            generation=global_sprint,
            junior_reasoning=junior_reasoning,
            sprint_ledger=sprint_ledger,
        )

        if new_playbook != playbook:
            playbook = new_playbook
            _save_playbook(playbook, global_sprint, task.name, diagnosis)
            patch_counter[0] += 1

            # Inject related tasks into the curriculum queue
            if router is not None and diagnosis:
                router.inject_from_diagnosis(diagnosis)
        else:
            logger.warning("Coach returned unchanged playbook — no commit")

    return playbook, records, False


# ---------------------------------------------------------------------------
# Mode: single hand-authored task
# ---------------------------------------------------------------------------

def run(task_name: str, resume: bool = False,
        max_sprints: int = DEV_MAX_SPRINTS,
        budget_usd: float | None = None):
    if task_name not in TASK_REGISTRY:
        logger.error("Unknown task '%s'. Available: %s",
                     task_name, ", ".join(TASK_REGISTRY.keys()))
        sys.exit(1)
    if budget_usd:
        logger.info("Budget cap: $%.2f", budget_usd)
    playbook = _bootstrap(resume)
    playbook, records, _ = _run_task(TASK_REGISTRY[task_name], playbook, max_sprints, budget_usd)
    _print_summary(records, budget_usd)


# ---------------------------------------------------------------------------
# Mode: all hand-authored tasks in rotation
# ---------------------------------------------------------------------------

def run_all_tasks(resume: bool = False,
                  max_sprints_per_task: int = DEV_MAX_SPRINTS,
                  budget_usd: float | None = None):
    if budget_usd:
        logger.info("Budget cap: $%.2f — rotating through all tasks", budget_usd)
    playbook = _bootstrap(resume)
    tasks = list(TASK_REGISTRY.values())
    all_records: list[dict] = []
    cycle = 0
    sprint_offset = 0

    while not _budget_exhausted(budget_usd):
        cycle += 1
        logger.info("\n*** CURRICULUM CYCLE %d ***", cycle)
        all_passed = True
        for task in tasks:
            if _budget_exhausted(budget_usd):
                break
            playbook, records, passed = _run_task(
                task, playbook, max_sprints_per_task, budget_usd, sprint_offset,
            )
            all_records.extend(records)
            sprint_offset += len(records)
            if not passed:
                all_passed = False
        if all_passed and not _budget_exhausted(budget_usd):
            logger.info("All tasks passed in cycle %d — next cycle", cycle)

    _print_summary(all_records, budget_usd)


# ---------------------------------------------------------------------------
# Mode: MBPP curriculum with difficulty escalation
# ---------------------------------------------------------------------------

def run_mbpp(resume: bool = False,
             max_sprints_per_task: int = DEV_MAX_SPRINTS,
             budget_usd: float | None = None,
             wins_to_advance: int = MBPP_WINS_TO_ADVANCE,
             batch_size: int = MBPP_TASK_BATCH,
             no_coach: bool = False,
             baseline: bool = False):
    from mbpp_loader import load_mbpp_tasks, mbpp_total
    from task_router import TaskRouter

    mode_tag = []
    if no_coach:
        mode_tag.append("no-coach/eval-only")
    if baseline:
        mode_tag.append("BASELINE/seed-playbook")
    label = " | ".join(mode_tag) if mode_tag else "training"

    if budget_usd:
        logger.info("Budget cap: $%.2f — MBPP [%s]", budget_usd, label)

    _init_git()
    if baseline:
        seed = SKILLS_DIR / "CORE_PRINCIPLES_seed.md"
        if not seed.exists():
            logger.error("Seed file not found: %s", seed)
            sys.exit(1)
        playbook = seed.read_text(encoding="utf-8")
        sha = "seed"
        logger.info("BASELINE mode — using seed playbook (no learned rules)")
    else:
        playbook = _bootstrap(resume)
        sha = _git_short_sha()

    all_records: list[dict] = []
    sprint_offset = 0
    patch_counter = [0]

    total_tasks = mbpp_total()
    train_count = total_tasks - MBPP_EVAL_TASK_COUNT   # last 50 held out for --eval
    all_tasks = load_mbpp_tasks(start=0, count=train_count)
    logger.info("MBPP [%s]: %d tasks | Playbook @%s", label, train_count, sha)
    router = TaskRouter(all_tasks)
    consecutive_wins = 0

    while not _budget_exhausted(budget_usd):
        task = router.next()
        if task is None:
            logger.info("Curriculum complete.")
            break

        logger.info("\n*** MBPP | %s | streak %d/%d | %d remaining ***",
                    task.name, consecutive_wins, wins_to_advance, router.remaining)

        if no_coach:
            if _budget_exhausted(budget_usd):
                break
            sprint_offset += 1
            try:
                success, _code, _fb, _jr = run_sprint(task, playbook)
            except Exception as exc:
                logger.error("Sprint crashed: %s", exc)
                success = False
            all_records.append({
                "sprint": sprint_offset, "task": task.name,
                "playbook_sha": sha, "success": success, "cost": 0.0,
            })
            logger.info("  %s", "PASS" if success else "FAIL")
        else:
            playbook, records, passed = _run_task(
                task, playbook, max_sprints_per_task, budget_usd, sprint_offset,
                patch_counter=patch_counter, router=router,
            )
            all_records.extend(records)
            sprint_offset += len(records)

            if _budget_exhausted(budget_usd):
                break

            if passed:
                consecutive_wins += 1
                if consecutive_wins >= wins_to_advance:
                    consecutive_wins = 0
                    logger.info("*** WIN STREAK %d — advancing to next task ***", wins_to_advance)
            else:
                consecutive_wins = 0

    _print_summary(all_records, budget_usd, mbpp_task_index=router.current_linear_idx)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _log_scoreboard(records: list[dict]):
    passed = sum(1 for r in records if r["success"])
    logger.info("Scoreboard: %d passed / %d sprints | session $%.4f",
                passed, len(records), SESSION_LEDGER.total_cost)


def _print_summary(records: list[dict], budget_usd: float | None,
                   mbpp_task_index: int | None = None):
    passed = sum(1 for r in records if r["success"])
    total = len(records)
    tasks_seen = list(dict.fromkeys(r["task"] for r in records))
    evolutions = _git_log_count() - 1   # subtract the seed commit

    # Pass@1: for each unique task, did it pass on the first attempt?
    task_first_attempt: dict[str, bool] = {}
    for r in records:
        if r["task"] not in task_first_attempt:
            task_first_attempt[r["task"]] = r["success"]
    pass_at_1 = sum(1 for v in task_first_attempt.values() if v)
    total_tasks = len(task_first_attempt)
    pass_at_1_pct = 100 * pass_at_1 / total_tasks if total_tasks else 0

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║         DEV ARENA — FINAL SUMMARY               ║")
    logger.info("╠══════════════════════════════════════════════════╣")
    if mbpp_task_index is not None:
        logger.info("║  MBPP Level : %d tasks covered", mbpp_task_index)
    else:
        logger.info("║  Tasks      : %s", ", ".join(tasks_seen[:5]))
    logger.info("║  Sprints    : %d  (%d unique tasks)", total, total_tasks)
    logger.info("║  Pass@1     : %d/%d  (%.0f%%)", pass_at_1, total_tasks, pass_at_1_pct)
    logger.info("║  All-passes : %d  (%.0f%% of sprints)", passed, 100 * passed / total if total else 0)
    logger.info("║  Evolutions : %d playbook patches (git commits)", evolutions)
    logger.info("║  Total $    : $%.4f%s",
                SESSION_LEDGER.total_cost,
                "  (budget $%.2f)" % budget_usd if budget_usd else "")
    logger.info("║  Tokens     : %s", SESSION_LEDGER.summary())
    logger.info("╚══════════════════════════════════════════════════╝")
    logger.info("  git -C skills log --oneline   # full playbook history")
    logger.info("")
    for r in records[-30:]:
        status = "PASS" if r["success"] else "FAIL"
        logger.info("  Sprint %02d | @%-7s | %-22s | %-4s | $%.4f",
                    r["sprint"], r["playbook_sha"], r["task"][-22:], status, r["cost"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_leetcode(difficulty: str = "Easy",
                 resume: bool = False,
                 max_sprints_per_task: int = DEV_MAX_SPRINTS,
                 budget_usd: float | None = None,
                 wins_to_advance: int = MBPP_WINS_TO_ADVANCE,
                 no_coach: bool = False,
                 baseline: bool = False):
    """
    Run LeetCode tasks as a curriculum (harder than MBPP).

    no_coach: one attempt per task, no coaching, pure pass@1 measurement.
    baseline: use seed playbook instead of evolved one — for A/B comparison.
    """
    from leetcode_loader import load_leetcode_tasks, leetcode_total
    from task_router import TaskRouter

    mode_tag = []
    if no_coach:
        mode_tag.append("no-coach/eval-only")
    if baseline:
        mode_tag.append("BASELINE/seed-playbook")
    label = " | ".join(mode_tag) if mode_tag else "training"

    if budget_usd:
        logger.info("Budget cap: $%.2f — LeetCode %s [%s]", budget_usd, difficulty, label)

    _init_git()
    if baseline:
        seed = SKILLS_DIR / "CORE_PRINCIPLES_seed.md"
        if not seed.exists():
            logger.error("Seed file not found: %s", seed)
            sys.exit(1)
        playbook = seed.read_text(encoding="utf-8")
        sha = "seed"
        logger.info("BASELINE mode — using seed playbook (no learned rules)")
    else:
        playbook = _bootstrap(resume)
        sha = _git_short_sha()

    all_records: list[dict] = []
    sprint_offset = 0
    patch_counter = [0]

    total = leetcode_total(difficulty)
    all_tasks = load_leetcode_tasks(difficulty=difficulty, start=0, count=total)
    router = TaskRouter(all_tasks)
    consecutive_wins = 0

    logger.info("LeetCode %s [%s]: %d tasks | Playbook @%s", difficulty, label, total, sha)

    while not _budget_exhausted(budget_usd):
        task = router.next()
        if task is None:
            logger.info("Curriculum complete.")
            break

        logger.info("\n*** LC-%s | %s | streak %d/%d | %d remaining ***",
                    difficulty.upper(), task.name, consecutive_wins, wins_to_advance, router.remaining)

        if no_coach:
            # Single attempt, no coaching, no playbook updates
            if _budget_exhausted(budget_usd):
                break
            sprint_offset += 1
            try:
                success, _code, _fb, _jr = run_sprint(task, playbook)
            except Exception as exc:
                logger.error("Sprint crashed: %s", exc)
                success = False
            all_records.append({
                "sprint": sprint_offset, "task": task.name,
                "playbook_sha": sha, "success": success, "cost": 0.0,
            })
            logger.info("  %s", "PASS" if success else "FAIL")
        else:
            playbook, records, passed = _run_task(
                task, playbook, max_sprints_per_task, budget_usd, sprint_offset,
                patch_counter=patch_counter, router=router,
            )
            all_records.extend(records)
            sprint_offset += len(records)

            if _budget_exhausted(budget_usd):
                break

            if passed:
                consecutive_wins += 1
                if consecutive_wins >= wins_to_advance:
                    consecutive_wins = 0
                    logger.info("*** WIN STREAK %d — advancing ***", wins_to_advance)
            else:
                consecutive_wins = 0

    _print_summary(all_records, budget_usd)


def run_eval(budget_usd: float | None = None):
    """
    Eval mode: run the Junior on the held-out 50 tasks WITHOUT any coaching
    or playbook updates. Reports pass@1 to measure true generalization.

    Run with: python dev_loop.py --eval [--budget 0.20]
    """
    from mbpp_loader import load_mbpp_tasks, mbpp_total

    _init_git()
    if not SKILLS_FILE.exists():
        logger.error("No DEV_SKILLS.md found — run training first")
        sys.exit(1)

    total_tasks = mbpp_total()
    eval_start = total_tasks - MBPP_EVAL_TASK_COUNT
    eval_tasks = load_mbpp_tasks(start=eval_start, count=MBPP_EVAL_TASK_COUNT)
    playbook = _load_playbook()
    sha = _git_short_sha()

    logger.info("=" * 65)
    logger.info("EVAL MODE — %d held-out tasks | Playbook @%s", len(eval_tasks), sha)
    logger.info("No coaching, no playbook updates — pure generalization test")
    logger.info("=" * 65)

    pass_at_1 = 0
    records: list[dict] = []
    sprint_num = 0

    for task in eval_tasks:
        if budget_usd and SESSION_LEDGER.total_cost >= budget_usd:
            logger.info("Budget reached — stopping eval early.")
            break

        sprint_num += 1
        logger.info("EVAL %d/%d  |  %s", sprint_num, len(eval_tasks), task.name)

        try:
            success, _code, _fb, _jr = run_sprint(task, playbook)
        except Exception as exc:
            logger.error("Eval sprint crashed: %s", exc)
            success = False

        if success:
            pass_at_1 += 1
        records.append({"task": task.name, "success": success})
        logger.info("  %s", "PASS" if success else "FAIL")

    total = len(records)
    pct = 100 * pass_at_1 / total if total else 0
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║            EVAL RESULTS — Pass@1                ║")
    logger.info("╠══════════════════════════════════════════════════╣")
    logger.info("║  Playbook   : @%s (%d evolutions)", sha, _git_log_count() - 1)
    logger.info("║  Tasks eval : %d  (held-out, never trained on)", total)
    logger.info("║  Pass@1     : %d/%d  (%.0f%%)", pass_at_1, total, pct)
    logger.info("║  Total $    : $%.4f", SESSION_LEDGER.total_cost)
    logger.info("╚══════════════════════════════════════════════════╝")
    logger.info("")
    for r in records:
        logger.info("  %-24s  %s", r["task"], "PASS" if r["success"] else "FAIL")


def run_real_world(resume: bool = False, max_sprints: int = 5, budget_usd: float | None = None):
    """Run the RealWorldJunior agent on a realistic task."""
    from real_world_agent import run_real_world_sprint
    from real_world_task import get_real_world_task
    
    _init_git()
    playbook = _bootstrap(resume)
    task, workspace = get_real_world_task()
    
    logger.info("=" * 65)
    logger.info("REAL WORLD MODE — Task: %s | Workspace: %s", task.name, workspace)
    logger.info("=" * 65)
    
    records = []
    
    for sprint in range(1, max_sprints + 1):
        if _budget_exhausted(budget_usd):
            break
            
        if sprint > 1:
            logger.info("Resetting workspace for Sprint %d...", sprint)
            subprocess.run(["git", "-C", str(workspace), "reset", "--hard"], capture_output=True)
            subprocess.run(["git", "-C", str(workspace), "clean", "-fd"], capture_output=True)
            
        logger.info("--- SPRINT %d ---", sprint)
        sprint_ledger = CostLedger()
        cost_before = SESSION_LEDGER.total_cost
        
        # In real-world mode, the Junior runs a loop, and the trajectory is what we log
        try:
            success, trajectory, feedback = run_real_world_sprint(
                task.name, task.description, playbook, workspace, sprint_ledger=sprint_ledger
            )
        except Exception as exc:
            logger.error("RealWorldJunior crashed: %s", exc)
            break
            
        sprint_cost = SESSION_LEDGER.total_cost - cost_before
        records.append({"sprint": sprint, "task": task.name, "success": success, "cost": sprint_cost, "playbook_sha": _git_short_sha()})
        
        logger.info("RealWorld Sprint %d: %s | $%.4f", sprint, "PASS" if success else "FAIL", sprint_cost)
        
        if success:
            logger.info("TASK SOLVED!")
            break
            
        # Call the Coach on failure, passing the trajectory instead of just raw code
        new_playbook, diagnosis = reflect_and_learn(
            task=task,
            bad_code="",  # Not used; trajectory contains everything
            error_traceback=feedback,
            skills_text=playbook,
            generation=sprint,
            junior_reasoning=trajectory,
            sprint_ledger=sprint_ledger
        )
        
        if new_playbook != playbook:
            playbook = new_playbook
            _save_playbook(playbook, sprint, task.name, diagnosis)
            
    _print_summary(records, budget_usd)

def run_swe_bench(instance_id: str, resume: bool = False, max_sprints: int = 5, budget_usd: float | None = None):
    """Run the RealWorldJunior agent on a SWE-bench task using Docker."""
    from real_world_agent import run_real_world_sprint
    from swe_bench_task import setup_swe_bench_task, teardown_swe_bench_task
    
    _init_git()
    playbook = _bootstrap(resume)
    
    instance_ids = [instance_id]
    if instance_id.isdigit():
        logger.info("Loading %s SWE-bench tasks...", instance_id)
        from datasets import load_dataset
        ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
        n = int(instance_id)
        instance_ids = [ds[i]['instance_id'] for i in range(min(n, len(ds)))]

    all_records = []
    
    for current_id in instance_ids:
        if _budget_exhausted(budget_usd):
            break
            
        try:
            task, tools, container_name = setup_swe_bench_task(current_id)
        except Exception as e:
            logger.error("Failed to setup SWE-bench task: %s", e)
            continue
            
        logger.info("=" * 65)
        logger.info("SWE-BENCH MODE — Task: %s | Container: %s", task.name, container_name)
        logger.info("=" * 65)
        
        try:
            for sprint in range(1, max_sprints + 1):
                if _budget_exhausted(budget_usd):
                    break
                    
                if sprint > 1:
                    logger.info("Resetting workspace for Sprint %d...", sprint)
                    tools._exec("git reset --hard")
                    tools._exec("git clean -fd")
                    
                logger.info("--- SPRINT %d ---", sprint)
                sprint_ledger = CostLedger()
                cost_before = SESSION_LEDGER.total_cost
                
                try:
                    # We pass the DockerTools instance directly
                    success, trajectory, feedback = run_real_world_sprint(
                        task.name, task.description, playbook, Path("/testbed"),
                        max_iterations=35, sprint_ledger=sprint_ledger, tools=tools
                    )
                except Exception as exc:
                    logger.error("RealWorldJunior crashed: %s", exc)
                    break
                    
                sprint_cost = SESSION_LEDGER.total_cost - cost_before
                all_records.append({"sprint": sprint, "task": task.name, "success": success, "cost": sprint_cost, "playbook_sha": _git_short_sha()})
                
                logger.info("SWE-bench Sprint %d: %s | $%.4f", sprint, "PASS" if success else "FAIL", sprint_cost)
                
                if success:
                    logger.info("TASK SOLVED (Agent claimed VERIFICATION: PASSED)!")
                    break
                    
                new_playbook, diagnosis = reflect_and_learn(
                    task=task,
                    bad_code="",
                    error_traceback=feedback,
                    skills_text=playbook,
                    generation=sprint,
                    junior_reasoning=trajectory,
                    sprint_ledger=sprint_ledger
                )
                
                if new_playbook != playbook:
                    playbook = new_playbook
                    _save_playbook(playbook, sprint, task.name, diagnosis)
        finally:
            teardown_swe_bench_task(container_name)
        
    _print_summary(all_records, budget_usd)

def run_consolidate():
    """
    On-demand Librarian: read current CORE_PRINCIPLES.md, consolidate it, commit.
    Run with: python dev_loop.py --consolidate
    """
    from librarian import consolidate
    _init_git()
    if not SKILLS_FILE.exists():
        logger.error("No CORE_PRINCIPLES.md found — nothing to consolidate")
        sys.exit(1)
    current = _load_playbook()
    logger.info("Consolidating playbook (%d lines)...", current.count("\n") + 1)
    cleaned, summary = consolidate(current)
    if cleaned == current:
        logger.info("Librarian found nothing to change.")
        return
    SKILLS_FILE.write_text(cleaned, encoding="utf-8")
    _git_commit("librarian: %s" % summary)
    logger.info("Done — %s", summary)
    logger.info("Session: %s", SESSION_LEDGER.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dev Arena sprint loop")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", choices=list(TASK_REGISTRY.keys()),
                      help="Run a single hand-authored task")
    mode.add_argument("--all-tasks", action="store_true",
                      help="Cycle through all hand-authored tasks")
    mode.add_argument("--mbpp", action="store_true",
                      help="Run MBPP curriculum with failure-targeted routing")
    mode.add_argument("--leetcode", choices=["Easy", "Medium", "Hard"],
                      metavar="DIFFICULTY",
                      help="Run LeetCode curriculum at given difficulty (Easy/Medium/Hard)")
    mode.add_argument("--consolidate", action="store_true",
                      help="Run the Librarian once on the current CORE_PRINCIPLES.md and commit")
    mode.add_argument("--eval", action="store_true",
                      help="Evaluate current playbook on held-out tasks (no coaching)")
    mode.add_argument("--real", action="store_true",
                      help="Run the RealWorldJunior on a hand-crafted workspace task")
    mode.add_argument("--swe-bench", type=str, metavar="INSTANCE_ID",
                      help="Run the RealWorldJunior on a specific SWE-bench Lite task via Docker")

    parser.add_argument("--resume", action="store_true",
                        help="Resume from the current CORE_PRINCIPLES.md (preserve git history)")
    parser.add_argument("--sprints", type=int, default=DEV_MAX_SPRINTS,
                        help="Max sprints per task (default: %d)" % DEV_MAX_SPRINTS)
    parser.add_argument("--budget", type=float, default=None, metavar="USD",
                        help="Stop when total spend reaches this amount")
    parser.add_argument("--wins", type=int, default=MBPP_WINS_TO_ADVANCE,
                        help="[MBPP] Consecutive passes before advancing (default: %d)"
                             % MBPP_WINS_TO_ADVANCE)
    parser.add_argument("--no-coach", action="store_true",
                        help="Eval-only: one attempt per task, no coaching or playbook updates")
    parser.add_argument("--baseline", action="store_true",
                        help="Use seed playbook (no learned rules) — for A/B comparison")

    args = parser.parse_args()

    if args.eval:
        run_eval(budget_usd=args.budget)
    elif args.real:
        run_real_world(resume=args.resume, max_sprints=args.sprints, budget_usd=args.budget)
    elif args.swe_bench:
        run_swe_bench(instance_id=args.swe_bench, resume=args.resume, max_sprints=args.sprints, budget_usd=args.budget)
    elif args.leetcode:
        run_leetcode(difficulty=args.leetcode, resume=args.resume,
                     max_sprints_per_task=args.sprints, budget_usd=args.budget,
                     wins_to_advance=args.wins,
                     no_coach=args.no_coach, baseline=args.baseline)
    elif args.consolidate:
        run_consolidate()
    elif args.mbpp:
        run_mbpp(resume=args.resume, max_sprints_per_task=args.sprints,
                 budget_usd=args.budget, wins_to_advance=args.wins,
                 no_coach=args.no_coach, baseline=args.baseline)
    elif args.all_tasks:
        run_all_tasks(resume=args.resume, max_sprints_per_task=args.sprints,
                      budget_usd=args.budget)
    else:
        run(task_name=args.task, resume=args.resume,
            max_sprints=args.sprints, budget_usd=args.budget)
