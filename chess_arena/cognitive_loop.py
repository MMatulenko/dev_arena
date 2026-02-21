"""
Cognitive Learning Loop
=======================
Orchestrates N generations of: play → Coach reviews → rewrite SKILLS.md.
Tracks ELO after every game and stops when --budget USD is exhausted.

Run:
  python cognitive_loop.py --budget 5          # run until $5 spent
  python cognitive_loop.py --resume --budget 5 # resume from latest skill
  python cognitive_loop.py --no-fallback       # pure LLM, no minimax endgame
"""

import argparse
import logging
import sys
from pathlib import Path

from config import (
    COGNITIVE_MAX_GENERATIONS,
    COGNITIVE_WIN_STREAK,
    SF_LEVEL_MAX,
    SF_LEVEL_WINS_TO_BUMP,
    SKILLS_DIR,
    STOCKFISH_PATH,
    STOCKFISH_SKILL_LEVEL,
)
from cognitive_arena import SESSION_LEDGER, play_cognitive_match, reflect_and_learn
from elo_tracker import current_elo, load_history, print_elo_chart, update_elo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cognitive_loop")

# ---------------------------------------------------------------------------
# Playbook helpers
# ---------------------------------------------------------------------------

def _load_playbook(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _save_playbook(content: str, version: int) -> Path:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = SKILLS_DIR / f"SKILLS_v{version}.md"
    path.write_text(content, encoding="utf-8")
    logger.info("Playbook v%d saved → %s", version, path)
    return path


def _find_latest_playbook() -> tuple[str, int]:
    existing = sorted(
        SKILLS_DIR.glob("SKILLS_v*.md"),
        key=lambda p: int(p.stem.replace("SKILLS_v", "")),
        reverse=True,
    )
    if existing:
        latest = existing[0]
        version = int(latest.stem.replace("SKILLS_v", ""))
        logger.info("Resuming from %s (version %d)", latest.name, version)
        return _load_playbook(latest), version
    raise FileNotFoundError("No SKILLS_vN.md found in %s" % SKILLS_DIR)


def _load_latest_skill_code() -> str | None:
    py_skills = sorted(
        SKILLS_DIR.glob("skill_v*.py"),
        key=lambda p: int(p.stem.replace("skill_v", "")),
        reverse=True,
    )
    if py_skills:
        code = py_skills[0].read_text(encoding="utf-8")
        logger.info("Using %s as minimax endgame fallback", py_skills[0].name)
        return code
    logger.warning("No minimax skill found — endgame will use random moves")
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(resume: bool = False, use_skill_fallback: bool = True,
        budget_usd: float | None = None):
    """
    Args:
        resume:            start from the latest SKILLS_vN.md
        use_skill_fallback: use minimax for endgame when LLM quota exhausted
        budget_usd:        stop cleanly when session spend reaches this amount
    """
    records: list[dict] = []
    win_streak = 0
    sf_level = STOCKFISH_SKILL_LEVEL
    consecutive_wins_at_level = 0

    if budget_usd:
        logger.info("Budget cap: $%.2f", budget_usd)

    # Bootstrap
    if resume:
        try:
            playbook, current_version = _find_latest_playbook()
        except FileNotFoundError:
            logger.warning("No existing playbook found — starting fresh")
            resume = False

    if not resume:
        v0 = SKILLS_DIR / "SKILLS_v0.md"
        if not v0.exists():
            logger.error("SKILLS_v0.md not found — aborting")
            sys.exit(1)
        playbook = _load_playbook(v0)
        current_version = 0

    fallback_code = _load_latest_skill_code() if use_skill_fallback else None

    history = load_history()
    elo = current_elo(history)
    logger.info("Starting ELO: %d", elo)

    for gen in range(current_version, current_version + COGNITIVE_MAX_GENERATIONS):

        # Budget check before each game
        if budget_usd and SESSION_LEDGER.total_cost >= budget_usd:
            logger.info("Budget of $%.2f reached ($%.4f spent) — stopping.",
                        budget_usd, SESSION_LEDGER.total_cost)
            break

        remaining = (budget_usd - SESSION_LEDGER.total_cost) if budget_usd else None
        logger.info("=" * 65)
        logger.info("GENERATION %d  |  Playbook v%d  |  Stockfish Lvl%d  |  ELO %d%s",
                    gen, current_version, sf_level, elo,
                    "  |  Budget left: $%.2f" % remaining if remaining is not None else "")
        logger.info("=" * 65)

        cost_before = SESSION_LEDGER.total_cost
        try:
            result, pgn, blunders = play_cognitive_match(
                skills_text=playbook,
                fallback_skill_code=fallback_code,
                stockfish_path=STOCKFISH_PATH,
                skill_level=sf_level,
                generation=gen,
            )
        except Exception as exc:
            logger.error("Match failed: %s", exc)
            break

        game_cost = SESSION_LEDGER.total_cost - cost_before

        # ELO update
        elo_before, elo_after, history = update_elo(
            result=result,
            generation=gen,
            playbook_version=current_version,
            stockfish_skill=sf_level,
            game_cost=game_cost,
        )
        elo = elo_after
        delta_str = "%+d" % (elo_after - elo_before)
        logger.info("ELO: %d → %d (%s)  |  spent $%.4f  |  total $%.4f",
                    elo_before, elo_after, delta_str,
                    game_cost, SESSION_LEDGER.total_cost)

        records.append({"generation": gen, "playbook_version": current_version,
                         "result": result, "elo": elo_after, "sf_level": sf_level})
        _log_scoreboard(records)

        if result == "1-0":
            win_streak += 1
            consecutive_wins_at_level += 1
            logger.info("Win streak: %d / %d", win_streak, COGNITIVE_WIN_STREAK)

            # Bump Stockfish difficulty after enough wins at current level
            if consecutive_wins_at_level >= SF_LEVEL_WINS_TO_BUMP and sf_level < SF_LEVEL_MAX:
                sf_level += 1
                consecutive_wins_at_level = 0
                logger.info("Stockfish level bumped to %d — raising the bar!", sf_level)

            if win_streak >= COGNITIVE_WIN_STREAK:
                logger.info("Achieved %d consecutive wins — done.", COGNITIVE_WIN_STREAK)
                break
            continue

        win_streak = 0
        consecutive_wins_at_level = 0

        # Budget check before Coach call
        if budget_usd and SESSION_LEDGER.total_cost >= budget_usd:
            logger.info("Budget exhausted before Coach call — stopping.")
            break

        new_playbook = reflect_and_learn(
            pgn, playbook, result, generation=gen, blunders=blunders
        )
        if new_playbook != playbook:
            current_version = gen + 1
            playbook = new_playbook
            _save_playbook(playbook, current_version)
        else:
            logger.warning("Coach returned unchanged playbook — reusing v%d", current_version)

    _print_summary(records, budget_usd)
    print_elo_chart(load_history())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _log_scoreboard(records: list[dict]):
    wins   = sum(1 for r in records if r["result"] == "1-0")
    losses = sum(1 for r in records if r["result"] == "0-1")
    draws  = sum(1 for r in records if r["result"] == "1/2-1/2")
    logger.info("Scoreboard — W:%d L:%d D:%d  |  Current ELO: %d",
                wins, losses, draws, records[-1]["elo"])


def _print_summary(records: list[dict], budget_usd: float | None):
    wins   = sum(1 for r in records if r["result"] == "1-0")
    losses = sum(1 for r in records if r["result"] == "0-1")
    draws  = sum(1 for r in records if r["result"] == "1/2-1/2")
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║         COGNITIVE ARENA — FINAL SUMMARY             ║")
    logger.info("╠══════════════════════════════════════════════════════╣")
    logger.info("║  Games     : %d", len(records))
    logger.info("║  W / L / D : %d / %d / %d", wins, losses, draws)
    if records:
        logger.info("║  Final ELO : %d  (started at %d, Δ%+d)",
                    records[-1]["elo"],
                    load_history()[0]["elo_before"] if load_history() else 600,
                    records[-1]["elo"] - (load_history()[0]["elo_before"] if load_history() else 600))
    logger.info("║  Total spent: $%.4f%s",
                SESSION_LEDGER.total_cost,
                "  (budget $%.2f)" % budget_usd if budget_usd else "")
    logger.info("╚══════════════════════════════════════════════════════╝")
    for r in records:
        logger.info("  Gen %03d | Playbook v%-3d | %-7s | ELO %d",
                    r["generation"], r["playbook_version"], r["result"], r["elo"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive Chess learning loop")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest SKILLS_vN.md")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable minimax endgame fallback")
    parser.add_argument("--budget", type=float, default=None, metavar="USD",
                        help="Stop when total spend reaches this amount (e.g. --budget 5)")
    args = parser.parse_args()
    run(resume=args.resume,
        use_skill_fallback=not args.no_fallback,
        budget_usd=args.budget)
