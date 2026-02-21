import logging
import sys
from pathlib import Path

from config import (
    MAX_GENERATIONS,
    WIN_STREAK_TO_STOP,
    STOCKFISH_PATH,
    STOCKFISH_SKILL_LEVEL,
    SKILLS_DIR,
)
from arena import play_match
from skill_writer import generate_initial_skill, improve_skill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("learning_loop")

# ---------------------------------------------------------------------------
# Skill persistence helpers
# ---------------------------------------------------------------------------

def _load_skill(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _save_skill(code: str, generation: int) -> Path:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = SKILLS_DIR / f"skill_v{generation}.py"
    path.write_text(code, encoding="utf-8")
    logger.info("Skill v%d saved to %s", generation, path)
    return path


def _find_latest_skill() -> tuple[str, int]:
    """
    Scan skills/ for the highest-numbered skill_vN.py and return (code, N).
    Falls back to skill_v0.py.
    """
    existing = sorted(SKILLS_DIR.glob("skill_v*.py"),
                      key=lambda p: int(p.stem.replace("skill_v", "")),
                      reverse=True)
    if existing:
        latest = existing[0]
        gen = int(latest.stem.replace("skill_v", ""))
        logger.info("Resuming from %s (generation %d)", latest.name, gen)
        return _load_skill(latest), gen
    raise FileNotFoundError("No skill files found in %s" % SKILLS_DIR)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _skill_won(result: str) -> bool:
    return result == "1-0"


def _skill_lost_or_drew(result: str) -> bool:
    return result in ("0-1", "1/2-1/2", "*")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(resume: bool = False):
    """
    Main learning loop.

    Args:
        resume: If True, pick up from the latest saved skill version.
                If False, generate a fresh initial skill from the LLM.
    """
    records: list[dict] = []
    win_streak = 0

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    if resume:
        try:
            skill_code, start_gen = _find_latest_skill()
        except FileNotFoundError:
            logger.warning("No existing skills found — starting fresh")
            resume = False

    if not resume:
        v0_path = SKILLS_DIR / "skill_v0.py"
        if v0_path.exists():
            logger.info("Loading hand-crafted baseline skill_v0.py")
            skill_code = _load_skill(v0_path)
        else:
            logger.info("Generating initial skill from LLM")
            skill_code = generate_initial_skill()
            if skill_code is None:
                logger.error("Could not generate initial skill — aborting")
                return
            _save_skill(skill_code, 0)
        start_gen = 0

    current_skill_version = start_gen

    # ------------------------------------------------------------------
    # Generational loop
    # ------------------------------------------------------------------
    for gen in range(start_gen, start_gen + MAX_GENERATIONS):
        logger.info("=" * 60)
        logger.info("GENERATION %d  |  Skill v%d  |  Stockfish Lvl%d",
                    gen, current_skill_version, STOCKFISH_SKILL_LEVEL)
        logger.info("=" * 60)

        try:
            result, pgn = play_match(
                skill_code=skill_code,
                stockfish_path=STOCKFISH_PATH,
                skill_level=STOCKFISH_SKILL_LEVEL,
                generation=gen,
            )
        except Exception as exc:
            logger.error("Match failed for generation %d: %s", gen, exc)
            break

        records.append({"generation": gen, "skill_version": current_skill_version,
                         "result": result})
        _log_scoreboard(records)

        if _skill_won(result):
            win_streak += 1
            logger.info("Win streak: %d / %d", win_streak, WIN_STREAK_TO_STOP)
            if win_streak >= WIN_STREAK_TO_STOP:
                logger.info("Skill AI achieved %d consecutive wins — stopping.", WIN_STREAK_TO_STOP)
                break
            # After a win, keep the current skill — no rewrite needed
            continue

        # Loss or draw — ask the LLM to improve
        win_streak = 0
        logger.info("Requesting skill improvement from LLM (result=%s)", result)
        new_code = improve_skill(
            current_code=skill_code,
            pgn=pgn,
            result=result,
            skill_level=STOCKFISH_SKILL_LEVEL,
        )

        if new_code is not None:
            current_skill_version = gen + 1
            skill_code = new_code
            _save_skill(skill_code, current_skill_version)
        else:
            logger.warning("Skill improvement failed — reusing current skill v%d",
                           current_skill_version)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    _print_summary(records)


def _log_scoreboard(records: list[dict]):
    wins = sum(1 for r in records if r["result"] == "1-0")
    losses = sum(1 for r in records if r["result"] == "0-1")
    draws = sum(1 for r in records if r["result"] == "1/2-1/2")
    total = len(records)
    logger.info("Scoreboard after %d games — W:%d L:%d D:%d", total, wins, losses, draws)


def _print_summary(records: list[dict]):
    logger.info("")
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║         FINAL SUMMARY                ║")
    logger.info("╠══════════════════════════════════════╣")
    wins = sum(1 for r in records if r["result"] == "1-0")
    losses = sum(1 for r in records if r["result"] == "0-1")
    draws = sum(1 for r in records if r["result"] == "1/2-1/2")
    logger.info("║  Games played : %d", len(records))
    logger.info("║  Wins         : %d", wins)
    logger.info("║  Losses       : %d", losses)
    logger.info("║  Draws        : %d", draws)
    logger.info("╚══════════════════════════════════════╝")
    for r in records:
        logger.info("  Gen %03d | Skill v%-3d | %s",
                    r["generation"], r["skill_version"], r["result"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Skill Chess learning loop")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest saved skill version")
    args = parser.parse_args()
    run(resume=args.resume)
