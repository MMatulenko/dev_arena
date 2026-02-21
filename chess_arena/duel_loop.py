"""
Duel Learning Loop — Agent A vs Agent B
=========================================
Two LLM agents with separate SKILLS.md playbooks evolve by playing each other.
Stockfish analyses every move (guidance only — never plays).
The losing agent's Coach patches their playbook with a surgical JSON diff.
Colors alternate each generation for fair ELO.

Run:
  python duel_loop.py --budget 5          # run until $5 spent
  python duel_loop.py --resume            # resume from latest saved versions
  python duel_loop.py --no-fallback       # pure LLM, no minimax endgame

Monitor:
  tail -f /tmp/duel.log
  grep -E "(Gen |ELO|Coach|Budget)" /tmp/duel.log
"""

import argparse
import logging
import sys
from pathlib import Path

import chess

from config import (
    DUEL_MAX_GENERATIONS,
    DUEL_LLM_MOVES,
    DUEL_SKILLS_DIR,
    STOCKFISH_PATH,
)
from duel_arena import SESSION_LEDGER, coach_patch, play_duel
from duel_elo import current_elos, load_history, print_duel_chart, update_elos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("duel_loop")


# ---------------------------------------------------------------------------
# Playbook helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _playbook_path(agent: str) -> Path:
    """Single canonical file per agent — always overwritten in place."""
    return DUEL_SKILLS_DIR / ("SKILLS_%s.md" % agent.upper())


def _save(content: str, agent: str, version: int) -> Path:
    DUEL_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = _playbook_path(agent)
    path.write_text(content, encoding="utf-8")
    logger.info("Playbook %s updated (v%d) → %s", agent.upper(), version, path)
    return path


def _find_latest(agent: str) -> tuple[str, int]:
    """Load the single playbook file; fall back to seed if not yet created."""
    path = _playbook_path(agent)
    if path.exists():
        logger.info("Loading %s playbook from %s", agent.upper(), path.name)
        return _load(path), 0  # version tracked in memory only

    # First run — copy seed into the canonical file
    seed = DUEL_SKILLS_DIR / ("SKILLS_%s_v0.md" % agent.upper())
    if seed.exists():
        content = _load(seed)
        path.write_text(content, encoding="utf-8")
        logger.info("Initialized %s playbook from seed → %s", agent.upper(), path.name)
        return content, 0
    raise FileNotFoundError("No playbook found for agent %s. Expected %s" % (agent, path))


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------

def _log_scoreboard(records: list[dict]):
    if not records:
        return
    wins_a = sum(1 for r in records if r["winner"] == "A")
    wins_b = sum(1 for r in records if r["winner"] == "B")
    draws   = sum(1 for r in records if r["winner"] == "draw")
    logger.info("  Scoreboard: A=%d  B=%d  Draw=%d  (total %d games)",
                wins_a, wins_b, draws, len(records))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(resume: bool = False, budget_usd: float | None = None):

    DUEL_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    playbook_a, version_a = _find_latest("a")
    playbook_b, version_b = _find_latest("b")

    history = load_history()
    elo_a, elo_b = current_elos(history)

    records: list[dict] = []
    start_gen = len(history.get("a", [])) + 1

    if budget_usd:
        logger.info("Budget cap: $%.2f", budget_usd)
    logger.info("Starting ELOs: A=%d  B=%d", elo_a, elo_b)

    for gen in range(start_gen, start_gen + DUEL_MAX_GENERATIONS):

        # Budget gate
        if budget_usd and SESSION_LEDGER.total_cost >= budget_usd:
            logger.info("Budget of $%.2f reached ($%.4f spent) — stopping.",
                        budget_usd, SESSION_LEDGER.total_cost)
            break

        # Alternate colors every game for fair ELO
        agent_a_is_white = (gen % 2 == 1)
        remaining = (budget_usd - SESSION_LEDGER.total_cost) if budget_usd else None

        logger.info("=" * 70)
        logger.info(
            "DUEL Gen %d  |  A-v%d (%s) vs B-v%d (%s)  |  ELO A=%d B=%d%s",
            gen, version_a, "White" if agent_a_is_white else "Black",
            version_b, "Black" if agent_a_is_white else "White",
            elo_a, elo_b,
            "  |  Budget left: $%.2f" % remaining if remaining is not None else "",
        )
        logger.info("=" * 70)

        cost_before = SESSION_LEDGER.total_cost
        try:
            result, pgn, blunders_white, blunders_black = play_duel(
                skills_a=playbook_a,
                skills_b=playbook_b,
                agent_a_color=chess.WHITE if agent_a_is_white else chess.BLACK,
                generation=gen,
                stockfish_path=STOCKFISH_PATH,
            )
        except Exception as exc:
            logger.error("Game failed: %s", exc)
            break

        game_cost = SESSION_LEDGER.total_cost - cost_before

        # ELO update
        elo_a_before, elo_a_after, elo_b_before, elo_b_after, history = update_elos(
            result=result,
            agent_a_is_white=agent_a_is_white,
            generation=gen,
            version_a=version_a,
            version_b=version_b,
            game_cost=game_cost,
        )
        elo_a, elo_b = elo_a_after, elo_b_after

        # Determine winner from A's perspective
        if agent_a_is_white:
            winner = "A" if result == "1-0" else ("B" if result == "0-1" else "draw")
        else:
            winner = "A" if result == "0-1" else ("B" if result == "1-0" else "draw")

        logger.info("Result: %s  →  Winner: Agent-%s", result, winner.upper())
        logger.info("ELO  A: %d → %d (%+d)  |  B: %d → %d (%+d)",
                    elo_a_before, elo_a_after, elo_a_after - elo_a_before,
                    elo_b_before, elo_b_after, elo_b_after - elo_b_before)
        logger.info("Game cost: $%.4f  |  Session total: $%.4f", game_cost, SESSION_LEDGER.total_cost)

        records.append({"generation": gen, "winner": winner, "result": result,
                        "elo_a": elo_a_after, "elo_b": elo_b_after})
        _log_scoreboard(records)

        # Budget gate before Coach call
        if budget_usd and SESSION_LEDGER.total_cost >= budget_usd:
            logger.info("Budget exhausted before Coach call — stopping.")
            break

        # Coach patches the loser's playbook
        # In a draw, patch the agent that made more blunders
        if winner == "A":
            loser_agent = "B"
            loser_blunders = blunders_black if agent_a_is_white else blunders_white
            loser_color = "Black" if agent_a_is_white else "White"
            new_b, _ = coach_patch(pgn, playbook_b, loser_blunders, "B", loser_color,
                                   result, game_ledger=None)
            if new_b != playbook_b:
                version_b = gen + 1
                playbook_b = new_b
                _save(playbook_b, "b", version_b)

        elif winner == "B":
            loser_agent = "A"
            loser_blunders = blunders_white if agent_a_is_white else blunders_black
            loser_color = "White" if agent_a_is_white else "Black"
            new_a, _ = coach_patch(pgn, playbook_a, loser_blunders, "A", loser_color,
                                   result, game_ledger=None)
            if new_a != playbook_a:
                version_a = gen + 1
                playbook_a = new_a
                _save(playbook_a, "a", version_a)

        else:  # draw — patch whichever agent had more blunders
            b_white_blunders = blunders_white if not agent_a_is_white else blunders_black
            b_black_blunders = blunders_black if not agent_a_is_white else blunders_white
            if len(blunders_white) >= len(blunders_black):
                w_color = "White" if agent_a_is_white else "Black"
                w_agent = "A" if agent_a_is_white else "B"
                w_play  = playbook_a if agent_a_is_white else playbook_b
                new_w, _ = coach_patch(pgn, w_play, blunders_white, w_agent, w_color,
                                       result, game_ledger=None)
                if agent_a_is_white and new_w != playbook_a:
                    version_a = gen + 1; playbook_a = new_w; _save(playbook_a, "a", version_a)
                elif not agent_a_is_white and new_w != playbook_b:
                    version_b = gen + 1; playbook_b = new_w; _save(playbook_b, "b", version_b)
            else:
                b_color = "Black" if agent_a_is_white else "White"
                b_agent = "B" if agent_a_is_white else "A"
                b_play  = playbook_b if agent_a_is_white else playbook_a
                new_b2, _ = coach_patch(pgn, b_play, blunders_black, b_agent, b_color,
                                        result, game_ledger=None)
                if agent_a_is_white and new_b2 != playbook_b:
                    version_b = gen + 1; playbook_b = new_b2; _save(playbook_b, "b", version_b)
                elif not agent_a_is_white and new_b2 != playbook_a:
                    version_a = gen + 1; playbook_a = new_b2; _save(playbook_a, "a", version_a)

        logger.info("Playbooks now: A=v%d  B=v%d", version_a, version_b)

    print_duel_chart(load_history())
    logger.info("Session total: %s", SESSION_LEDGER.summary())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duel Learning Loop — Agent A vs Agent B")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest saved playbook versions")
    parser.add_argument("--budget", type=float, default=None, metavar="USD",
                        help="Stop when total spend reaches this amount")
    args = parser.parse_args()
    run(resume=args.resume, budget_usd=args.budget)
