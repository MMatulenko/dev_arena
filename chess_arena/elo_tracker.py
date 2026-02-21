"""
ELO Tracker
===========
Computes ELO after each game and persists history to elo_history.json.
"""

import json
import logging
from pathlib import Path

from config import (
    ELO_INITIAL,
    ELO_K_FACTOR,
    ELO_HISTORY_FILE,
    STOCKFISH_ELO_MAP,
)

logger = logging.getLogger(__name__)


def _expected_score(player_elo: int, opponent_elo: int) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400))


def _result_score(result: str) -> float:
    """Convert PGN result string to White's score (1=win, 0.5=draw/timeout, 0=loss)."""
    if result == "1-0":
        return 1.0
    if result in ("1/2-1/2", "*"):  # * = max-moves reached, counts as draw
        return 0.5
    return 0.0


def load_history() -> list[dict]:
    if ELO_HISTORY_FILE.exists():
        try:
            return json.loads(ELO_HISTORY_FILE.read_text())
        except Exception:
            pass
    return []


def current_elo(history: list[dict]) -> int:
    if history:
        return history[-1]["elo_after"]
    return ELO_INITIAL


def update_elo(result: str, generation: int,
               playbook_version: int,
               stockfish_skill: int,
               game_cost: float) -> tuple[int, int, list[dict]]:
    """
    Update ELO based on game result.
    Returns (elo_before, elo_after, updated_history).
    """
    history = load_history()
    elo_before = current_elo(history)
    opponent_elo = STOCKFISH_ELO_MAP.get(stockfish_skill, 800)

    score = _result_score(result)
    expected = _expected_score(elo_before, opponent_elo)
    delta = round(ELO_K_FACTOR * (score - expected))
    elo_after = max(100, elo_before + delta)  # floor at 100

    record = {
        "generation":       generation,
        "playbook_version": playbook_version,
        "result":           result,
        "score":            score,
        "elo_before":       elo_before,
        "elo_after":        elo_after,
        "elo_delta":        delta,
        "opponent_elo":     opponent_elo,
        "stockfish_skill":  stockfish_skill,
        "game_cost_usd":    round(game_cost, 5),
    }
    history.append(record)
    ELO_HISTORY_FILE.write_text(json.dumps(history, indent=2))
    return elo_before, elo_after, history


def print_elo_chart(history: list[dict]):
    """Render a compact ASCII ELO trend chart."""
    if not history:
        return

    elos = [h["elo_after"] for h in history]
    lo, hi = min(elos + [ELO_INITIAL]), max(elos + [ELO_INITIAL])
    span = max(hi - lo, 1)
    width = 40

    logger.info("─" * 60)
    logger.info("ELO PROGRESS  (start=%d  now=%d  Δ%+d)",
                ELO_INITIAL, elos[-1], elos[-1] - ELO_INITIAL)

    for h in history[-15:]:  # last 15 games
        bar_len = int((h["elo_after"] - lo) / span * width)
        bar = "█" * bar_len
        icon = "W" if h["result"] == "1-0" else ("D" if h["result"] == "1/2-1/2" else "L")
        logger.info("  Gen%03d [%s] ELO %4d (%+d)  %s",
                    h["generation"], icon, h["elo_after"], h["elo_delta"], bar)
    logger.info("─" * 60)
