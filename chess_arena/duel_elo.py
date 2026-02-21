"""
Dual ELO Tracker — Agent A vs Agent B
=======================================
Both agents start at DUEL_ELO_INITIAL (1200) and update against each other.
ELO is computed from the true A-vs-B result, giving a clean relative skill signal.
History is persisted to duel_elo_history.json.
"""

import json
import logging

from config import DUEL_ELO_FILE, DUEL_ELO_INITIAL, DUEL_ELO_K_FACTOR

logger = logging.getLogger(__name__)


def _expected(player_elo: int, opponent_elo: int) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400))


def _white_score(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result in ("1/2-1/2", "*"):
        return 0.5
    return 0.0


def load_history() -> dict:
    """Load {a: [...], b: [...]} from disk."""
    if DUEL_ELO_FILE.exists():
        try:
            return json.loads(DUEL_ELO_FILE.read_text())
        except Exception:
            pass
    return {"a": [], "b": []}


def current_elos(history: dict) -> tuple[int, int]:
    """Return (elo_a, elo_b) from latest history records."""
    elo_a = history["a"][-1]["elo_after"] if history["a"] else DUEL_ELO_INITIAL
    elo_b = history["b"][-1]["elo_after"] if history["b"] else DUEL_ELO_INITIAL
    return elo_a, elo_b


def update_elos(
    result: str,
    agent_a_is_white: bool,
    generation: int,
    version_a: int,
    version_b: int,
    game_cost: float,
) -> tuple[int, int, int, int, dict]:
    """
    Update both agents' ELOs given the game result.
    Returns (elo_a_before, elo_a_after, elo_b_before, elo_b_after, history).
    """
    history = load_history()
    elo_a, elo_b = current_elos(history)

    white_score = _white_score(result)
    if agent_a_is_white:
        score_a = white_score
        score_b = 1.0 - white_score
        expected_a = _expected(elo_a, elo_b)
        expected_b = 1.0 - expected_a
    else:
        score_a = 1.0 - white_score
        score_b = white_score
        expected_b = _expected(elo_b, elo_a)
        expected_a = 1.0 - expected_b

    delta_a = round(DUEL_ELO_K_FACTOR * (score_a - expected_a))
    delta_b = round(DUEL_ELO_K_FACTOR * (score_b - expected_b))
    new_elo_a = max(100, elo_a + delta_a)
    new_elo_b = max(100, elo_b + delta_b)

    record_a = {
        "generation":    generation,
        "version":       version_a,
        "result":        result,
        "color":         "White" if agent_a_is_white else "Black",
        "score":         score_a,
        "elo_before":    elo_a,
        "elo_after":     new_elo_a,
        "elo_delta":     delta_a,
        "opponent_elo":  elo_b,
        "game_cost_usd": round(game_cost, 5),
    }
    record_b = {
        "generation":    generation,
        "version":       version_b,
        "result":        result,
        "color":         "Black" if agent_a_is_white else "White",
        "score":         score_b,
        "elo_before":    elo_b,
        "elo_after":     new_elo_b,
        "elo_delta":     delta_b,
        "opponent_elo":  elo_a,
        "game_cost_usd": round(game_cost, 5),
    }
    history["a"].append(record_a)
    history["b"].append(record_b)
    DUEL_ELO_FILE.write_text(json.dumps(history, indent=2))

    return elo_a, new_elo_a, elo_b, new_elo_b, history


def print_duel_chart(history: dict):
    """Render a side-by-side ASCII ELO trend chart for A and B."""
    records_a = history.get("a", [])
    records_b = history.get("b", [])
    if not records_a:
        return

    elo_a = DUEL_ELO_INITIAL
    elo_b = DUEL_ELO_INITIAL

    logger.info("═" * 70)
    logger.info("DUEL ELO CHART")
    logger.info("  Gen  | Result | Agent-A ELO (Δ)    | Agent-B ELO (Δ)    | Lead")
    logger.info("─" * 70)

    for i, (ra, rb) in enumerate(zip(records_a[-15:], records_b[-15:])):
        result_icon = "A wins" if ra["score"] == 1.0 else ("B wins" if ra["score"] == 0.0 else "Draw ")
        a_delta = "%+d" % ra["elo_delta"]
        b_delta = "%+d" % rb["elo_delta"]
        lead_diff = ra["elo_after"] - rb["elo_after"]
        lead = "A +%d" % lead_diff if lead_diff > 0 else ("B +%d" % abs(lead_diff) if lead_diff < 0 else "tied")
        logger.info("  %03d  | %s | %4d (%4s)         | %4d (%4s)         | %s",
                    ra["generation"], result_icon,
                    ra["elo_after"], a_delta,
                    rb["elo_after"], b_delta,
                    lead)

    elo_a_final = records_a[-1]["elo_after"] if records_a else DUEL_ELO_INITIAL
    elo_b_final = records_b[-1]["elo_after"] if records_b else DUEL_ELO_INITIAL
    logger.info("─" * 70)
    logger.info("  FINAL: Agent-A=%d  Agent-B=%d  (started at %d each)",
                elo_a_final, elo_b_final, DUEL_ELO_INITIAL)
    logger.info("═" * 70)
