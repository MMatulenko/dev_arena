"""
Duel Arena — LLM Agent A vs LLM Agent B
========================================
Two LLM agents, each with their own SKILLS.md playbook, play chess against each other.
Stockfish runs in ANALYSIS-ONLY mode — it identifies blunders but never plays a move.

Coach uses the surgical JSON-patch approach from dev_arena (not a full rewrite):
  { "diagnosis": "...", "old_text": "...", "new_text": "..." }
This keeps Coach token cost ~75% lower than a full playbook rewrite.
"""

import datetime
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field

import chess
import chess.engine
import chess.pgn

from arena import get_skill_move
from config import (
    ANALYSIS_DEPTH,
    ANALYSIS_MOVE_TIME,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    DUEL_BLUNDER_CP_THRESHOLD,
    DUEL_COACH_MAX_TOKENS,
    DUEL_EVAL_WIN_THRESHOLD,
    DUEL_GAMES_DIR,
    DUEL_LLM_MOVES,
    DUEL_MAX_RETRIES,
    DUEL_PLAYER_MAX_TOKENS,
    DUEL_TEMPERATURE,
    LLM_PROVIDER,
    MAX_MOVES_PER_GAME,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    STOCKFISH_PATH,
    get_pricing,
)

logger = logging.getLogger(__name__)

_API_RETRY_DELAYS = [3, 8, 20]


# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------

@dataclass
class CostLedger:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, in_tok: int, out_tok: int) -> float:
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.calls += 1
        price_in, price_out = get_pricing()
        return (in_tok * price_in + out_tok * price_out) / 1_000_000

    @property
    def total_cost(self) -> float:
        price_in, price_out = get_pricing()
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000

    def summary(self) -> str:
        return "calls=%d | in=%d tok | out=%d tok | $%.4f" % (
            self.calls, self.input_tokens, self.output_tokens, self.total_cost
        )


SESSION_LEDGER = CostLedger()


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, max_tokens: int,
               temperature: float = DUEL_TEMPERATURE) -> tuple[str, int, int]:
    provider = LLM_PROVIDER.lower()

    for attempt, delay in enumerate([0] + _API_RETRY_DELAYS):
        if delay:
            logger.warning("API error — retrying in %ds (attempt %d/%d)",
                           delay, attempt, len(_API_RETRY_DELAYS))
            time.sleep(delay)
        try:
            if provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                msg = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text, msg.usage.input_tokens, msg.usage.output_tokens

            elif provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                usage = resp.usage
                return (resp.choices[0].message.content,
                        usage.prompt_tokens, usage.completion_tokens)

            else:
                raise ValueError("Unknown LLM_PROVIDER: %s" % provider)

        except Exception as exc:
            msg_str = str(exc)
            if any(c in msg_str for c in ("500", "529", "overloaded", "Internal server")):
                if attempt < len(_API_RETRY_DELAYS):
                    continue
            raise


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _format_board(board: chess.Board) -> str:
    lines = str(board).split("\n")
    ranks = "87654321"
    numbered = ["%s  %s" % (ranks[i], l) for i, l in enumerate(lines)]
    return "\n".join(numbered) + "\n   a b c d e f g h"


def _last_n_moves(board: chess.Board, n: int = 6) -> str:
    moves = list(board.move_stack)[-n:]
    if not moves:
        return "None"
    tmp = board.copy()
    for _ in moves:
        tmp.pop()
    parts = []
    for mv in moves:
        parts.append(tmp.san(mv))
        tmp.push(mv)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Stockfish analysis (analysis-only — no move generation)
# ---------------------------------------------------------------------------

def _score_str(score: chess.engine.PovScore) -> str:
    white = score.white()
    if white.is_mate():
        m = white.mate()
        return "Mate in %d for %s" % (abs(m), "White" if m > 0 else "Black")
    cp = white.score()
    if cp is None:
        return "?"
    pawns = cp / 100.0
    side = "White" if pawns > 0 else ("Black" if pawns < 0 else "equal")
    return ("%+.2f pawns (%s better)" % (pawns, side)) if side != "equal" else "0.00 (equal)"


def _pv_san(board: chess.Board, pv: list[chess.Move], max_moves: int = 4) -> str:
    tmp = board.copy()
    parts = []
    for mv in pv[:max_moves]:
        if tmp.is_legal(mv):
            parts.append(tmp.san(mv))
            tmp.push(mv)
        else:
            break
    return " ".join(parts) if parts else "—"


def analyse_position(engine: chess.engine.SimpleEngine,
                     board: chess.Board,
                     depth: int = ANALYSIS_DEPTH) -> dict:
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth, time=ANALYSIS_MOVE_TIME))
        score = info.get("score", chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
        pv = info.get("pv", [])
        best_san = board.san(pv[0]) if pv and board.is_legal(pv[0]) else "—"
        cp_raw = score.white().score(mate_score=10000) or 0
        return {
            "score_str":    _score_str(score),
            "best_move_san": best_san,
            "pv_san":       _pv_san(board, pv),
            "cp_white":     cp_raw,
        }
    except Exception as exc:
        logger.debug("Analysis failed: %s", exc)
        return {"score_str": "?", "best_move_san": "?", "pv_san": "—", "cp_white": 0}


# ---------------------------------------------------------------------------
# Player Agent (shared by A and B)
# ---------------------------------------------------------------------------

_PLAYER_PROMPT = """\
You are a Chess Player. Your name is Agent {agent}.

=== YOUR CHESS PLAYBOOK ===
{skills}
===========================
{game_notes}
Current position (you play {color}):
{board}

Move number: {move_number}
Recent moves: {recent_moves}
Your legal moves: {legal_moves}

Apply your Playbook rules to pick the best move.
First write 1-2 sentences of reasoning, then on a new line write exactly:
MOVE: <your chosen move in SAN notation>

Important: the MOVE: line must contain only the plain move — no **, *, or ` characters.
"""

_GAME_NOTES_HEADER = """\
=== LIVE GAME FEEDBACK (from this game so far) ===
{notes}
===================================================

"""

_PLAYER_RETRY_PROMPT = """\
That move '{bad_move}' is not legal. You must pick from this list:
{legal_moves}

Reply with MOVE: <legal move> only.
"""


def _clean_move_candidate(raw: str) -> str:
    return re.sub(r"[*`\[\]]", "", raw).strip().rstrip(".")


def _parse_move(reply: str, board: chess.Board, legal_sans: list[str]) -> chess.Move | None:
    lower_map = {s.lower(): s for s in legal_sans}

    match = re.search(r"MOVE:\s*(.+?)(?:\s*$)", reply, re.IGNORECASE | re.MULTILINE)
    if match:
        candidate = _clean_move_candidate(match.group(1))
        if candidate in legal_sans:
            try:
                return board.parse_san(candidate)
            except Exception:
                pass
        if candidate.lower() in lower_map:
            try:
                return board.parse_san(lower_map[candidate.lower()])
            except Exception:
                pass

    # Fallback: scan every token in reply
    for word in re.split(r"[\s,*`\[\]]+", reply):
        word = word.strip().rstrip(".")
        if not word:
            continue
        if word in legal_sans:
            try:
                return board.parse_san(word)
            except Exception:
                continue
        if word.lower() in lower_map:
            try:
                return board.parse_san(lower_map[word.lower()])
            except Exception:
                continue
    return None


def get_agent_move(board: chess.Board, skills_text: str, agent: str,
                   game_ledger: CostLedger | None = None,
                   game_notes: list[str] | None = None) -> tuple[chess.Move, str]:
    """LLM agent picks a move from their playbook, with live game feedback injected."""
    legal_moves = list(board.legal_moves)
    legal_sans = [board.san(m) for m in legal_moves]
    color = "White" if board.turn == chess.WHITE else "Black"

    notes_block = ""
    if game_notes:
        notes_block = _GAME_NOTES_HEADER.format(notes="\n".join(game_notes))

    prompt = _PLAYER_PROMPT.format(
        agent=agent,
        skills=skills_text,
        color=color,
        board=_format_board(board),
        move_number=board.fullmove_number,
        recent_moves=_last_n_moves(board),
        legal_moves=", ".join(legal_sans),
        game_notes=notes_block,
    )

    last_bad = ""
    for attempt in range(1 + DUEL_MAX_RETRIES):
        try:
            if attempt == 0:
                reply, in_tok, out_tok = _call_llm(prompt, max_tokens=DUEL_PLAYER_MAX_TOKENS)
            else:
                retry_prompt = _PLAYER_RETRY_PROMPT.format(
                    bad_move=last_bad, legal_moves=", ".join(legal_sans)
                )
                reply, in_tok, out_tok = _call_llm(retry_prompt, max_tokens=128)

            call_cost = SESSION_LEDGER.add(in_tok, out_tok)
            if game_ledger is not None:
                game_ledger.add(in_tok, out_tok)

            move = _parse_move(reply, board, legal_sans)
            if move:
                reasoning_match = re.search(r"^(.+?)(?=\nMOVE:)", reply, re.DOTALL)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                if reasoning:
                    logger.info("  [%s] Reasoning: %s…", agent, reasoning[:120])
                logger.info("  [%s] Cost: $%.4f | session=$%.4f [in=%d out=%d tok]",
                            agent, call_cost, SESSION_LEDGER.total_cost, in_tok, out_tok)
                return move, reasoning

            last_bad = re.search(r"MOVE:\s*(.+)", reply, re.IGNORECASE)
            last_bad = _clean_move_candidate(last_bad.group(1)) if last_bad else reply[:20]

        except Exception as exc:
            logger.error("[%s] LLM call failed: %s", agent, exc)
            if attempt >= DUEL_MAX_RETRIES:
                break

    logger.warning("[%s] All LLM attempts failed — using random move", agent)
    return random.choice(list(board.legal_moves)), ""


# ---------------------------------------------------------------------------
# Surgical patch helper (from dev_arena)
# ---------------------------------------------------------------------------

def _apply_patch(skills_text: str, old_text: str, new_text: str) -> tuple[str, bool]:
    if not old_text:
        return skills_text, False
    old_norm = old_text.replace("\r\n", "\n")
    new_norm = new_text.replace("\r\n", "\n")
    text_norm = skills_text.replace("\r\n", "\n")
    if old_norm not in text_norm:
        return skills_text, False
    return text_norm.replace(old_norm, new_norm, 1), True


# ---------------------------------------------------------------------------
# Coach Agent — outputs a surgical JSON patch
# ---------------------------------------------------------------------------

_COACH_PROMPT = """\
You are a Grandmaster Chess Coach. Agent {agent} played {color} and {outcome}.

=== AGENT {agent}'s CURRENT PLAYBOOK ===
{skills}
=========================================

=== GAME (PGN) ===
{pgn}
==================

=== TOP BLUNDERS BY AGENT {agent} (Stockfish full-strength analysis) ===
{blunders}
=======================================================================

Each blunder: what the agent played, what Stockfish recommended, and centipawns lost.

YOUR JOB — output a minimal surgical patch to the playbook:
1. Diagnose which rule is missing or insufficient (one sentence).
2. Decide the SMALLEST edit: extend an existing section with 1-2 new bullets,
   or add a new ## section at the end.
3. Output ONLY valid JSON — no markdown fences, no extra text:
   {{
     "diagnosis": "<one sentence>",
     "old_text":  "<exact verbatim text from the playbook above to replace>",
     "new_text":  "<same text plus new rule(s) appended — mark each with (New)>"
   }}

RULES:
- old_text MUST appear verbatim in the playbook shown above.
- To add a section, set old_text to the last line of the file and new_text to
  that line followed by the new section.
- Keep new rules short and actionable (one bullet each).
"""


def coach_patch(pgn: str, loser_skills: str, blunders: list[dict],
                agent: str, color: str, result: str,
                game_ledger: CostLedger | None = None) -> tuple[str, str]:
    """
    Coach produces a surgical JSON patch for the losing agent's playbook.
    Returns (patched_skills, diagnosis).
    """
    outcome = "lost" if (
        (color == "White" and result == "0-1") or
        (color == "Black" and result == "1-0")
    ) else "drew"

    if blunders:
        blunder_lines = []
        for i, b in enumerate(blunders[:3], 1):
            side = "White" if b["side"] == chess.WHITE else "Black"
            reasoning_snippet = b.get("reasoning", "").strip()
            reasoning_line = (
                "\n   Agent's reasoning at the time: \"%s\"" % reasoning_snippet[:300]
                if reasoning_snippet else ""
            )
            blunder_lines.append(
                "%d. Move %d (%s) — played %r, SF best: %r (line: %s), lost %d cp%s" % (
                    i, b["move_number"], side, b["ai_move"],
                    b["best_move"], b["pv"], abs(b["cp_loss"]),
                    reasoning_line,
                )
            )
        blunder_text = "\n".join(blunder_lines)
    else:
        blunder_text = "(No blunders detected — game was close)"

    prompt = _COACH_PROMPT.format(
        agent=agent,
        color=color,
        outcome=outcome,
        skills=loser_skills,
        pgn=pgn,
        blunders=blunder_text,
    )

    logger.info("Coach patching %s's playbook (result=%s)…", agent, result)
    try:
        raw, in_tok, out_tok = _call_llm(
            prompt, max_tokens=DUEL_COACH_MAX_TOKENS, temperature=0.3
        )
        call_cost = SESSION_LEDGER.add(in_tok, out_tok)
        if game_ledger is not None:
            game_ledger.add(in_tok, out_tok)
        logger.info("  Coach[%s] cost: $%.4f | session=$%.4f [in=%d out=%d tok]",
                    agent, call_cost, SESSION_LEDGER.total_cost, in_tok, out_tok)

        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())

        patch = json.loads(raw)
        diagnosis = patch.get("diagnosis", "no diagnosis")
        old_text  = patch.get("old_text", "")
        new_text  = patch.get("new_text", "")

        logger.info("  Coach[%s] diagnosis: %s", agent, diagnosis)

        patched, ok = _apply_patch(loser_skills, old_text, new_text)
        if ok:
            logger.info("  Patch applied cleanly (%d → %d chars)", len(loser_skills), len(patched))
            return patched, diagnosis

        # Fallback: append new section
        logger.warning("  Patch old_text not found — appending as fallback")
        fallback = loser_skills.rstrip() + "\n\n" + new_text.strip() + "\n"
        return fallback, diagnosis

    except json.JSONDecodeError as exc:
        logger.error("Coach[%s] returned invalid JSON (%s) — keeping playbook", agent, exc)
        return loser_skills, ""
    except Exception as exc:
        logger.error("Coach[%s] failed: %s", agent, exc)
        return loser_skills, ""


# ---------------------------------------------------------------------------
# Match runner — A vs B with Stockfish analysis
# ---------------------------------------------------------------------------

def play_duel(
    skills_a: str,
    skills_b: str,
    agent_a_color: chess.Color,
    generation: int,
    stockfish_path: str = STOCKFISH_PATH,
) -> tuple[str, str, list[dict], list[dict]]:
    """
    Run one full game: Agent A vs Agent B.
    Stockfish analyses every position (analysis-only, does not play).
    Colors alternate each game — pass agent_a_color externally.

    Returns (result_str, pgn_str, blunders_white, blunders_black)
    where blunders_* are lists of blunder dicts from each side's perspective.
    """
    board = chess.Board()
    game = chess.pgn.Game()

    if agent_a_color == chess.WHITE:
        white_agent, black_agent = "A", "B"
        white_skills, black_skills = skills_a, skills_b
    else:
        white_agent, black_agent = "B", "A"
        white_skills, black_skills = skills_b, skills_a

    game.headers["White"] = "Agent-%s" % white_agent
    game.headers["Black"] = "Agent-%s" % black_agent
    game.headers["Date"] = datetime.date.today().isoformat()
    game.headers["Event"] = "Duel Generation %d" % generation
    node = game

    try:
        analysis_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as exc:
        logger.error("Failed to launch Stockfish at %s: %s", stockfish_path, exc)
        raise

    price_in, price_out = get_pricing()
    model_name = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else OPENAI_MODEL
    logger.info(
        "Gen %d | Agent-A(%s) vs Agent-B(%s) | model=%s ($%.2f/$%.2f per MTok)",
        generation, "White" if agent_a_color == chess.WHITE else "Black",
        "Black" if agent_a_color == chess.WHITE else "White",
        model_name, price_in, price_out,
    )

    game_ledger = CostLedger()
    white_llm_moves = 0
    black_llm_moves = 0
    move_count = 0
    blunders_white: list[dict] = []
    blunders_black: list[dict] = []
    eval_result: str | None = None
    # Per-agent live feedback injected into the next move prompt
    notes_white: list[str] = []
    notes_black: list[str] = []

    try:
        while not board.is_game_over() and move_count < MAX_MOVES_PER_GAME:
            both_exhausted = (white_llm_moves >= DUEL_LLM_MOVES and
                              black_llm_moves >= DUEL_LLM_MOVES)

            # Once both sides have used all LLM moves, evaluate and stop —
            # no minimax, no more play. Pure LLM phase is what we measure.
            if both_exhausted:
                final_eval = analyse_position(analysis_engine, board)
                cp = final_eval["cp_white"]
                if cp >= DUEL_EVAL_WIN_THRESHOLD:
                    eval_result = "1-0"
                elif cp <= -DUEL_EVAL_WIN_THRESHOLD:
                    eval_result = "0-1"
                else:
                    eval_result = "1/2-1/2"
                logger.info(
                    "LLM phase complete — Stockfish eval: %s (%+d cp) → declared %s",
                    final_eval["score_str"], cp, eval_result,
                )
                break

            # Analyse BEFORE the move
            pre = analyse_position(analysis_engine, board)
            cp_before = pre["cp_white"]

            if board.turn == chess.WHITE:
                agent = white_agent
                logger.info("Move %d — Agent-%s (White, LLM #%d/%d) thinking…",
                            board.fullmove_number, agent, white_llm_moves + 1, DUEL_LLM_MOVES)
                move, reasoning = get_agent_move(board, white_skills, agent, game_ledger,
                                                 game_notes=notes_white or None)
                white_llm_moves += 1
            else:
                agent = black_agent
                logger.info("Move %d — Agent-%s (Black, LLM #%d/%d) thinking…",
                            board.fullmove_number, agent, black_llm_moves + 1, DUEL_LLM_MOVES)
                move, reasoning = get_agent_move(board, black_skills, agent, game_ledger,
                                                 game_notes=notes_black or None)
                black_llm_moves += 1

            move_san = board.san(move)
            board.push(move)

            # Analyse AFTER the move
            post = analyse_position(analysis_engine, board)
            cp_after = post["cp_white"]

            if board.turn == chess.BLACK:  # white just moved
                cp_loss = cp_after - cp_before
                logger.info("  Agent-%s White: %s | eval: %s | SF best: %s (line: %s)",
                            white_agent, move_san, post["score_str"],
                            pre["best_move_san"], pre["pv_san"])
                if cp_loss < -DUEL_BLUNDER_CP_THRESHOLD:
                    note = ("⚠ Move %d: You played %s but %s was better (line: %s) — lost %d cp. "
                            "Avoid repeating this pattern." % (
                                board.fullmove_number, move_san,
                                pre["best_move_san"], pre["pv_san"], abs(cp_loss)))
                    notes_white.append(note)
                    logger.info("  → Note added to White: %s", note)
                    blunders_white.append({
                        "move_number": board.fullmove_number,
                        "side": chess.WHITE,
                        "ai_move": move_san,
                        "best_move": pre["best_move_san"],
                        "pv": pre["pv_san"],
                        "score_before": pre["score_str"],
                        "cp_loss": cp_loss,
                        "reasoning": reasoning,
                    })
                elif move_san == pre["best_move_san"]:
                    note = ("✓ Move %d: %s matched the engine's top recommendation — well played!" %
                            (board.fullmove_number, move_san))
                    notes_white.append(note)
            else:  # black just moved
                cp_loss = -(cp_after - cp_before)
                logger.info("  Agent-%s Black: %s | eval: %s | SF best: %s (line: %s)",
                            black_agent, move_san, post["score_str"],
                            pre["best_move_san"], pre["pv_san"])
                if cp_loss < -DUEL_BLUNDER_CP_THRESHOLD:
                    note = ("⚠ Move %d: You played %s but %s was better (line: %s) — lost %d cp. "
                            "Avoid repeating this pattern." % (
                                board.fullmove_number, move_san,
                                pre["best_move_san"], pre["pv_san"], abs(cp_loss)))
                    notes_black.append(note)
                    logger.info("  → Note added to Black: %s", note)
                    blunders_black.append({
                        "move_number": board.fullmove_number,
                        "side": chess.BLACK,
                        "ai_move": move_san,
                        "best_move": pre["best_move_san"],
                        "pv": pre["pv_san"],
                        "score_before": pre["score_str"],
                        "cp_loss": cp_loss,
                        "reasoning": reasoning,
                    })
                elif move_san == pre["best_move_san"]:
                    note = ("✓ Move %d: %s matched the engine's top recommendation — well played!" %
                            (board.fullmove_number, move_san))
                    notes_black.append(note)

            node = node.add_variation(move)
            move_count += 1
    finally:
        analysis_engine.quit()

    # Result: natural game-over takes priority, otherwise use Stockfish evaluation
    if board.is_game_over():
        result_str = board.result()
    else:
        result_str = eval_result or "1/2-1/2"
    game.headers["Result"] = result_str
    pgn_str = str(game)

    blunders_white.sort(key=lambda b: b["cp_loss"])
    blunders_black.sort(key=lambda b: b["cp_loss"])

    logger.info("─" * 60)
    logger.info("Game over: %s after %d half-moves | White=Agent-%s  Black=Agent-%s",
                result_str, move_count, white_agent, black_agent)
    if blunders_white:
        logger.info("  White worst: move %d (%s) lost %d cp vs %s",
                    blunders_white[0]["move_number"], blunders_white[0]["ai_move"],
                    abs(blunders_white[0]["cp_loss"]), blunders_white[0]["best_move"])
    if blunders_black:
        logger.info("  Black worst: move %d (%s) lost %d cp vs %s",
                    blunders_black[0]["move_number"], blunders_black[0]["ai_move"],
                    abs(blunders_black[0]["cp_loss"]), blunders_black[0]["best_move"])
    logger.info("Game cost: $%.4f | session=$%.4f", game_ledger.total_cost, SESSION_LEDGER.total_cost)
    logger.info("─" * 60)

    DUEL_GAMES_DIR.mkdir(parents=True, exist_ok=True)
    pgn_path = DUEL_GAMES_DIR / ("duel_gen%03d.pgn" % generation)
    pgn_path.write_text(pgn_str)
    logger.info("PGN saved → %s", pgn_path)

    return result_str, pgn_str, blunders_white, blunders_black
