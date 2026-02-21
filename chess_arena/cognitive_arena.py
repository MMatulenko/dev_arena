"""
Cognitive Arena
===============
Player Agent  — LLM reads SKILLS.md + board → picks a legal SAN move.
               Hybrid: LLM for the first COGNITIVE_LLM_MOVES half-moves,
               then falls back to the minimax skill for endgame.
Coach Agent   — After a loss/draw, LLM rewrites the playbook.
               Coach prompt includes top-3 blunder moments identified by
               full-strength Stockfish analysis (free — no LLM tokens used).

Move Analysis — after every move, Stockfish evaluates the position and prints
               score + best continuation so you can see WHY it played what it played.
"""

import datetime
import logging
import random
import re
from dataclasses import dataclass, field

import chess
import chess.engine
import chess.pgn

from arena import get_skill_move
from config import (
    ANALYSIS_DEPTH,
    ANALYSIS_MOVE_TIME,
    COGNITIVE_COACH_MAX_TOKENS,
    COGNITIVE_LLM_MOVES,
    COGNITIVE_MAX_RETRIES,
    COGNITIVE_PLAYER_MAX_TOKENS,
    COGNITIVE_TEMPERATURE,
    GAMES_DIR,
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    MAX_MOVES_PER_GAME,
    STOCKFISH_MOVE_TIME,
    STOCKFISH_SKILL_LEVEL,
    STOCKFISH_PATH,
    get_pricing,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------

@dataclass
class CostLedger:
    """Accumulates token usage and dollar cost across a session."""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, in_tok: int, out_tok: int) -> float:
        """Record usage and return the incremental cost of this call."""
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
        return (
            f"calls={self.calls} | "
            f"in={self.input_tokens:,} tok | "
            f"out={self.output_tokens:,} tok | "
            f"total=${self.total_cost:.4f}"
        )


# Module-level ledger — persists across all generations in a session
SESSION_LEDGER = CostLedger()


# ---------------------------------------------------------------------------
# Board formatter
# ---------------------------------------------------------------------------

_PIECE_SYMBOLS = {
    (chess.PAWN,   chess.WHITE): "♙", (chess.PAWN,   chess.BLACK): "♟",
    (chess.KNIGHT, chess.WHITE): "♘", (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.WHITE): "♗", (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK,   chess.WHITE): "♖", (chess.ROOK,   chess.BLACK): "♜",
    (chess.QUEEN,  chess.WHITE): "♕", (chess.QUEEN,  chess.BLACK): "♛",
    (chess.KING,   chess.WHITE): "♔", (chess.KING,   chess.BLACK): "♚",
}


def _format_board(board: chess.Board) -> str:
    rows = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        row = f"{rank + 1} "
        for file_ in range(8):
            sq = chess.square(file_, rank)
            piece = board.piece_at(sq)
            row += (_PIECE_SYMBOLS.get((piece.piece_type, piece.color), "?") if piece else "·") + " "
        rows.append(row)
    return "\n".join(rows)


def _last_n_moves(board: chess.Board, n: int = 6) -> str:
    moves = list(board.move_stack)[-n:]
    tmp = board.copy()
    for _ in moves:
        tmp.pop()
    parts = []
    for move in moves:
        parts.append(tmp.san(move))
        tmp.push(move)
    return " ".join(parts) if parts else "(opening)"


# ---------------------------------------------------------------------------
# LLM adapter — returns (text, input_tokens, output_tokens)
# Retries on transient 500/529 errors with exponential backoff.
# ---------------------------------------------------------------------------

_API_RETRY_DELAYS = [3, 8, 20]  # seconds between retries on server errors


def _call_llm(prompt: str, max_tokens: int,
              temperature: float = COGNITIVE_TEMPERATURE) -> tuple[str, int, int]:
    import time
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
            # Retry on transient server / overload errors
            if any(code in msg_str for code in ("500", "529", "overloaded", "Internal server")):
                if attempt < len(_API_RETRY_DELAYS):
                    continue
            raise  # non-transient or out of retries


# ---------------------------------------------------------------------------
# Player Agent prompts
# ---------------------------------------------------------------------------

_PLAYER_PROMPT = """\
You are playing chess as White. Study your playbook carefully, then choose the single best legal move.

=== YOUR PLAYBOOK (SKILLS.md) ===
{skills}
=================================

=== CURRENT BOARD ===
{board}
=====================
Move number: {move_number}
Recent moves: {recent_moves}

Legal moves you may play (Standard Algebraic Notation):
{legal_moves}

Instructions:
1. Briefly reason which rule in your playbook applies most right now (1-3 sentences).
2. Choose the single best legal move from the list above.
3. End your reply with exactly this line (replace X with your chosen move, no markdown):
MOVE: X

Important: the MOVE: line must contain only the plain move text — no **, *, or ` characters.
"""

_PLAYER_RETRY_PROMPT = """\
You previously chose "{bad_move}" which is not in the legal move list.

Legal moves: {legal_moves}

Choose exactly one move from the list above and reply with ONLY:
MOVE: X
"""

# ---------------------------------------------------------------------------
# Player Agent
# ---------------------------------------------------------------------------

def _clean_move_candidate(raw: str) -> str:
    """Strip markdown decorators (**, *, `, [, ]) that Haiku sometimes adds."""
    return re.sub(r"[*`\[\]]", "", raw).strip().rstrip(".")


def _parse_move(reply: str, board: chess.Board, legal_sans: list[str]) -> chess.Move | None:
    lower_map = {s.lower(): s for s in legal_sans}

    # Primary: find MOVE: line and clean it
    match = re.search(r"MOVE:\s*(.+?)(?:\s*$)", reply, re.IGNORECASE | re.MULTILINE)
    if match:
        candidate = _clean_move_candidate(match.group(1))
        if candidate and candidate in legal_sans:
            try:
                return board.parse_san(candidate)
            except Exception:
                pass
        if candidate and candidate.lower() in lower_map:
            try:
                return board.parse_san(lower_map[candidate.lower()])
            except Exception:
                pass

    # Fallback: scan every word in reply for an exact legal SAN match
    # (handles "MOVE: **e4**" where candidate becomes empty after stripping)
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


def get_cognitive_move(board: chess.Board, skills_text: str,
                       fallback_skill_code: str | None = None,
                       game_ledger: CostLedger | None = None) -> chess.Move:
    """
    Player Agent: LLM reads SKILLS.md and picks a move.
    Prints per-call and running cost after each API response.
    """
    legal_moves = list(board.legal_moves)
    legal_sans = [board.san(m) for m in legal_moves]
    move_number = board.fullmove_number

    prompt = _PLAYER_PROMPT.format(
        skills=skills_text,
        board=_format_board(board),
        move_number=move_number,
        recent_moves=_last_n_moves(board),
        legal_moves=", ".join(legal_sans),
    )

    last_bad: str = ""
    for attempt in range(1 + COGNITIVE_MAX_RETRIES):
        try:
            if attempt == 0:
                reply, in_tok, out_tok = _call_llm(prompt, max_tokens=COGNITIVE_PLAYER_MAX_TOKENS)
            else:
                retry_prompt = _PLAYER_RETRY_PROMPT.format(
                    bad_move=last_bad,
                    legal_moves=", ".join(legal_sans),
                )
                reply, in_tok, out_tok = _call_llm(retry_prompt, max_tokens=128)

            # Record cost
            call_cost = SESSION_LEDGER.add(in_tok, out_tok)
            if game_ledger is not None:
                game_ledger.add(in_tok, out_tok)
            game_cost = game_ledger.total_cost if game_ledger else 0.0

            move = _parse_move(reply, board, legal_sans)
            if move is not None:
                reasoning = reply.split("MOVE:")[0].strip()
                if reasoning:
                    logger.info("  Reasoning : %s", reasoning[:250])
                logger.info(
                    "  Cost      : this call=$%.4f | game=$%.4f | session=$%.4f"
                    "  [in=%d out=%d tok]",
                    call_cost, game_cost, SESSION_LEDGER.total_cost, in_tok, out_tok,
                )
                return move

            m = re.search(r"MOVE:\s*\[?([^\]\n]+)\]?", reply, re.IGNORECASE)
            last_bad = m.group(1).strip() if m else "?"
            logger.warning("  Attempt %d: illegal move '%s', retrying [cost $%.4f]",
                           attempt + 1, last_bad, call_cost)

        except Exception as exc:
            logger.error("LLM call failed on attempt %d: %s", attempt + 1, exc)

    logger.warning("All LLM retries failed — using fallback")
    if fallback_skill_code:
        return get_skill_move(board, fallback_skill_code, depth=2)
    return random.choice(legal_moves)


# ---------------------------------------------------------------------------
# Coach Agent
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stockfish position analysis
# ---------------------------------------------------------------------------

def _score_str(score: chess.engine.PovScore) -> str:
    """Format a PovScore as a human-readable string from White's perspective."""
    white_score = score.white()
    if white_score.is_mate():
        m = white_score.mate()
        return f"Mate in {abs(m)} for {'White' if m > 0 else 'Black'}"
    cp = white_score.score()
    if cp is None:
        return "?"
    pawns = cp / 100.0
    side = "White" if pawns > 0 else "Black" if pawns < 0 else "equal"
    return f"{pawns:+.2f} pawns ({side} better)" if side != "equal" else "0.00 (equal)"


def _pv_san(board: chess.Board, pv: list[chess.Move], max_moves: int = 4) -> str:
    """Convert a principal variation (list of moves) to SAN notation."""
    tmp = board.copy()
    parts = []
    for move in pv[:max_moves]:
        if tmp.is_legal(move):
            parts.append(tmp.san(move))
            tmp.push(move)
        else:
            break
    return " ".join(parts) if parts else "—"


def analyse_position(engine: chess.engine.SimpleEngine,
                     board: chess.Board,
                     depth: int = ANALYSIS_DEPTH) -> dict:
    """
    Run full-strength Stockfish analysis on the current position.
    Returns dict with keys: score_str, best_move_san, pv_san, cp_white.
    """
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
# Coach Agent prompt (enriched with blunder analysis)
# ---------------------------------------------------------------------------

_COACH_PROMPT = """\
You are a Grandmaster Chess Coach. Your student played White and the game result was: {result}.

=== STUDENT'S CURRENT PLAYBOOK ===
{skills}
==================================

=== GAME HISTORY (PGN) ===
{pgn}
==========================

=== TOP BLUNDERS IDENTIFIED BY STOCKFISH (full-strength engine analysis) ===
{blunders}
============================================================================

Each blunder shows:
- Move number and what White played
- What Stockfish said was the best move instead
- The best continuation (Principal Variation) — this is EXACTLY why Stockfish's move was better
- How many centipawns White lost by making the wrong choice

Task:
1. Focus on the blunders above. What strategic concept did White fail to apply?
2. Rewrite the complete SKILLS.md document. Keep every correct rule. Add 1-2 new, specific rules that directly address the blunder pattern.
3. Mark updated rules with "(Updated)" and new rules with "(New)" at the end of the line.
4. Output ONLY the raw Markdown content — no code fences, no preamble, no explanation.
"""


def reflect_and_learn(pgn: str, skills_text: str, result: str,
                      generation: int, blunders: list[dict] | None = None) -> str:
    logger.info("Coach Agent analyzing generation %d (result=%s)", generation, result)

    # Format blunder report for Coach
    if blunders:
        blunder_lines = []
        for i, b in enumerate(blunders[:3], 1):
            blunder_lines.append(
                f"{i}. Move {b['move_number']} — White played {b['ai_move']!r}\n"
                f"   Stockfish best: {b['best_move']!r}  (Best line: {b['pv']})\n"
                f"   Cost: {b['cp_loss']:+d} centipawns  |  Position was: {b['score_before']}"
            )
        blunder_text = "\n".join(blunder_lines)
    else:
        blunder_text = "(No blunder analysis available)"

    prompt = _COACH_PROMPT.format(
        result=result, skills=skills_text, pgn=pgn, blunders=blunder_text
    )

    try:
        new_skills, in_tok, out_tok = _call_llm(
            prompt, max_tokens=COGNITIVE_COACH_MAX_TOKENS, temperature=0.4
        )
        call_cost = SESSION_LEDGER.add(in_tok, out_tok)
        logger.info(
            "  Coach cost: this call=$%.4f | session=$%.4f  [in=%d out=%d tok]",
            call_cost, SESSION_LEDGER.total_cost, in_tok, out_tok,
        )

        new_skills = new_skills.strip()
        if "#" not in new_skills:
            logger.error("Coach returned non-markdown response — keeping old playbook")
            return skills_text

        logger.info("Coach rewrote playbook (%d chars → %d chars)",
                    len(skills_text), len(new_skills))
        return new_skills
    except Exception as exc:
        logger.error("Coach Agent failed: %s", exc)
        return skills_text


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def play_cognitive_match(
    skills_text: str,
    fallback_skill_code: str | None = None,
    stockfish_path: str = STOCKFISH_PATH,
    skill_level: int = STOCKFISH_SKILL_LEVEL,
    generation: int = 0,
) -> tuple[str, str, list[dict]]:
    """
    Run one full cognitive game: Cognitive AI (White) vs Stockfish (Black).

    After every move, full-strength Stockfish analyses the position and prints:
      - Evaluation score (centipawns / mate distance)
      - Best move suggestion + principal variation

    Returns (result_str, pgn_str, blunders) where blunders is a list of the
    worst AI decisions ranked by centipawn loss.
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "CognitiveAI"
    game.headers["Black"] = f"Stockfish-Lvl{skill_level}"
    game.headers["Date"] = datetime.date.today().isoformat()
    game.headers["Event"] = f"Cognitive Generation {generation}"
    node = game

    try:
        # One engine instance for playing, one for full-strength analysis
        play_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        play_engine.configure({"Skill Level": skill_level})
        analysis_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        # Analysis engine uses full strength (no skill limit)
    except Exception as exc:
        logger.error("Failed to launch Stockfish at %s: %s", stockfish_path, exc)
        raise

    price_in, price_out = get_pricing()
    model_name = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else OPENAI_MODEL
    logger.info("Generation %d | CognitiveAI (White) vs Stockfish Lvl%d | model=%s ($%.2f/$%.2f per MTok)",
                generation, skill_level, model_name, price_in, price_out)

    game_ledger = CostLedger()
    white_llm_moves = 0
    move_count = 0
    blunders: list[dict] = []

    try:
        while not board.is_game_over() and move_count < MAX_MOVES_PER_GAME:
            # Analyse BEFORE the move to know the "correct" score and best move
            pre_analysis = analyse_position(analysis_engine, board)
            cp_before = pre_analysis["cp_white"]

            if board.turn == chess.WHITE:
                use_llm = white_llm_moves < COGNITIVE_LLM_MOVES
                if use_llm:
                    logger.info("Move %d (LLM #%d/%d) — thinking...",
                                board.fullmove_number, white_llm_moves + 1, COGNITIVE_LLM_MOVES)
                    move = get_cognitive_move(board, skills_text, fallback_skill_code,
                                             game_ledger=game_ledger)
                    white_llm_moves += 1
                else:
                    move = get_skill_move(board, fallback_skill_code, depth=3) \
                           if fallback_skill_code else random.choice(list(board.legal_moves))

                ai_san = board.san(move)

                # Analyse AFTER the AI's move to measure the cost of the choice
                board.push(move)
                post_analysis = analyse_position(analysis_engine, board)
                cp_after = post_analysis["cp_white"]
                cp_loss = cp_after - cp_before  # negative = AI move was worse for White

                mode = "LLM" if use_llm else "minimax"
                logger.info("  [%s] CognitiveAI: %s  |  eval: %s  |  SF best was: %s  (line: %s)",
                            mode, ai_san, post_analysis["score_str"],
                            pre_analysis["best_move_san"], pre_analysis["pv_san"])
                if cp_loss < -30:  # blunder threshold: lost >0.3 pawns
                    blunders.append({
                        "move_number":  board.fullmove_number,
                        "ai_move":      ai_san,
                        "best_move":    pre_analysis["best_move_san"],
                        "pv":           pre_analysis["pv_san"],
                        "score_before": pre_analysis["score_str"],
                        "cp_loss":      cp_loss,
                    })
            else:
                sf_result = play_engine.play(board, chess.engine.Limit(time=STOCKFISH_MOVE_TIME))
                move = sf_result.move
                sf_san = board.san(move)
                board.push(move)
                post_analysis = analyse_position(analysis_engine, board)
                logger.info("  Stockfish:    %s  |  eval after: %s  |  best continuation: %s",
                            sf_san, post_analysis["score_str"], post_analysis["pv_san"])

            node = node.add_variation(move)
            move_count += 1
    finally:
        play_engine.quit()
        analysis_engine.quit()

    result_str = board.result()
    game.headers["Result"] = result_str
    pgn_str = str(game)

    # Sort blunders by severity (largest centipawn loss first)
    blunders.sort(key=lambda b: b["cp_loss"])

    logger.info("─" * 60)
    logger.info("Game over: %s after %d half-moves", result_str, move_count)
    if blunders:
        logger.info("Top blunder: move %d (%s) — lost %d cp vs best %s",
                    blunders[0]["move_number"], blunders[0]["ai_move"],
                    abs(blunders[0]["cp_loss"]), blunders[0]["best_move"])
    logger.info("Game cost : $%.4f  (%s)", game_ledger.total_cost, game_ledger.summary())
    logger.info("Session   : $%.4f  (%s)", SESSION_LEDGER.total_cost, SESSION_LEDGER.summary())
    logger.info("─" * 60)

    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    pgn_path = GAMES_DIR / f"cognitive_gen{generation:03d}.pgn"
    pgn_path.write_text(pgn_str)
    logger.info("PGN saved → %s", pgn_path)

    return result_str, pgn_str, blunders
