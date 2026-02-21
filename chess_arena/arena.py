import logging
import random
import chess
import chess.engine
import chess.pgn
import datetime
from pathlib import Path
from config import (
    STOCKFISH_PATH,
    STOCKFISH_SKILL_LEVEL,
    STOCKFISH_MOVE_TIME,
    SEARCH_DEPTH,
    MAX_MOVES_PER_GAME,
    GAMES_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe skill loader
# ---------------------------------------------------------------------------

def load_evaluate_fn(skill_code: str):
    """
    Compile and return the evaluate_board function from the given skill code string.
    Returns None if compilation or extraction fails.
    """
    local_env = {"chess": chess}
    try:
        exec(compile(skill_code, "<skill>", "exec"), local_env)
        fn = local_env.get("evaluate_board")
        if fn is None or not callable(fn):
            logger.error("skill code does not define a callable evaluate_board()")
            return None
        return fn
    except SyntaxError as exc:
        logger.error("SyntaxError in skill code: %s", exc)
        return None
    except Exception as exc:
        logger.error("Unexpected error loading skill code: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Minimax with alpha-beta pruning
# ---------------------------------------------------------------------------

def minimax(board: chess.Board, depth: int, alpha: float, beta: float,
            maximizing: bool, evaluate_fn) -> float:
    """
    Standard minimax with alpha-beta pruning.
    White maximizes, Black minimizes.
    """
    if depth == 0 or board.is_game_over():
        try:
            return evaluate_fn(board)
        except Exception as exc:
            logger.warning("evaluate_fn raised during search: %s", exc)
            return 0.0

    if maximizing:
        best = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, alpha, beta, False, evaluate_fn)
            board.pop()
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = float("inf")
        for move in board.legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, alpha, beta, True, evaluate_fn)
            board.pop()
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


# ---------------------------------------------------------------------------
# Skill AI move selection
# ---------------------------------------------------------------------------

def get_skill_move(board: chess.Board, skill_code: str, depth: int = SEARCH_DEPTH) -> chess.Move:
    """
    Use the AI's written skill code + minimax to pick a move.
    Falls back to a random legal move if the skill is broken.
    """
    evaluate_fn = load_evaluate_fn(skill_code)
    legal = list(board.legal_moves)

    if evaluate_fn is None:
        logger.warning("Falling back to random move due to bad skill code")
        return random.choice(legal)

    best_move = None
    # White maximizes
    is_white = board.turn == chess.WHITE
    best_score = -float("inf") if is_white else float("inf")

    for move in legal:
        board.push(move)
        score = minimax(board, depth - 1, -float("inf"), float("inf"),
                        not is_white, evaluate_fn)
        board.pop()

        if is_white and score > best_score:
            best_score = score
            best_move = move
        elif not is_white and score < best_score:
            best_score = score
            best_move = move

    return best_move if best_move else random.choice(legal)


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def play_match(skill_code: str,
               stockfish_path: str = STOCKFISH_PATH,
               skill_level: int = STOCKFISH_SKILL_LEVEL,
               generation: int = 0) -> tuple[str, str]:
    """
    Run one full game: Skill AI (White) vs Stockfish (Black).

    Returns:
        result_str  - e.g. "1-0", "0-1", "1/2-1/2"
        pgn_str     - full PGN text of the game
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "SkillAI"
    game.headers["Black"] = f"Stockfish-Lvl{skill_level}"
    game.headers["Date"] = datetime.date.today().isoformat()
    game.headers["Event"] = f"Generation {generation}"

    node = game

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Skill Level": skill_level})
    except Exception as exc:
        logger.error("Failed to launch Stockfish at %s: %s", stockfish_path, exc)
        raise

    logger.info("Generation %d | SkillAI (White) vs Stockfish Lvl%d (Black)",
                generation, skill_level)

    move_count = 0
    try:
        while not board.is_game_over() and move_count < MAX_MOVES_PER_GAME:
            if board.turn == chess.WHITE:
                move = get_skill_move(board, skill_code)
                logger.debug("SkillAI plays: %s", move.uci())
            else:
                result = engine.play(board, chess.engine.Limit(time=STOCKFISH_MOVE_TIME))
                move = result.move
                logger.debug("Stockfish plays: %s", move.uci())

            board.push(move)
            node = node.add_variation(move)
            move_count += 1
    finally:
        engine.quit()

    result_str = board.result()
    game.headers["Result"] = result_str
    pgn_str = str(game)

    logger.info("Generation %d result: %s after %d moves", generation, result_str, move_count)

    # Auto-save PGN
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    pgn_path = GAMES_DIR / f"game_gen{generation:03d}.pgn"
    pgn_path.write_text(pgn_str)
    logger.info("PGN saved to %s", pgn_path)

    return result_str, pgn_str
