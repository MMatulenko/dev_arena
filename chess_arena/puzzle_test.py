"""
Puzzle Test Suite
=================
Tests the Skill AI (minimax) or Cognitive AI (LLM) against a bank of mate-in-N positions.
For each puzzle, renders:
  - ASCII board in terminal
  - SVG file saved to boards/puzzle_<id>.svg

Run:
  python puzzle_test.py                                    # minimax skill_v0
  python puzzle_test.py --skill skills/skill_v3.py         # specific minimax version
  python puzzle_test.py --cognitive skills/SKILLS_v2.md    # cognitive playbook
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.svg

from arena import get_skill_move, load_evaluate_fn
from cognitive_arena import get_cognitive_move
from config import SKILLS_DIR, SEARCH_DEPTH

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("puzzle_test")

BOARDS_DIR = Path(__file__).parent / "boards"

# ---------------------------------------------------------------------------
# Puzzle bank  (FEN, expected UCI moves that lead to forced mate, category)
# ---------------------------------------------------------------------------

@dataclass
class Puzzle:
    id: str
    category: str          # "mate-in-1", "mate-in-2", "mate-in-3"
    fen: str
    # All moves that constitute a correct solution (first move must match one of these).
    # For mate-in-1: the single mating move UCI string.
    # For mate-in-2/3: the first correct move UCI string(s).
    correct_first_moves: list[str]
    description: str = ""


PUZZLES: list[Puzzle] = [
    # ── Mate in 1 ────────────────────────────────────────────────────────────
    Puzzle(
        id="m1_01",
        category="mate-in-1",
        fen="6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        correct_first_moves=["e1e8"],
        description="Rook to e8 — back rank mate",
    ),
    Puzzle(
        id="m1_02",
        category="mate-in-1",
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        correct_first_moves=["h5f7"],
        description="Scholar's mate — Qxf7#",
    ),
    Puzzle(
        id="m1_03",
        category="mate-in-1",
        fen="k7/8/1K6/8/8/8/8/7R w - - 0 1",
        correct_first_moves=["h1a1"],
        description="Rook to a1 — side mate with king support",
    ),
    Puzzle(
        id="m1_04",
        category="mate-in-1",
        fen="8/8/8/8/8/6K1/8/5k1R w - - 0 1",
        correct_first_moves=["h1f1"],
        description="Rook to f1 — smothered corner mate",
    ),
    Puzzle(
        id="m1_05",
        category="mate-in-1",
        fen="r2qkb1r/ppp2ppp/2np1n2/4p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QK2R w KQkq - 0 7",
        correct_first_moves=["d1d8"],
        description="Queen to d8 — back rank mate with bishop cover",
    ),
    # ── Mate in 2 ────────────────────────────────────────────────────────────
    Puzzle(
        id="m2_01",
        category="mate-in-2",
        fen="r1bk3r/ppp2ppp/2p5/2b5/2B5/8/PPP2PPP/RNB1K2R w KQ - 0 9",
        correct_first_moves=["c4f7"],
        description="Bishop takes f7 — Légal-style fork leading to mate",
    ),
    Puzzle(
        id="m2_02",
        category="mate-in-2",
        fen="1r4k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        correct_first_moves=["a1a8"],
        description="Rook exchange sac forcing back-rank mate",
    ),
    Puzzle(
        id="m2_03",
        category="mate-in-2",
        fen="4r1k1/ppp2ppp/8/8/8/8/PPP2PPP/4R1K1 w - - 0 1",
        correct_first_moves=["e1e8"],
        description="Rook trade forces decisive back-rank",
    ),
    Puzzle(
        id="m2_04",
        category="mate-in-2",
        fen="6k1/R7/6K1/8/8/8/8/8 w - - 0 1",
        correct_first_moves=["a7a8"],
        description="Rook to a8 — King + Rook vs King forced mate",
    ),
    Puzzle(
        id="m2_05",
        category="mate-in-2",
        fen="5rk1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        correct_first_moves=["e1e8"],
        description="Back-rank battle — first to the 8th wins",
    ),
    # ── Mate in 3 ────────────────────────────────────────────────────────────
    Puzzle(
        id="m3_01",
        category="mate-in-3",
        fen="r4rk1/ppp2ppp/8/8/8/8/PPP2PPP/R4RK1 w - - 0 1",
        correct_first_moves=["f1f8", "a1a8"],
        description="Rook sac on open file leads to forced mate",
    ),
    Puzzle(
        id="m3_02",
        category="mate-in-3",
        fen="8/8/8/8/8/1K6/8/R6k w - - 0 1",
        correct_first_moves=["a1h1", "a1a2", "b3c2"],
        description="K+R vs K endgame — force mate in 3",
    ),
    Puzzle(
        id="m3_03",
        category="mate-in-3",
        fen="8/8/8/8/8/2K5/8/1R5k w - - 0 1",
        correct_first_moves=["b1h1", "b1b2", "c3d2"],
        description="K+R vs K — edge-of-board mating net",
    ),
]

# Group by category for summary
CATEGORIES = ["mate-in-1", "mate-in-2", "mate-in-3"]
# Required search depth per category
REQUIRED_DEPTH = {"mate-in-1": 1, "mate-in-2": 3, "mate-in-3": 5}

# ---------------------------------------------------------------------------
# ASCII board renderer
# ---------------------------------------------------------------------------

PIECE_SYMBOLS = {
    (chess.PAWN,   chess.WHITE): "♙", (chess.PAWN,   chess.BLACK): "♟",
    (chess.KNIGHT, chess.WHITE): "♘", (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.WHITE): "♗", (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK,   chess.WHITE): "♖", (chess.ROOK,   chess.BLACK): "♜",
    (chess.QUEEN,  chess.WHITE): "♕", (chess.QUEEN,  chess.BLACK): "♛",
    (chess.KING,   chess.WHITE): "♔", (chess.KING,   chess.BLACK): "♚",
}


def ascii_board(board: chess.Board, highlight: chess.Move | None = None) -> str:
    highlight_squares: set[int] = set()
    if highlight:
        highlight_squares = {highlight.from_square, highlight.to_square}

    lines = []
    lines.append("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")
    for rank in range(7, -1, -1):
        row = f"{rank + 1} │"
        for file_ in range(8):
            sq = chess.square(file_, rank)
            piece = board.piece_at(sq)
            symbol = PIECE_SYMBOLS.get((piece.piece_type, piece.color), " ") if piece else " "
            marker = "*" if sq in highlight_squares else " "
            row += f"{marker}{symbol}{marker}│"
        lines.append(row)
        if rank > 0:
            lines.append("  ├───┼───┼───┼───┼───┼───┼───┼───┤")
    lines.append("  └───┴───┴───┴───┴───┴───┴───┴───┘")
    lines.append("    a   b   c   d   e   f   g   h  ")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def save_svg(board: chess.Board, puzzle: Puzzle, move: chess.Move | None,
             solved: bool):
    BOARDS_DIR.mkdir(parents=True, exist_ok=True)
    arrows = []
    if move:
        color = "#00aa00" if solved else "#cc0000"
        arrows = [chess.svg.Arrow(move.from_square, move.to_square, color=color)]
    svg = chess.svg.board(board, arrows=arrows, size=360)
    out = BOARDS_DIR / f"puzzle_{puzzle.id}.svg"
    out.write_text(svg)
    return out


# ---------------------------------------------------------------------------
# Puzzle runner
# ---------------------------------------------------------------------------

def run_puzzle(puzzle: Puzzle, skill_code: str | None = None,
               depth: int = SEARCH_DEPTH,
               skills_text: str | None = None) -> bool:
    """
    Test a single puzzle.
    Pass skill_code for minimax mode, skills_text for cognitive mode.
    """
    board = chess.Board(puzzle.fen)
    cognitive_mode = skills_text is not None
    effective_depth = max(depth, REQUIRED_DEPTH[puzzle.category])

    mode_label = "Cognitive (LLM)" if cognitive_mode else f"Minimax depth-{effective_depth}"
    print(f"\n{'═' * 62}")
    print(f"  [{puzzle.category.upper()}] {puzzle.id}: {puzzle.description}")
    print(f"  Mode: {mode_label}")
    print(f"  Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"  FEN: {puzzle.fen}")
    print(f"{'─' * 62}")
    print(ascii_board(board))

    # Pick a move
    if cognitive_mode:
        move = get_cognitive_move(board, skills_text)
    else:
        move = get_skill_move(board, skill_code, depth=effective_depth)

    move_uci = move.uci() if move else "none"
    solved = move_uci in puzzle.correct_first_moves

    result_tag = "✓ SOLVED" if solved else "✗ MISSED"
    print(f"\n  AI played  : {move_uci}  ({board.san(move) if move else '?'})")
    print(f"  Expected   : {puzzle.correct_first_moves}")
    print(f"  Result     : {result_tag}")

    if move:
        after = board.copy()
        after.push(move)
        print(f"\n  Board after AI move:")
        print(ascii_board(after, highlight=None))

    svg_path = save_svg(board, puzzle, move, solved)
    print(f"  SVG saved  : {svg_path}")
    return solved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Skill AI / Cognitive AI puzzle accuracy test")
    parser.add_argument("--skill", default=str(SKILLS_DIR / "skill_v0.py"),
                        help="Path to the minimax skill .py file")
    parser.add_argument("--depth", type=int, default=SEARCH_DEPTH,
                        help="Base minimax depth (raised per puzzle category)")
    parser.add_argument("--cognitive", metavar="SKILLS_MD",
                        help="Path to a SKILLS_vN.md playbook — enables Cognitive (LLM) mode")
    args = parser.parse_args()

    cognitive_mode = args.cognitive is not None
    skill_code: str | None = None
    skills_text: str | None = None

    if cognitive_mode:
        playbook_path = Path(args.cognitive)
        if not playbook_path.exists():
            print(f"ERROR: playbook not found: {playbook_path}")
            sys.exit(1)
        skills_text = playbook_path.read_text(encoding="utf-8")
        print(f"\nMode       : Cognitive (LLM reads SKILLS.md)")
        print(f"Playbook   : {playbook_path}")
    else:
        skill_path = Path(args.skill)
        if not skill_path.exists():
            print(f"ERROR: skill file not found: {skill_path}")
            sys.exit(1)
        skill_code = skill_path.read_text(encoding="utf-8")
        if load_evaluate_fn(skill_code) is None:
            print("ERROR: skill code failed to load — check syntax")
            sys.exit(1)
        print(f"\nMode       : Minimax")
        print(f"Skill file : {skill_path}")
        print(f"Depth      : {args.depth} (raised to minimum per category)")

    results: dict[str, list[bool]] = {c: [] for c in CATEGORIES}

    for puzzle in PUZZLES:
        solved = run_puzzle(
            puzzle,
            skill_code=skill_code,
            depth=args.depth,
            skills_text=skills_text,
        )
        results[puzzle.category].append(solved)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print("  ACCURACY SUMMARY")
    print(f"{'─' * 62}")
    total_correct = 0
    total_puzzles = 0
    for cat in CATEGORIES:
        cat_results = results[cat]
        correct = sum(cat_results)
        total = len(cat_results)
        pct = 100 * correct / total if total else 0
        bar = "█" * correct + "░" * (total - correct)
        print(f"  {cat:<14} {bar}  {correct}/{total}  ({pct:.0f}%)")
        total_correct += correct
        total_puzzles += total

    overall = 100 * total_correct / total_puzzles if total_puzzles else 0
    label = Path(args.cognitive).name if cognitive_mode else Path(args.skill).name
    print(f"{'─' * 62}")
    print(f"  OVERALL        {total_correct}/{total_puzzles}  ({overall:.0f}%)")
    print(f"  Source     : {label}")
    print(f"  SVG boards : {BOARDS_DIR}/")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()
