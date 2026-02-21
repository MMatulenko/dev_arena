import logging
import re
from config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert chess AI researcher specializing in writing Python evaluation functions.
You write clean, correct Python code using the python-chess library (import chess).
You ONLY output a single Python function named evaluate_board(board: chess.Board) -> float.
Positive return values mean White is winning; negative mean Black is winning.
Do NOT include any explanation, markdown fences, or imports — output raw Python code only.
"""

_IMPROVE_TEMPLATE = """\
You are a chess AI that learns by rewriting its own Python strategy code.

## Match Result
You played as White against Stockfish (Skill Level {skill_level}) and the result was: {result}

## Game Moves (UCI)
{moves}

## Your Current evaluate_board() Code
```python
{current_code}
```

## Task
Analyze the game. Identify strategic weaknesses (e.g., poor king safety, ignoring center,
hanging pieces, weak endgame). Then write an IMPROVED evaluate_board(board) function that
addresses these weaknesses.

Rules:
- Use only the python-chess library (chess module is already available as `chess`).
- Return a float: positive = good for White, negative = good for Black.
- The function signature must be exactly: def evaluate_board(board: chess.Board) -> float
- Output ONLY the raw Python function — no imports, no markdown, no explanation.
"""

_INITIAL_SKILL_TEMPLATE = """\
Write a Python evaluate_board(board: chess.Board) -> float function for a chess-playing AI.

Requirements:
- Use only the python-chess library (chess module available as `chess`).
- Positive return = good for White, negative = good for Black.
- Include at minimum: material counting, center control, mobility, king safety.
- Function signature must be exactly: def evaluate_board(board: chess.Board) -> float
- Output ONLY the raw Python function — no imports, no markdown, no explanation.
"""

# ---------------------------------------------------------------------------
# Code extractor / validator
# ---------------------------------------------------------------------------

def _extract_code(raw: str) -> str:
    """Strip markdown fences and return only the Python function text."""
    # Remove ```python ... ``` or ``` ... ```
    raw = re.sub(r"```(?:python)?\n?", "", raw)
    raw = raw.replace("```", "").strip()
    return raw


def _validate_code(code: str) -> bool:
    """Return True if the code compiles and defines evaluate_board."""
    try:
        import chess as _chess
        compiled = compile(code, "<skill>", "exec")
        local_env = {"chess": _chess}
        exec(compiled, local_env)
        if "evaluate_board" not in local_env or not callable(local_env["evaluate_board"]):
            logger.error("LLM response does not define a callable evaluate_board()")
            return False
        return True
    except SyntaxError as exc:
        logger.error("LLM returned code with SyntaxError: %s", exc)
        return False
    except Exception as exc:
        logger.error("LLM code failed validation: %s", exc)
        return False


# ---------------------------------------------------------------------------
# LLM adapters
# ---------------------------------------------------------------------------

def _call_anthropic(system: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def _call_llm(user_prompt: str) -> str:
    provider = LLM_PROVIDER.lower()
    logger.info("Calling LLM provider: %s", provider)
    if provider == "anthropic":
        return _call_anthropic(_SYSTEM_PROMPT, user_prompt)
    elif provider == "openai":
        return _call_openai(_SYSTEM_PROMPT, user_prompt)
    else:
        raise ValueError("Unknown LLM_PROVIDER: %s. Use 'anthropic' or 'openai'." % provider)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_initial_skill() -> str | None:
    """
    Ask the LLM to produce a starting evaluate_board() stronger than pure material.
    Returns validated Python code string, or None on failure.
    """
    logger.info("Requesting initial skill from LLM")
    try:
        raw = _call_llm(_INITIAL_SKILL_TEMPLATE)
        code = _extract_code(raw)
        if _validate_code(code):
            logger.info("Initial skill generated successfully")
            return code
        logger.error("Initial skill failed validation")
        return None
    except Exception as exc:
        logger.error("LLM call failed for initial skill: %s", exc)
        return None


def improve_skill(current_code: str, pgn: str, result: str,
                  skill_level: int = 0, retries: int = 2) -> str | None:
    """
    Send the game result + current code to the LLM and ask for an improved version.
    Returns validated Python code string, or None if all retries fail.
    """
    # Extract UCI move list from PGN for a compact representation
    moves = _extract_uci_moves(pgn)

    prompt = _IMPROVE_TEMPLATE.format(
        skill_level=skill_level,
        result=result,
        moves=", ".join(moves) if moves else "(no moves recorded)",
        current_code=current_code,
    )

    for attempt in range(1, retries + 1):
        logger.info("Skill improvement attempt %d/%d", attempt, retries)
        try:
            raw = _call_llm(prompt)
            code = _extract_code(raw)
            if _validate_code(code):
                logger.info("Improved skill validated on attempt %d", attempt)
                return code
        except Exception as exc:
            logger.error("LLM call failed on attempt %d: %s", attempt, exc)

    logger.error("All %d improvement attempts failed", retries)
    return None


def _extract_uci_moves(pgn: str) -> list[str]:
    """Pull UCI moves out of a PGN string (stored in comment or headers by arena.py)."""
    import chess.pgn
    import io
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is None:
            return []
        board = game.board()
        uci_moves = []
        for move in game.mainline_moves():
            uci_moves.append(move.uci())
            board.push(move)
        return uci_moves
    except Exception as exc:
        logger.warning("Could not parse PGN for UCI moves: %s", exc)
        return []
