import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
SKILLS_DIR = BASE_DIR / "skills"
GAMES_DIR = BASE_DIR / "games"

# Engine
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/local/bin/stockfish")
STOCKFISH_SKILL_LEVEL = int(os.getenv("STOCKFISH_SKILL_LEVEL", "0"))
STOCKFISH_MOVE_TIME = float(os.getenv("STOCKFISH_MOVE_TIME", "0.1"))

# Search
SEARCH_DEPTH = int(os.getenv("SEARCH_DEPTH", "3"))
MAX_MOVES_PER_GAME = int(os.getenv("MAX_MOVES_PER_GAME", "150"))

# Learning loop
MAX_GENERATIONS = int(os.getenv("MAX_GENERATIONS", "50"))
WIN_STREAK_TO_STOP = int(os.getenv("WIN_STREAK_TO_STOP", "5"))

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Cognitive arena
COGNITIVE_LLM_MOVES = int(os.getenv("COGNITIVE_LLM_MOVES", "20"))   # LLM for first N White half-moves, minimax for rest
COGNITIVE_MAX_RETRIES = int(os.getenv("COGNITIVE_MAX_RETRIES", "2"))
COGNITIVE_TEMPERATURE = float(os.getenv("COGNITIVE_TEMPERATURE", "0.2"))
COGNITIVE_PLAYER_MAX_TOKENS = int(os.getenv("COGNITIVE_PLAYER_MAX_TOKENS", "300"))
COGNITIVE_COACH_MAX_TOKENS = int(os.getenv("COGNITIVE_COACH_MAX_TOKENS", "2048"))
COGNITIVE_MAX_GENERATIONS = int(os.getenv("COGNITIVE_MAX_GENERATIONS", "999"))
COGNITIVE_WIN_STREAK = int(os.getenv("COGNITIVE_WIN_STREAK", "5"))

# Stockfish analysis
ANALYSIS_DEPTH = int(os.getenv("ANALYSIS_DEPTH", "16"))       # depth for post-move analysis
ANALYSIS_MOVE_TIME = float(os.getenv("ANALYSIS_MOVE_TIME", "0.3"))  # seconds for analysis
SF_LEVEL_WINS_TO_BUMP = int(os.getenv("SF_LEVEL_WINS_TO_BUMP", "2"))  # consecutive wins before level bumps
SF_LEVEL_MAX = int(os.getenv("SF_LEVEL_MAX", "20"))

# ELO tracking (single-agent vs Stockfish)
ELO_INITIAL = int(os.getenv("ELO_INITIAL", "600"))
ELO_K_FACTOR = int(os.getenv("ELO_K_FACTOR", "32"))
ELO_HISTORY_FILE = BASE_DIR / "elo_history.json"

# Duel arena — LLM A vs LLM B
DUEL_ELO_INITIAL = int(os.getenv("DUEL_ELO_INITIAL", "1200"))   # calibrated mid-range start
DUEL_ELO_K_FACTOR = int(os.getenv("DUEL_ELO_K_FACTOR", "32"))
DUEL_ELO_FILE = BASE_DIR / "duel_elo_history.json"
DUEL_GAMES_DIR = BASE_DIR / "duel_games"
DUEL_SKILLS_DIR = SKILLS_DIR / "duel"
DUEL_LLM_MOVES = int(os.getenv("DUEL_LLM_MOVES", "20"))         # LLM half-moves per side; game ends by eval after this
DUEL_MAX_RETRIES = int(os.getenv("DUEL_MAX_RETRIES", "2"))
DUEL_TEMPERATURE = float(os.getenv("DUEL_TEMPERATURE", "0.2"))
DUEL_PLAYER_MAX_TOKENS = int(os.getenv("DUEL_PLAYER_MAX_TOKENS", "300"))
DUEL_COACH_MAX_TOKENS = int(os.getenv("DUEL_COACH_MAX_TOKENS", "600"))  # patch is small — much cheaper
DUEL_MAX_GENERATIONS = int(os.getenv("DUEL_MAX_GENERATIONS", "999"))
DUEL_BLUNDER_CP_THRESHOLD = int(os.getenv("DUEL_BLUNDER_CP_THRESHOLD", "30"))  # centipawns lost = blunder
# Once both agents exhaust LLM moves, Stockfish evaluates and declares winner.
# Positions beyond this threshold (in centipawns) are considered decisive.
DUEL_EVAL_WIN_THRESHOLD = int(os.getenv("DUEL_EVAL_WIN_THRESHOLD", "150"))

# Approximate ELO for Stockfish skill levels (chess.com estimates)
STOCKFISH_ELO_MAP: dict[int, int] = {
    0: 800,  1: 1100, 2: 1300, 3: 1400, 4: 1500,
    5: 1600, 6: 1700, 7: 1800, 8: 1900, 9: 2000, 10: 2100,
    11: 2200, 12: 2300, 13: 2400, 14: 2500, 15: 2600,
    16: 2700, 17: 2800, 18: 2900, 19: 3000, 20: 3200,
}

# LLM pricing ($/million tokens) — update if model changes
# claude-haiku-4-5: $0.80 in / $4.00 out
# claude-sonnet-4-6: $3.00 in / $15.00 out
# gpt-4o-mini: $0.15 in / $0.60 out
_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":          (0.80, 4.00),
    "claude-haiku-4-0":          (0.80, 4.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-sonnet-4-5":         (3.00, 15.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-opus-4-5":           (15.00, 75.00),
    "gpt-4o":                    (2.50, 10.00),
    "gpt-4o-mini":               (0.15, 0.60),
}

def get_pricing() -> tuple[float, float]:
    """Return (price_per_m_input, price_per_m_output) for current model."""
    model = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else OPENAI_MODEL
    return _PRICING.get(model, (1.00, 5.00))  # safe fallback
