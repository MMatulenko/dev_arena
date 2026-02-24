import os
from pathlib import Path

from dotenv import load_dotenv

# Reuse the API keys from chess_skill_arena's .env so they stay in one place
_ENV_PATH = Path(__file__).parent.parent / "chess_skill_arena" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

BASE_DIR = Path(__file__).parent
SKILLS_DIR = BASE_DIR / "skills"
SKILLS_FILE = SKILLS_DIR / "CORE_PRINCIPLES.md"   # single evolving file, tracked by git

# LLM provider — shared with chess_skill_arena
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
ANTHROPIC_COACH_MODEL = os.getenv("ANTHROPIC_COACH_MODEL", "claude-3-5-sonnet-20241022")
OPENAI_JUNIOR_MODEL = os.getenv("OPENAI_JUNIOR_MODEL", "gpt-4o-mini")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", OPENAI_JUNIOR_MODEL)
OPENAI_COACH_MODEL = os.getenv("OPENAI_COACH_MODEL", "gpt-5.2")

# Dev-specific arena settings
DEV_MAX_SPRINTS = int(os.getenv("DEV_MAX_SPRINTS", "8"))
DEV_CODER_MAX_TOKENS = int(os.getenv("DEV_CODER_MAX_TOKENS", "4096"))
DEV_COACH_MAX_TOKENS = int(os.getenv("DEV_COACH_MAX_TOKENS", "4096"))  # Swe-bench problems have huge context
DEV_TEMPERATURE = float(os.getenv("DEV_TEMPERATURE", "0.2"))
DEV_MAX_RETRIES = int(os.getenv("DEV_MAX_RETRIES", "2"))
DEV_TEST_TIMEOUT = int(os.getenv("DEV_TEST_TIMEOUT", "8"))    # seconds before killing runaway code

# MBPP curriculum settings
MBPP_WINS_TO_ADVANCE = int(os.getenv("MBPP_WINS_TO_ADVANCE", "2"))
MBPP_TASK_BATCH = int(os.getenv("MBPP_TASK_BATCH", "50"))
# Last N tasks are held out for eval — never seen during training
MBPP_EVAL_TASK_COUNT = int(os.getenv("MBPP_EVAL_TASK_COUNT", "50"))

# Librarian: consolidate playbook every N new commits
LIBRARIAN_EVERY_N = int(os.getenv("LIBRARIAN_EVERY_N", "10"))

# Duel mode
DUEL_ELO_START = int(os.getenv("DUEL_ELO_START", "1000"))
DUEL_ELO_K = int(os.getenv("DUEL_ELO_K", "32"))          # K-factor for ELO update
DUEL_SKILLS_DIR = BASE_DIR / "skills" / "duel"

# LLM pricing ($/million tokens)
_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":          (0.80,  4.00),
    "claude-haiku-4-0":          (0.80,  4.00),
    "claude-3-5-haiku-20241022": (0.80,  4.00),
    "claude-sonnet-4-5":         (3.00,  15.00),
    "claude-sonnet-4-6":         (3.00,  15.00),
    "claude-opus-4-5":           (15.00, 75.00),
    "gpt-4o":                    (2.50,  10.00),
    "gpt-4o-mini":               (0.15,  0.60),
    "gpt-5.2":                   (1.75,  14.00),
}


def get_pricing() -> tuple[float, float]:
    """Return (price_per_m_input, price_per_m_output) for the active model."""
    model = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else OPENAI_JUNIOR_MODEL
    return _PRICING.get(model, (1.00, 5.00))


def get_coach_pricing() -> tuple[float, float]:
    """Return pricing for the coach model."""
    model = ANTHROPIC_COACH_MODEL if LLM_PROVIDER == "anthropic" else OPENAI_COACH_MODEL
    return _PRICING.get(model, (3.00, 15.00))
