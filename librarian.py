"""
Playbook Librarian
==================
Periodically consolidates DEV_SKILLS.md by:
  - Merging near-duplicate rules into single precise rules
  - Resolving contradictions (more specific rule wins)
  - Removing accumulated "(New)" markers
  - Regrouping scattered related rules into proper sections

Called automatically from dev_loop every LIBRARIAN_EVERY_N commits.
The result is committed with message: "librarian: N→M lines"
"""

import logging
import re

from config import DEV_COACH_MAX_TOKENS, DEV_TEMPERATURE
from dev_arena import SESSION_LEDGER, _call_llm

logger = logging.getLogger(__name__)

_LIBRARIAN_PROMPT = """\
You are a senior technical writer consolidating a Developer Playbook.

The playbook has grown to {n_lines} lines through incremental patches and now
contains near-duplicates, contradictions, and scattered related rules.

=== CURRENT DEV_SKILLS.md ===
{skills}
=== END ===

YOUR TASK — output a clean, consolidated version:
1. Merge near-duplicate rules into one precise, concrete rule.
2. Resolve contradictions — keep the more specific/recently-added rule.
3. Group related rules into the closest existing section (or create a new ## section).
4. Remove ALL "(New)" markers — every rule is now permanent.
5. Keep EVERY unique principle — do not silently drop any concept.
6. Keep section headings (## 1., ## 2., etc.) and the overall structure.

Output ONLY the raw Markdown — no fences, no preamble, no explanation.
Target: consolidate without losing anything. Shorter is better.
"""


def consolidate(skills_text: str,
                sprint_ledger=None) -> tuple[str, str]:
    """
    Ask an LLM to consolidate the playbook.
    Returns (consolidated_text, one_line_summary).
    Returns the original text unchanged if the LLM fails.
    """
    n_lines = skills_text.count("\n") + 1
    prompt = _LIBRARIAN_PROMPT.format(skills=skills_text, n_lines=n_lines)

    try:
        raw, in_tok, out_tok = _call_llm(
            prompt,
            max_tokens=max(DEV_COACH_MAX_TOKENS * 3, 2048),
            temperature=0.1,
        )
        call_cost = SESSION_LEDGER.add(in_tok, out_tok)
        if sprint_ledger is not None:
            sprint_ledger.add(in_tok, out_tok)

        logger.info("Librarian cost: $%.4f | session=$%.4f [in=%d out=%d tok]",
                    call_cost, SESSION_LEDGER.total_cost, in_tok, out_tok)

        result = raw.strip()
        # Sanity: must still be markdown with sections
        if "#" not in result or len(result) < len(skills_text) * 0.3:
            logger.warning("Librarian output looks wrong (too short or no headings) — keeping original")
            return skills_text, "no change"

        n_lines_after = result.count("\n") + 1
        summary = "%d→%d lines" % (n_lines, n_lines_after)
        logger.info("Librarian consolidated: %s", summary)
        return result, summary

    except Exception as exc:
        logger.error("Librarian failed: %s", exc)
        return skills_text, "failed"


def should_run(commit_count: int, every_n: int) -> bool:
    """True when we've hit a multiple of every_n (excluding 0)."""
    return commit_count > 0 and commit_count % every_n == 0
