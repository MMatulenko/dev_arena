"""
Dev Arena — Core Agents
=======================
Junior Coder Agent  — reads DEV_SKILLS.md + task description, writes a Python function.
Senior Dev Coach    — reads the crash traceback, outputs a surgical JSON patch
                      (old_text / new_text) that is applied to DEV_SKILLS.md in-place.
                      Only the changed fragment is regenerated — existing rules are
                      never touched, token cost is ~75% lower than a full rewrite.
CI Test Runner      — executes the AI's code against a hidden trap suite (the real machine).
"""

import json
import logging
import re
import signal
import time
from dataclasses import dataclass

from config import (
    DEV_TEST_TIMEOUT,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    DEV_COACH_MAX_TOKENS,
    DEV_CODER_MAX_TOKENS,
    DEV_MAX_RETRIES,
    DEV_TEMPERATURE,
    LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL,
    OPENAI_COACH_MODEL, ANTHROPIC_COACH_MODEL,
    get_pricing,
    get_coach_pricing,
)
from tasks import Task

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

    def add(self, in_tok: int, out_tok: int, coach: bool = False) -> float:
        """Record usage; return the incremental cost of this call."""
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.calls += 1
        price_in, price_out = get_coach_pricing() if coach else get_pricing()
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


SESSION_LEDGER = CostLedger()

# ---------------------------------------------------------------------------
# LLM adapter — returns (text, input_tokens, output_tokens)
# Retries on transient 500/529/overloaded errors with exponential backoff.
# ---------------------------------------------------------------------------

_API_RETRY_DELAYS = [3, 8, 20]


def _call_llm(prompt: str, max_tokens: int,
              temperature: float = DEV_TEMPERATURE) -> tuple[str, int, int]:
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
            if any(code in msg_str for code in ("500", "529", "overloaded", "Internal server")):
                if attempt < len(_API_RETRY_DELAYS):
                    continue
            raise


# ---------------------------------------------------------------------------
# Junior Coder Agent
# ---------------------------------------------------------------------------

_JUNIOR_PROMPT = """\
You are a Python Developer.

=== YOUR DEVELOPER PLAYBOOK (CORE_PRINCIPLES.md) ===
{skills}
================================================

TASK:
{task_description}

INSTRUCTIONS:
1. Think step-by-step about the problem and how your Playbook rules apply.
2. Write only the requested Python function — no test code, no `if __name__` block.
3. Output your reasoning first (1-3 sentences), then the code in a ```python block.
"""


def generate_code(task: Task, skills_text: str,
                  sprint_ledger: CostLedger | None = None) -> tuple[str, str]:
    """
    Junior Coder reads CORE_PRINCIPLES.md and writes the function.
    Returns (code, junior_reasoning) — reasoning is the pre-code prose the
    Junior wrote to explain its approach; passed to the Coach on failure.
    """
    logger.info("Junior Coder reading CORE_PRINCIPLES.md and writing code...")
    prompt = _JUNIOR_PROMPT.format(
        skills=skills_text,
        task_description=task.description,
    )

    reply, in_tok, out_tok = _call_llm(prompt, max_tokens=DEV_CODER_MAX_TOKENS)

    call_cost = SESSION_LEDGER.add(in_tok, out_tok)
    if sprint_ledger is not None:
        sprint_ledger.add(in_tok, out_tok)

    logger.info("Junior cost: $%.4f | session=$%.4f [in=%d out=%d tok]",
                call_cost, SESSION_LEDGER.total_cost, in_tok, out_tok)

    # Split reasoning (everything before the code block) from the code itself
    match = re.search(r"```python\n(.*?)\n```", reply, re.DOTALL)
    if match:
        code = match.group(1).strip()
        junior_reasoning = reply[:match.start()].strip()
        if junior_reasoning:
            logger.info("Junior reasoning: %s", junior_reasoning)
        return code, junior_reasoning

    logger.warning("No ```python block found in Junior's reply — using raw output")
    return reply.strip(), ""


# ---------------------------------------------------------------------------
# CI / Test Runner
# ---------------------------------------------------------------------------

class _CodeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _CodeTimeout()


def run_sprint(task: Task, skills_text: str,
               sprint_ledger: CostLedger | None = None) -> tuple[bool, str, str, str]:
    """
    Generate code then run the hidden test suite with a hard timeout.
    Returns (success, code_str, feedback, junior_reasoning).
    """
    code, junior_reasoning = generate_code(task, skills_text, sprint_ledger=sprint_ledger)
    logger.info("--- Generated Code ---\n%s\n----------------------", code)

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(DEV_TEST_TIMEOUT)
    try:
        success, feedback = task.run_tests(code)
    except _CodeTimeout:
        success = False
        feedback = (
            "TIMEOUT: code execution exceeded %ds — likely an infinite loop.\n"
            "Check for loops where the step/increment can become 0 or negative."
            % DEV_TEST_TIMEOUT
        )
        logger.warning("Test timed out after %ds for task %s", DEV_TEST_TIMEOUT, task.name)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return success, code, feedback, junior_reasoning


# ---------------------------------------------------------------------------
# Playbook operation helpers
# ---------------------------------------------------------------------------

def _op_replace(text: str, old: str, new: str) -> tuple[str, bool]:
    """Surgical find-and-replace. Returns (result, success)."""
    if not old:
        return text, False
    old_n = old.replace("\r\n", "\n")
    new_n = new.replace("\r\n", "\n")
    base  = text.replace("\r\n", "\n")
    if old_n not in base:
        return text, False
    return base.replace(old_n, new_n, 1), True


def _op_delete(text: str, fragment: str) -> tuple[str, bool]:
    """
    Remove an exact fragment (typically one bullet line) from the playbook.
    Strips the surrounding blank line if it becomes orphaned.
    Returns (result, success).
    """
    if not fragment:
        return text, False
    norm = text.replace("\r\n", "\n")
    frag = fragment.replace("\r\n", "\n").strip()
    # Match the fragment as a complete line
    lines = norm.splitlines(keepends=True)
    new_lines = [l for l in lines if l.rstrip("\n").strip() != frag]
    if len(new_lines) == len(lines):
        return text, False   # not found
    return "".join(new_lines), True


def _apply_ops(skills_text: str, ops: list[dict]) -> tuple[str, list[str]]:
    """
    Apply a list of Coach operations in order.
    Supported ops: {"op": "replace", "old_text": ..., "new_text": ...}
                   {"op": "delete",  "text": ...}
    Returns (updated_text, list_of_applied_op_descriptions).
    """
    result = skills_text
    applied: list[str] = []

    for i, op in enumerate(ops):
        kind = op.get("op", "")
        if kind == "replace":
            patched, ok = _op_replace(result, op.get("old_text", ""), op.get("new_text", ""))
            if ok:
                result = patched
                applied.append("replace[%d]" % i)
            else:
                # Fallback: append new_text as a new section
                nt = op.get("new_text", "").strip()
                if nt:
                    result = result.rstrip() + "\n\n" + nt + "\n"
                    applied.append("replace[%d]→fallback-append" % i)
                else:
                    logger.warning("Op replace[%d] old_text not found — skipped", i)

        elif kind == "delete":
            cleaned, ok = _op_delete(result, op.get("text", ""))
            if ok:
                result = cleaned
                applied.append("delete[%d]" % i)
            else:
                logger.warning("Op delete[%d] text not found — skipped", i)

        else:
            logger.warning("Unknown op kind '%s' — skipped", kind)

    return result, applied


# ---------------------------------------------------------------------------
# Senior Dev Coach Agent  (multi-op mode, forced tool use for 100% reliable JSON)
# ---------------------------------------------------------------------------

_COACH_SYSTEM = """\
You are a Staff Engineer coaching a Junior Developer.
Your job is to evolve a software engineering playbook (CORE_PRINCIPLES.md) \
based on observed failures. You MUST call exactly one of the three tools provided.
"""

_COACH_PROMPT = """\
The Junior tried to solve this task:
{task_description}

=== JUNIOR'S EXECUTION TRAJECTORY ===
{junior_reasoning}

{bad_code}

=== THE CRASH / TEST FAILURE ===
{error_traceback}

=== CURRENT CORE_PRINCIPLES.md ===
{skills}

*** THE DUAL-AXIS EVOLUTION PROTOCOL ***

Analyze WHY the Junior failed, then call exactly ONE of the three tools:

1. `no_update_required` — if the failure is a test-environment quirk, missing \
   global, or puzzle-specific requirement that cannot generalize.

2. `tool_upgrade_ticket` — if the Junior failed because a tool is physically \
   inadequate (wrong parameter type accepted, output truncated, regex not supported, \
   etc.). A cognitive rule cannot fix a hardware deficiency.

3. `playbook_patch` — if the failure is a genuine logic or architecture flaw. \
   old_text MUST appear VERBATIM in CORE_PRINCIPLES.md above. \
   Never mention specific variable names, test frameworks, or task names in rules.
"""

# Anthropic tool schemas for forced structured output
_COACH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "no_update_required",
            "description": "Signal that no playbook update is warranted (test-env quirk, missing global, puzzle-specific requirement).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "One sentence explaining why no update is needed."}
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_upgrade_ticket",
            "description": "Signal that a physical tool deficiency caused the failure. Recommends a platform-engineer fix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "description": "Name of the deficient tool (e.g. file_patch, file_read)."},
                    "recommendation": {"type": "string", "description": "One sentence: how the platform engineer should fix the Python tool."}
                },
                "required": ["tool_name", "recommendation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "playbook_patch",
            "description": "Update CORE_PRINCIPLES.md with a surgical patch when the Junior made a genuine cognitive/architecture mistake.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string", "description": "One sentence: what universal principle did the Junior miss?"},
                    "reasoning": {"type": "string", "description": "2-4 sentences explaining why this generalizes."},
                    "ops": {
                        "type": "array",
                        "description": "List of patch operations.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"type": "string", "enum": ["replace", "delete"]},
                                "old_text": {"type": "string", "description": "Verbatim text to replace/delete from CORE_PRINCIPLES.md"},
                                "new_text": {"type": "string", "description": "Replacement text (only for 'replace' ops)"},
                                "text": {"type": "string", "description": "Text to delete (only for 'delete' ops)"}
                            },
                            "required": ["op"]
                        }
                    }
                },
                "required": ["diagnosis", "reasoning", "ops"]
            }
        }
    }
]


def reflect_and_learn(task: Task, bad_code: str, error_traceback: str,
                      skills_text: str, generation: int,
                      junior_reasoning: str = "",
                      sprint_ledger: CostLedger | None = None) -> tuple[str, str]:
    """
    Senior Dev Coach updates CORE_PRINCIPLES.md via forced tool use.
    Dispatches to OpenAI or Anthropic based on LLM_PROVIDER.
    """
    logger.info("Senior Dev Coach patching playbook for sprint %d...", generation)
    prompt = _COACH_PROMPT.format(
        task_description=task.description,
        junior_reasoning=junior_reasoning or "(no reasoning captured)",
        bad_code=bad_code,
        error_traceback=error_traceback,
        skills=skills_text,
    )

    provider = LLM_PROVIDER.lower()
    
    try:
        tool_name = ""
        args = {}
        
        if provider == "openai":
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)

            for attempt, delay in enumerate([0] + _API_RETRY_DELAYS):
                if delay:
                    logger.warning("Coach API error — retrying in %ds (attempt %d)", delay, attempt)
                    time.sleep(delay)
                try:
                    resp = client.chat.completions.create(
                        model=OPENAI_COACH_MODEL,
                        max_completion_tokens=DEV_COACH_MAX_TOKENS,
                        messages=[
                            {"role": "system", "content": _COACH_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        tools=_COACH_TOOLS,
                        tool_choice="required",
                    )
                    break
                except Exception:
                    if attempt < len(_API_RETRY_DELAYS):
                        continue
                    raise

            in_tok = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens
            
            call_cost = SESSION_LEDGER.add(in_tok, out_tok, coach=True)
            if sprint_ledger is not None:
                sprint_ledger.add(in_tok, out_tok, coach=True)

            logger.info("Coach (%s) cost: $%.4f | session=$%.4f [in=%d out=%d tok]",
                        OPENAI_COACH_MODEL, call_cost, SESSION_LEDGER.total_cost, in_tok, out_tok)

            msg = resp.choices[0].message
            tool_calls = msg.tool_calls
            if not tool_calls:
                logger.error("Coach returned no tool call — keeping existing playbook")
                return skills_text, ""

            tc = tool_calls[0]
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments)
            
        elif provider == "anthropic":
            from config import ANTHROPIC_COACH_MODEL
            import anthropic
            client_ant = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            # Anthropic tool schemas
            ant_tools = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"]
                }
                for t in _COACH_TOOLS
            ]
            
            # Use cache control for the long prompt
            system_block = [
                {
                    "type": "text",
                    "text": _COACH_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            
            for attempt, delay in enumerate([0] + _API_RETRY_DELAYS):
                if delay:
                    logger.warning("Coach API error — retrying in %ds (attempt %d)", delay, attempt)
                    time.sleep(delay)
                try:
                    resp = client_ant.messages.create(
                        model=ANTHROPIC_COACH_MODEL,
                        max_tokens=DEV_COACH_MAX_TOKENS,
                        temperature=DEV_TEMPERATURE,
                        system=system_block,
                        messages=[{"role": "user", "content": prompt}],
                        tools=ant_tools,
                        tool_choice={"type": "any"},
                        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
                    )
                    break
                except Exception:
                    if attempt < len(_API_RETRY_DELAYS):
                        continue
                    raise
                    
            in_tok = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
            call_cost = SESSION_LEDGER.add(in_tok, out_tok, coach=True)
            if sprint_ledger is not None:
                sprint_ledger.add(in_tok, out_tok, coach=True)

            logger.info("Coach (%s) cost: $%.4f | session=$%.4f [in=%d out=%d tok]",
                        ANTHROPIC_COACH_MODEL, call_cost, SESSION_LEDGER.total_cost, in_tok, out_tok)

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                logger.error("Coach returned no tool call — keeping existing playbook")
                return skills_text, ""
                
            tc = tool_uses[0]
            tool_name = tc.name
            args = tc.input
            
        else:
            raise ValueError("Unknown LLM_PROVIDER: %s" % provider)

        # Apply the selected tool
        if tool_name == "no_update_required":
            logger.info("Coach Gatekeeper: [NO_UPDATE_REQUIRED] — %s", args.get("reason", ""))
            return skills_text, "guardrail rejected update"

        if tool_name == "tool_upgrade_ticket":
            t_name = args.get("tool_name", "?")
            t_rec = args.get("recommendation", "")
            logger.warning("\n*** COACH ISSUED A TOOL UPGRADE TICKET ***\n"
                           "Tool: %s\nRecommendation: %s\n", t_name, t_rec)
            
            # Auto-save the ticket to TOOL_TICKETS.md
            from datetime import datetime
            from config import BASE_DIR
            
            ticket_path = BASE_DIR / "TOOL_TICKETS.md"
            date_str = datetime.now().strftime("%b %d, %Y")
            ticket_content = f"\n### New Ticket: `{t_name}`\n**Date Found:** {date_str}\n**Recommendation:** {t_rec}\n**Status:** 🔴 OPEN\n"
            
            with open(ticket_path, "a") as f:
                f.write(ticket_content)
                
            return skills_text, "tool upgrade ticket issued (playbook unchanged)"

        if tool_name == "playbook_patch":
            diagnosis = args.get("diagnosis", "no diagnosis")
            reasoning = args.get("reasoning", "")
            ops = args.get("ops", [])

            logger.info("Coach diagnosis: %s", diagnosis)
            if reasoning:
                logger.info("Coach reasoning: %s", reasoning)
            logger.info("Coach ops: %d operation(s)", len(ops))

            updated, applied = _apply_ops(skills_text, ops)

            if applied:
                logger.info("Applied ops: %s (%d -> %d chars)",
                            ", ".join(applied), len(skills_text), len(updated))
            else:
                logger.warning("No ops applied — playbook unchanged")

            return updated, diagnosis

        logger.error("Coach called unknown tool '%s'", tool_name)
        return skills_text, ""

    except Exception as exc:
        logger.error("Coach Agent failed: %s", exc)
        return skills_text, ""
