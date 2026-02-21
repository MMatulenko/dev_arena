"""
Real-world agent loop utilizing Tools.
Supports both Anthropic (claude) and OpenAI (gpt-4o-mini) providers via LLM_PROVIDER env var.
OpenAI gets automatic prompt caching for free on prefixes > 1024 tokens.
"""

import json
import logging
from pathlib import Path
from typing import Any

from config import (
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    OPENAI_API_KEY, OPENAI_JUNIOR_MODEL,
    LLM_PROVIDER, DEV_CODER_MAX_TOKENS, DEV_TEMPERATURE,
)
from dev_arena import CostLedger, SESSION_LEDGER
from tools import Tools

logger = logging.getLogger(__name__)

_REAL_WORLD_JUNIOR_PROMPT = """\
You are a Software Engineer solving a real-world task.
You have access to a workspace and tools to read, edit, and run files.

=== YOUR DEVELOPER PLAYBOOK (CORE_PRINCIPLES.md) ===
{skills}
================================================

TASK:
{task_description}

TOOL RULES (read before acting):
- AVAILABLE TOOLS: `file_read`, `file_write`, `file_patch`, `shell`, `grep`, `glob_search`, `run_tests`
- There is NO tool called `bash`. NEVER call `bash`. Use `shell` to run any shell command.
- Prefer specific tools: use `grep` instead of `shell('grep ...')`, `glob_search` instead of `shell('find ...')`.

INSTRUCTIONS:
1. EXPLORE: Use `file_read`, `grep`, or `glob_search` to find relevant code. Identify the exact location of the bug.
2. PLAN: Think step-by-step about what needs to be changed.
3. BUILD: Use `file_patch` for targeted edits, `file_write` for new files only.
4. VERIFY: Call `run_tests` to run the official CI test suite. DO NOT write your own test scripts. DO NOT try to run `pytest` via `shell`. Let the CI Oracle do the work.
5. DECLARE VICTORY: If `run_tests` returns `[CI ORACLE: ALL TESTS PASSED]`, IMMEDIATELY output "VERIFICATION: PASSED" and stop. IGNORE any deprecation warnings or unrelated output as long as the oracle says it passed! Do not double check.
6. REPEAT until solved.
7. When done, output "VERIFICATION: PASSED" if tests pass, or "VERIFICATION: FAILED" if you give up.
"""

# Anthropic tool definitions
_TOOLS_DEF = [
    {
        "name": "file_read",
        "description": "Reads a file from the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "offset": {"type": "integer", "description": "Starting line number as an integer (e.g. 241). DO NOT pass a list or array.", "default": 1},
                "limit": {"type": "integer", "description": "Number of lines to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "file_write",
        "description": "Writes a completely new file. DO NOT use on existing files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "content": {"type": "string", "description": "Full file content"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "file_patch",
        "description": "Applies a targeted string replacement to an existing file. Use this for ALL edits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "old_string": {"type": "string", "description": "Existing text to replace. Must match exactly unless use_regex is True."},
                "new_string": {"type": "string", "description": "The new text."},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences? default False.", "default": False},
                "use_regex": {"type": "boolean", "description": "Treat old_string as a regex pattern.", "default": False}
            },
            "required": ["path", "old_string", "new_string"]
        }
    },
    {
        "name": "shell",
        "description": "Executes a shell command in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)", "default": 60},
                "env": {
                    "type": "object",
                    "description": "Optional environment variables to inject (e.g. {'PYTHONPATH': '/my/path'})",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "grep",
        "description": "Searches for a regex pattern in files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Optional relative path to search within"},
                "glob_filter": {"type": "string", "description": "Optional glob pattern to filter files (e.g. '*.py')"},
                "context_lines": {"type": "integer", "description": "Number of lines of context to show around matches", "default": 0},
                "case_insensitive": {"type": "boolean", "description": "Whether to perform a case-insensitive search", "default": False},
                "output_mode": {"type": "string", "description": "Output mode: 'content', 'files', or 'count'", "default": "content"}
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "glob_search",
        "description": "Searches for files matching a glob pattern (e.g. '*.py'). This tool matches file paths/names ONLY, it does NOT search file contents. Use `grep` to search for text inside files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', '**/tests/*')"},
                "path": {"type": "string", "description": "Optional relative path to search within"}
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "run_tests",
        "description": "Runs the official test suite for this task and returns pass/fail results. Use this to verify your fix works. No arguments needed.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# OpenAI function-call format (input_schema → parameters, wrapped in "function" key)
_TOOLS_DEF_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        }
    }
    for t in _TOOLS_DEF
]


def _execute_tool(name: str, args: dict, tools: Any) -> str:
    """Shared tool dispatcher used by both provider branches."""
    if name == "file_read":
        return tools.file_read(**args)
    elif name == "file_write":
        return tools.file_write(**args)
    elif name == "file_patch":
        return tools.file_patch(**args)
    elif name in ("shell", "bash"):
        if name == "bash":
            logger.warning("[Tool alias] 'bash' called — routing to 'shell'")
        return tools.shell(**args)
    elif name == "grep":
        return tools.grep(**args)
    elif name == "glob_search":
        return tools.glob_search(**args)
    elif name == "run_tests":
        if hasattr(tools, "run_tests"):
            return tools.run_tests()
        return "run_tests: not available in this environment (no CI oracle configured)"
    else:
        return "Error: unknown tool %s" % name


def _save_trajectory(trajectory: list[str]) -> str:
    trajectory_str = "\n".join(trajectory)
    with open("last_trajectory.txt", "w") as f:
        f.write(trajectory_str)
    return trajectory_str


def run_real_world_sprint(task_name: str, task_description: str, skills_text: str,
                          workspace: Path,
                          max_iterations: int = 15,
                          sprint_ledger: CostLedger | None = None,
                          tools: Any = None) -> tuple[bool, str, str]:
    """
    Runs the RealWorldJunior agent in a loop until it declares PASS/FAIL or hits max_iterations.
    Dispatches to OpenAI (gpt-4o-mini, auto-cached) or Anthropic based on LLM_PROVIDER.
    Returns: (success, reasoning_trace, feedback)
    """
    logger.info("RealWorldJunior waking up for task: %s (provider=%s)", task_name, LLM_PROVIDER)
    if tools is None:
        tools = Tools(workspace)

    provider = LLM_PROVIDER.lower()
    system_prompt = _REAL_WORLD_JUNIOR_PROMPT.format(
        skills=skills_text,
        task_description=task_description,
    )
    trajectory: list[str] = []

    # ------------------------------------------------------------------ OpenAI
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # System prompt goes as first message; long prefix is auto-cached by OpenAI
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Begin."},
        ]

        for iteration in range(max_iterations):
            logger.debug("RealWorldJunior (OpenAI) iteration %d", iteration + 1)

            resp = client.chat.completions.create(
                model=OPENAI_JUNIOR_MODEL,
                max_tokens=DEV_CODER_MAX_TOKENS,
                temperature=DEV_TEMPERATURE,
                messages=messages,
                tools=_TOOLS_DEF_OPENAI,
                tool_choice="auto",
            )

            in_tok = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens
            cached_tok = getattr(resp.usage, "prompt_tokens_details", None)
            if cached_tok:
                cached = getattr(cached_tok, "cached_tokens", 0)
                logger.info("OpenAI cache hit: %d/%d input tokens cached", cached, in_tok)

            call_cost = SESSION_LEDGER.add(in_tok, out_tok)
            if sprint_ledger:
                sprint_ledger.add(in_tok, out_tok)
            logger.info("Junior (OpenAI) cost: $%.4f | session=$%.4f [in=%d out=%d tok]",
                        call_cost, SESSION_LEDGER.total_cost, in_tok, out_tok)

            choice = resp.choices[0]
            msg_obj = choice.message
            text_content = msg_obj.content or ""
            oai_tool_calls = msg_obj.tool_calls or []

            # Append assistant turn to history
            assistant_turn: dict = {"role": "assistant", "content": text_content}
            if oai_tool_calls:
                assistant_turn["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in oai_tool_calls
                ]
            messages.append(assistant_turn)

            if text_content:
                trajectory.append("AGENT:\n%s\n" % text_content)
                if text_content.strip():
                    logger.info("[Agent thought] %s", text_content.strip()[:300])
                if "VERIFICATION: PASSED" in text_content:
                    return True, _save_trajectory(trajectory), "Success"
                if "VERIFICATION: FAILED" in text_content:
                    return False, _save_trajectory(trajectory), "Agent declared failure."

            if not oai_tool_calls:
                messages.append({"role": "user", "content": "You did not use any tools and did not output VERIFICATION: PASSED or FAILED. Please continue."})
                trajectory.append("SYSTEM: Model returned no tools and no completion marker.")
                continue

            for tc in oai_tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                trajectory.append("TOOL CALL: %s(%s)" % (name, json.dumps(args)))
                logger.info("[Tool call ] %s(%s)", name, json.dumps(args)[:200])

                try:
                    res = _execute_tool(name, args, tools)
                except Exception as e:
                    res = "Tool execution failed: %s" % e

                if name == "run_tests" and "[CI ORACLE: ALL TESTS PASSED]" in res:
                    logger.info("[Auto-stop] CI Oracle confirmed all tests passed.")
                    trajectory.append("TOOL RESULT:\n%s\n" % res)
                    trajectory.append("SYSTEM: Auto-stopped — CI Oracle confirmed success.")
                    return True, _save_trajectory(trajectory), "Success"

                logger.info("[Tool result] %s", res.split("\n")[0][:200] if res else "")
                trajectory.append("TOOL RESULT:\n%s\n" % res)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": res})

        return False, _save_trajectory(trajectory), "Max iterations (%d) reached without declaring success." % max_iterations

    # --------------------------------------------------------------- Anthropic
    import anthropic
    client_ant = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    system_block = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    ant_messages: list[dict] = [{"role": "user", "content": "Begin."}]

    for iteration in range(max_iterations):
        logger.debug("RealWorldJunior (Anthropic) iteration %d", iteration + 1)

        msg = client_ant.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=DEV_CODER_MAX_TOKENS,
            temperature=DEV_TEMPERATURE,
            system=system_block,
            messages=ant_messages,
            tools=_TOOLS_DEF,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

        in_tok, out_tok = msg.usage.input_tokens, msg.usage.output_tokens
        call_cost = SESSION_LEDGER.add(in_tok, out_tok)
        if sprint_ledger:
            sprint_ledger.add(in_tok, out_tok)

        ant_messages.append({"role": "assistant", "content": msg.content})

        text_blocks = [blk.text for blk in msg.content if blk.type == "text"]
        if text_blocks:
            reasoning = text_blocks[-1]
            trajectory.append("AGENT:\n%s\n" % reasoning)
            if reasoning.strip():
                logger.info("[Agent thought] %s", reasoning.strip()[:300])
            if "VERIFICATION: PASSED" in reasoning:
                return True, _save_trajectory(trajectory), "Success"
            if "VERIFICATION: FAILED" in reasoning:
                return False, _save_trajectory(trajectory), "Agent declared failure."

        tool_results = []
        for blk in msg.content:
            if blk.type != "tool_use":
                continue
            name = blk.name
            args = blk.input
            trajectory.append("TOOL CALL: %s(%s)" % (name, json.dumps(args)))
            logger.info("[Tool call ] %s(%s)", name, json.dumps(args)[:200])

            try:
                res = _execute_tool(name, args, tools)
            except Exception as e:
                res = "Tool execution failed: %s" % e

            if name == "run_tests" and "[CI ORACLE: ALL TESTS PASSED]" in res:
                logger.info("[Auto-stop] CI Oracle confirmed all tests passed.")
                trajectory.append("TOOL RESULT:\n%s\n" % res)
                trajectory.append("SYSTEM: Auto-stopped — CI Oracle confirmed success.")
                return True, _save_trajectory(trajectory), "Success"

            logger.info("[Tool result] %s", res.split("\n")[0][:200] if res else "")
            trajectory.append("TOOL RESULT:\n%s\n" % res)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": blk.id,
                "content": res,
            })

        if not tool_results:
            trajectory.append("SYSTEM: Model returned no tools and no completion marker.")
            ant_messages.append({"role": "user", "content": "You did not use any tools and did not output VERIFICATION: PASSED or FAILED. Please continue."})
        else:
            ant_messages.append({"role": "user", "content": tool_results})

    return False, _save_trajectory(trajectory), "Max iterations (%d) reached without declaring success." % max_iterations
