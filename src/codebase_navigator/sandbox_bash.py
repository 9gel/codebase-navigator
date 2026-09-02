"""Read-only, whitelisted shell command execution shared by agents.

Provides a small, safe ``bash`` capability used by both the codebase-navigator
``cn ask`` agent and the eval baseline agent, so that both sides of the A/B
comparison get identical, bounded shell access.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "rg",
        "grep",
        "git",
        "find",
        "cat",
        "sed",
        "head",
        "tail",
        "wc",
        "ls",
        "file",
        "sort",
        "uniq",
        "awk",
        "xargs",
    }
)

ALLOWED_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "grep",
        "log",
        "ls-files",
        "show",
        "status",
        "diff",
        "blame",
        "rev-parse",
    }
)

# Shell tokens that would enable redirection, piping, or chaining.
BLOCKED_TOKENS: frozenset[str] = frozenset(
    {">", ">>", "<", "|", ";", "&", "&&", "||", "2>", "&>", "2>&1"}
)

# git flags that could escape the repository or read arbitrary objects.
BLOCKED_GIT_FLAGS: frozenset[str] = frozenset({"-C", "--git-dir", "--work-tree"})

MAX_OUTPUT_LINES = 200
TIMEOUT_SECONDS = 10.0


def _validate_command(command: str) -> tuple[list[str] | None, str | None]:
    """Tokenize and validate a command; return (argv, error)."""
    stripped = command.strip()
    if not stripped:
        return None, "Error: empty command."

    try:
        argv = shlex.split(stripped)
    except ValueError as e:
        return None, f"Error: invalid shell syntax: {e}"

    if not argv:
        return None, "Error: empty command."

    prog = argv[0]
    if "/" in prog or prog not in ALLOWED_COMMANDS:
        return None, (
            f"Error: command '{prog}' is not allowed. "
            f"Allowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}."
        )

    if prog == "git":
        if len(argv) < 2 or argv[1] not in ALLOWED_GIT_SUBCOMMANDS:
            return None, (
                "Error: git subcommand not allowed. "
                f"Allowed git subcommands: {', '.join(sorted(ALLOWED_GIT_SUBCOMMANDS))}."
            )
        for flag in BLOCKED_GIT_FLAGS:
            if flag in argv[2:]:
                return None, f"Error: git flag '{flag}' is not allowed."

    for tok in argv[1:]:
        if tok in BLOCKED_TOKENS:
            return None, f"Error: shell token '{tok}' is not allowed in this sandbox."
        if tok.startswith("$(") or "`" in tok:
            return None, "Error: command substitution is not allowed."

    return argv, None


def run_sandboxed_bash(folder: Path, command: str) -> str:
    """Execute a whitelisted, read-only command in ``folder`` and return its output."""
    argv, error = _validate_command(command)
    if error:
        return error
    assert argv is not None  # guaranteed when error is None

    try:
        res = subprocess.run(
            argv,
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return f"Error: command not found: {argv[0]}"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {TIMEOUT_SECONDS:.0f}s."

    out = res.stdout or ""
    if not out and res.stderr:
        out = res.stderr
    if not out:
        return "No output."

    lines = out.splitlines()
    if len(lines) > MAX_OUTPUT_LINES:
        out = "\n".join(lines[:MAX_OUTPUT_LINES])
        out += f"\n... [{len(lines) - MAX_OUTPUT_LINES} lines truncated]"
    return out


def bash_tool_spec() -> dict[str, Any]:
    """Return an OpenAI-compatible function-call spec for the shared bash tool."""
    allowed = ", ".join(sorted(ALLOWED_COMMANDS))
    git_allowed = ", ".join(sorted(ALLOWED_GIT_SUBCOMMANDS))
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run one read-only shell command. Allowed: " + allowed + ". "
                "git limited to: " + git_allowed + ". No pipes/redirection/substitution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run.",
                    },
                },
                "required": ["command"],
            },
        },
    }
