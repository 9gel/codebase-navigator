from __future__ import annotations

import shlex
from pathlib import Path

from codebase_navigator.sandbox_bash import (
    ALLOWED_COMMANDS,
    ALLOWED_GIT_SUBCOMMANDS,
    _validate_command,
    bash_tool_spec,
    run_sandboxed_bash,
)


def test_allowed_commands_are_read_only():
    assert "rg" in ALLOWED_COMMANDS
    assert "git" in ALLOWED_COMMANDS
    for forbidden in ("rm", "mv", "curl", "wget", "sh", "bash", "python", "python3"):
        assert forbidden not in ALLOWED_COMMANDS


def test_git_subcommands_are_read_only():
    assert "grep" in ALLOWED_GIT_SUBCOMMANDS
    assert "log" in ALLOWED_GIT_SUBCOMMANDS
    for forbidden in ("push", "pull", "fetch", "checkout", "commit", "reset", "clean"):
        assert forbidden not in ALLOWED_GIT_SUBCOMMANDS


def test_rejects_disallowed_command(tmp_path: Path):
    out = run_sandboxed_bash(tmp_path, "rm -rf .")
    assert "not allowed" in out


def test_rejects_disallowed_git_subcommand(tmp_path: Path):
    out = run_sandboxed_bash(tmp_path, "git push origin main")
    assert "git subcommand not allowed" in out


def test_rejects_pipe(tmp_path: Path):
    out = run_sandboxed_bash(tmp_path, "ls | wc -l")
    assert "not allowed" in out


def test_rejects_command_substitution(tmp_path: Path):
    out = run_sandboxed_bash(tmp_path, "ls $(id)")
    assert "not allowed" in out


def test_runs_allowed_command(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    out = run_sandboxed_bash(tmp_path, "cat a.txt")
    assert "hello" in out


def test_empty_command(tmp_path: Path):
    assert "empty" in run_sandboxed_bash(tmp_path, "   ").lower()


def test_bash_tool_spec_shape():
    spec = bash_tool_spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "bash"
    assert "command" in spec["function"]["parameters"]["properties"]
    assert spec["function"]["parameters"]["required"] == ["command"]


def test_validate_command_invalid_syntax():
    argv, err = _validate_command("ls 'unterminated")
    assert argv is None
    assert err is not None
    assert "invalid" in err.lower()


def test_tokenization_respects_quotes():
    argv, err = _validate_command("rg 'foo bar' src/")
    assert err is None
    assert argv == ["rg", "foo bar", "src/"]


def test_shlex_roundtrip():
    assert shlex.split("git grep -n 'pattern'") == ["git", "grep", "-n", "pattern"]


def test_disallowed_flag_rejected(tmp_path: Path):
    out = run_sandboxed_bash(tmp_path, "git -C /etc status")
    assert "not allowed" in out
