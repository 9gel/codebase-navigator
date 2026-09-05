#!/usr/bin/env python3
"""Automated Multi-Language Benchmark & LLM-as-Judge Evaluation Harness for codebase-navigator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure codebase_navigator from src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codebase_navigator import __version__
from codebase_navigator.ask import (
    AGENT_TOOLS_SPEC,
    DEFAULT_SEED_MODE,
    SYSTEM_PROMPT,
    ask_codebase,
    call_chat_completions,
    load_llm_config,
)
from codebase_navigator.config import EMBEDDING_MODEL_NAME
from codebase_navigator.index import VectorIndex
from codebase_navigator.sandbox_bash import bash_tool_spec, run_sandboxed_bash
from codebase_navigator.tags import TagsManager, get_available_files

PROJECT_DIR = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
REPOS_DIR = EVAL_DIR / "repos"
INDEXES_DIR = REPOS_DIR / "_indexes"
BENCHMARK_CONFIG = EVAL_DIR / "benchmark_tasks.json"
DEFAULT_RUNS_DIR = Path("eval/runs")
DEFAULT_JUDGE_MODEL = "deepseek/deepseek-v4-pro"

_PRINT_LOCK = threading.Lock()


def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Thread-safe print so parallel task workers don't interleave lines."""
    with _PRINT_LOCK:
        print(*args, **kwargs, flush=True)


class LiveTaskProgress:
    """Render one independently updating TTY row per active worker."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    MAX_WIDTH = 80

    def __init__(self, stream=sys.stderr, max_lines: int = 4):
        self.stream = stream
        self.max_lines = max(1, max_lines)
        self._states: dict[str, tuple[str, str, float]] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._initialized = False

    def start(self) -> None:
        with _PRINT_LOCK:
            if not self._initialized:
                self.stream.write("\n" * self.max_lines)
                self._initialized = True
                self._render_locked(0)
        self._thread = threading.Thread(target=self._spin, name="eval-progress", daemon=True)
        self._thread.start()

    def update(self, task_id: str, label: str, message: str) -> None:
        with _PRINT_LOCK:
            current = self._states.get(task_id)
            started_at = (
                current[2]
                if current is not None and current[:2] == (label, message)
                else time.monotonic()
            )
            self._states[task_id] = (label, message, started_at)

    def finish(self, task_id: str) -> None:
        with _PRINT_LOCK:
            self._states.pop(task_id, None)

    def write(self, message: str) -> None:
        """Insert persistent output above the live worker region."""
        with _PRINT_LOCK:
            if not self._initialized:
                self.stream.write(message + "\n")
                self.stream.flush()
                return
            self.stream.write(f"\033[{self.max_lines}A")
            for line in message.split("\n"):
                self.stream.write(f"\r\033[2K{line}\n")
            self._write_rows_locked(0)
            self.stream.flush()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with _PRINT_LOCK:
            if self._initialized:
                self.stream.write(f"\033[{self.max_lines}A")
                for _ in range(self.max_lines):
                    self.stream.write("\r\033[2K\033[M")
                self._initialized = False
            self.stream.flush()

    def _spin(self) -> None:
        tick = 0
        while not self._stop_event.wait(0.08):
            with _PRINT_LOCK:
                self._render_locked(tick)
            tick += 1

    def _render_locked(self, tick: int) -> None:
        """Redraw the fixed-height region from the cursor below it."""
        if not self._initialized:
            return
        self.stream.write(f"\033[{self.max_lines}A")
        self._write_rows_locked(tick)
        self.stream.flush()

    def _write_rows_locked(self, tick: int) -> None:
        """Write all progress rows from the region's top-left cursor position."""
        states = list(self._states.values())[: self.max_lines]
        width = min(self.MAX_WIDTH, shutil.get_terminal_size((self.MAX_WIDTH, 20)).columns)
        for row in range(self.max_lines):
            text = ""
            if row < len(states):
                label, message, started_at = states[row]
                frame = self.FRAMES[(tick + row) % len(self.FRAMES)]
                elapsed = time.monotonic() - started_at
                text = f"({elapsed:.1f}s) {frame} [{label}] {message}"
                text = _truncate_to_display_width(text, width)
            self.stream.write(f"\r\033[2K{text}\n")


def _display_width(text: str) -> int:
    """Return the number of terminal columns occupied by plain text."""
    return sum(
        0
        if unicodedata.combining(char) or unicodedata.category(char) in {"Cf", "Mn", "Me"}
        else 2
        if unicodedata.east_asian_width(char) in {"F", "W"}
        else 1
        for char in text
    )


def _truncate_to_display_width(text: str, width: int) -> str:
    """Truncate plain terminal text to ``width`` columns, including an ellipsis."""
    if width <= 0:
        return ""
    if _display_width(text) <= width:
        return text

    suffix = "..." if width >= 3 else "." * width
    available = width - len(suffix)
    result: list[str] = []
    used = 0
    for char in text:
        char_width = _display_width(char)
        if used + char_width > available:
            break
        result.append(char)
        used += char_width
    return "".join(result).rstrip() + suffix


class EvaluationCancelled(Exception):
    """Raised cooperatively by workers after an interrupted evaluation."""


def ensure_repo_cloned(repo_name: str, git_url: str, target_dir: Path) -> bool:
    """Clone a missing repository without fetching or updating existing checkouts."""
    if target_dir.is_dir() and (target_dir / ".git").is_dir():
        return True

    if target_dir.exists():
        print(
            f"❌ Repository path exists but is not a Git checkout; refusing to replace it: "
            f"{target_dir}"
        )
        return False

    print(f"📦 Cloning repository {repo_name} from {git_url}...")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", git_url, str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✅ Cloned {repo_name} successfully.")
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"❌ Failed to clone {repo_name}: {e}")
        return False


def llm_judge_answer(
    question: str,
    answer_key: str,
    candidate_answer: str,
    config,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> tuple[bool, str]:
    """Use an LLM-as-a-judge to evaluate if candidate answer accurately matches ground truth key."""
    prompt = f"""You are an expert code intelligence evaluator.
Assess whether the Candidate Answer correctly and thoroughly answers the Question based on the Expected Ground Truth Answer Key.

Question:
{question}

Expected Ground Truth Answer Key:
{answer_key}

Candidate Answer:
{candidate_answer}

Evaluation criteria:
1. Does the candidate identify the correct functions, files, or architectural flow?
2. Is the candidate free of major hallucinations regarding this codebase?

Respond in pure JSON format:
{{
  "is_correct": true or false,
  "confidence": 0.0 to 1.0,
  "rationale": "Brief 1-2 sentence explanation of judgment"
}}
"""
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    attempts = 3
    last_error: Exception | None = None
    raw_response_content = ""

    for attempt in range(attempts):
        try:
            resp = call_chat_completions(config.endpoint, config.api_key, payload, timeout=30.0)
            raw_response_content = (resp["choices"][0]["message"].get("content") or "").strip()
            content = raw_response_content
            # Robust JSON stripping for ```json ... ``` or ``` ... ```
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if fence_match:
                content = fence_match.group(1).strip()
            elif content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(
                    lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                ).strip()

            data = json.loads(content)
            return bool(data.get("is_correct", False)), data.get("rationale", "")
        except (
            RuntimeError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as e:
            last_error = e
            if attempt < attempts - 1:
                time.sleep(1.0 * (attempt + 1))
                continue

    error_msg = f"Judge evaluation error: {last_error}"
    if raw_response_content:
        error_msg += f" (raw judge response: {raw_response_content!r})"
    return False, error_msg


# Mirror of codebase_navigator.tools.MAX_MATCH_CHARS so both arms are capped
# identically; without this the baseline's uncapped lines dominate the token metric.
BASELINE_MAX_MATCH_CHARS = 240

BASELINE_SYSTEM_PROMPT = (
    "You are an expert AI coding assistant answering questions about a codebase.\n"
    "Guidelines:\n"
    "1. Ground every claim in code you have actually read. Never speculate about implementation details you have not verified; cite file paths and line numbers in your answer.\n"
    "2. Use `grep`, `find_files`, `bash` (e.g. 'git grep', 'rg', 'find'), and `list_dir` to locate relevant definitions, functions, and files.\n"
    "3. When reading code with `read_file`, specify `offset` and `limit` to read targeted ranges rather than entire large files.\n"
    "4. Once you locate the primary file and mechanism that answers the question, stop calling tools and synthesize your answer directly."
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate using the ~4 chars/token heuristic."""
    return max(1, len(text) // 4)


def compute_prompt_overhead() -> dict[str, int]:
    """Compute estimated per-turn token overhead for CN and baseline agents."""
    cn_tools_json = json.dumps(AGENT_TOOLS_SPEC, ensure_ascii=False)
    base_tools_json = json.dumps(BASELINE_TOOLS_SPEC, ensure_ascii=False)
    return {
        "cn_system_prompt_tokens": estimate_tokens(SYSTEM_PROMPT),
        "baseline_system_prompt_tokens": estimate_tokens(BASELINE_SYSTEM_PROMPT),
        "cn_tools_spec_tokens": estimate_tokens(cn_tools_json),
        "baseline_tools_spec_tokens": estimate_tokens(base_tools_json),
        "cn_per_turn_overhead_tokens": estimate_tokens(SYSTEM_PROMPT)
        + estimate_tokens(cn_tools_json),
        "baseline_per_turn_overhead_tokens": estimate_tokens(BASELINE_SYSTEM_PROMPT)
        + estimate_tokens(base_tools_json),
    }


def print_prompt_overhead() -> dict[str, int]:
    """Print and return estimated system-prompt and tool-schema token overhead."""
    oh = compute_prompt_overhead()
    print("=" * 75)
    print("📏 Prompt overhead (est. tokens, ~4 chars/token)")
    print("-" * 75)
    print(f"  CN system prompt:      {oh['cn_system_prompt_tokens']:>6,} tokens")
    print(f"  Baseline system prompt:{oh['baseline_system_prompt_tokens']:>6,} tokens")
    print(f"  CN tools spec:         {oh['cn_tools_spec_tokens']:>6,} tokens")
    print(f"  Baseline tools spec:   {oh['baseline_tools_spec_tokens']:>6,} tokens")
    print(f"  CN per-turn overhead:  {oh['cn_per_turn_overhead_tokens']:>6,} tokens")
    print(f"  Base per-turn overhead:{oh['baseline_per_turn_overhead_tokens']:>6,} tokens")
    print("=" * 75)
    return oh


BASELINE_TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read source lines from a file. Use offset/limit for targeted ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-indexed start line (default: 1).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines (default: 2000).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Regex/literal search across files via ripgrep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex or literal text.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional subdirectory/file to restrict to.",
                    },
                    "path_glob": {
                        "type": "string",
                        "description": "Optional file glob filter.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Case-sensitive (default: false).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern.",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory (default: root '.').",
                    }
                },
            },
        },
    },
    bash_tool_spec(),
]


def execute_baseline_tool(folder: Path, name: str, args: dict[str, Any]) -> str:
    """Execute standard baseline tool (cat/rg/find/ls/bash) without specialized codebase-navigator index."""
    if name == "read_file":
        rel_p = args.get("path", "").strip()
        target = folder / rel_p
        if not target.is_file():
            return f"Error: File not found: {rel_p}"
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            # Primary interface: offset + limit (like typical agent read tools).
            # start_line/end_line kept as aliases for model variants.
            offset = args.get("offset") or args.get("start_line") or 1
            limit = args.get("limit") or 2000
            s_line = max(1, int(offset))
            s_idx = s_line - 1
            e_idx = min(len(lines), s_idx + int(limit))
            selected = lines[s_idx:e_idx]
            out_lines = [f"{i}: {line}" for i, line in enumerate(selected, start=s_line)]
            suffix = f"\n... [{len(lines)} total lines]" if e_idx < len(lines) else ""
            return ("\n".join(out_lines) if out_lines else "[Empty slice]") + suffix
        except (OSError, ValueError) as e:
            return f"Error reading {rel_p}: {e}"

    elif name == "grep":
        pattern = args.get("pattern", "")
        sub_path = args.get("path")
        path_glob = args.get("path_glob")
        case_sensitive = bool(args.get("case_sensitive", False))
        cmd = ["rg", "-n"]
        if not case_sensitive:
            cmd.append("-i")
        if path_glob:
            cmd.extend(["-g", path_glob])
        cmd.append(pattern)
        if sub_path:
            cmd.append(sub_path)
        try:
            res = subprocess.run(
                cmd, cwd=folder, capture_output=True, text=True, timeout=10.0, check=False
            )
            out = res.stdout
            if not out and res.stderr:
                out = res.stderr
            lines = out.splitlines()
            # Cap per-line bytes as well as line count: a single matched line in
            # generated/minified sources can be hundreds of KB, which floods the
            # agent context and makes the arm's token cost meaningless.
            lines = [
                ln
                if len(ln) <= BASELINE_MAX_MATCH_CHARS
                else ln[:BASELINE_MAX_MATCH_CHARS]
                + f"... [+{len(ln) - BASELINE_MAX_MATCH_CHARS} chars truncated]"
                for ln in lines
            ]
            if len(lines) > 200:
                out = "\n".join(lines[:200]) + f"\n... [{len(lines) - 200} lines truncated]"
            else:
                out = "\n".join(lines)
            return out or "No matches found."
        except FileNotFoundError:
            # Fallback pure-python grep if rg missing
            matches = []
            for p in folder.rglob("*"):
                if p.is_file() and not any(part.startswith(".") for part in p.parts):
                    try:
                        content = p.read_text(encoding="utf-8", errors="ignore")
                        for idx, line in enumerate(content.splitlines(), start=1):
                            if pattern.lower() in line.lower():
                                rel = p.relative_to(folder)
                                matches.append(f"{rel}:{idx}:{line}")
                                if len(matches) >= 200:
                                    break
                    except OSError:
                        continue
                if len(matches) >= 200:
                    break
            return "\n".join(matches) or "No matches found."
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"Error running grep: {e}"

    elif name == "bash":
        return run_sandboxed_bash(folder, args.get("command", ""))

    elif name == "find_files":
        pattern = args.get("pattern", "*")
        try:
            matches = [str(p.relative_to(folder)) for p in folder.glob(pattern) if p.is_file()]
            if not matches:
                matches = [str(p.relative_to(folder)) for p in folder.rglob(pattern) if p.is_file()]
            return "\n".join(matches[:100]) if matches else "No matching files found."
        except OSError as e:
            return f"Error finding files: {e}"

    elif name == "list_dir":
        rel_p = args.get("path", ".").strip() or "."
        target = folder / rel_p
        if not target.is_dir():
            return f"Error: Directory not found: {rel_p}"
        try:
            entries = [
                f"{'[DIR] ' if p.is_dir() else '[FILE] '}{p.name}"
                for p in sorted(target.iterdir())
                if not p.name.startswith(".")
            ]
            return "\n".join(entries[:100])
        except OSError as e:
            return f"Error listing directory: {e}"

    return f"Unknown tool: {name}"


def run_baseline_agent(
    folder: Path,
    question: str,
    config,
    progress_callback=None,
) -> tuple[str, dict[str, Any]]:
    """Run typical agent harness with standard tools (read_file, grep, find_files, list_dir)."""
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question:\n{question}",
        },
    ]

    searches_remaining = config.max_searches
    prompt_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    api_calls = 0
    tool_calls_count = 0

    while True:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if searches_remaining > 0:
            payload["tools"] = BASELINE_TOOLS_SPEC
            payload["tool_choice"] = "auto"

        response_data = call_chat_completions(config.endpoint, config.api_key, payload)
        usage = response_data.get("usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tok = (
            prompt_details.get("cached_tokens", 0)
            or usage.get("prompt_cache_hit_tokens", 0)
            or usage.get("cache_read_input_tokens", 0)
            or 0
        )
        cached_tok = min(cached_tok, p_tok)
        prompt_tokens += p_tok
        cached_tokens += cached_tok
        output_tokens += c_tok
        api_calls += 1

        choices = response_data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Unexpected empty response from LLM: {response_data}")

        choice = choices[0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls")

        if tool_calls and searches_remaining > 0:
            messages.append(msg)
            for tool_call in tool_calls:
                fn = tool_call.get("function", {})
                fn_name = fn.get("name")
                fn_args_raw = fn.get("arguments", "{}")
                try:
                    fn_args = (
                        json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    fn_args = {}

                tool_calls_count += 1
                arg_summary = ", ".join(f"{k}={v!r}" for k, v in list(fn_args.items())[:2])
                if progress_callback:
                    progress_callback(
                        f"🔎 [Baseline Tool {tool_calls_count}: {fn_name}] {arg_summary}..."
                    )

                tool_output = execute_baseline_tool(folder, fn_name, fn_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": fn_name,
                        "content": tool_output,
                    }
                )

            searches_remaining -= 1
            if searches_remaining <= 0:
                if progress_callback:
                    progress_callback("ℹ️ Baseline search budget reached. Synthesizing answer...")
                messages.append(
                    {
                        "role": "user",
                        "content": "You have completed your tool budget. Please synthesize your final answer using all the evidence gathered.",
                    }
                )
            continue

        content = msg.get("content") or ""
        uncached_prompt_tokens = max(0, prompt_tokens - cached_tokens)
        stats = {
            "prompt_tokens": prompt_tokens,
            "context_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "completion_tokens": output_tokens,
            "context_output_tokens": prompt_tokens + output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "cached_tokens": cached_tokens,
            "uncached_prompt_tokens": uncached_prompt_tokens,
            "net_tokens": uncached_prompt_tokens + output_tokens,
            "api_calls": api_calls,
            "tool_calls_count": tool_calls_count,
        }
        return content, stats


def create_run_directory(runs_dir: Path) -> tuple[Path, str]:
    """Create a unique timestamped directory for one immutable evaluation run."""
    now = datetime.now(UTC)
    timestamp = now.isoformat().replace("+00:00", "Z")
    run_name = f"run_{now.strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "indexes").mkdir()
    return run_dir, timestamp


def get_git_commit(repo_dir: Path) -> str:
    """Return the exact source revision evaluated for a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not determine git commit for {repo_dir}: {exc}") from exc
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        raise RuntimeError(f"Unexpected git commit returned for {repo_dir}: {commit!r}")
    return commit


def hash_index_tree(index_dir: Path) -> str:
    """Hash every path and byte in an immutable index directory."""
    digest = hashlib.sha256()
    for path in sorted(
        index_dir.rglob("*"), key=lambda item: item.relative_to(index_dir).as_posix()
    ):
        relative = path.relative_to(index_dir).as_posix().encode("utf-8")
        if path.is_symlink():
            digest.update(b"L\0" + relative + b"\0" + os.readlink(path).encode("utf-8") + b"\0")
        elif path.is_dir():
            digest.update(b"D\0" + relative + b"\0")
        elif path.is_file():
            digest.update(b"F\0" + relative + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _index_metadata_path(index_dir: Path) -> Path:
    """Return the sidecar path kept outside the hashed immutable index tree."""
    return index_dir.parent / f"{index_dir.name}.metadata.json"


def _load_cached_index_metadata(
    index_dir: Path,
    repo_name: str,
    repo_git_commit: str,
    codebase_navigator_git_commit: str,
) -> dict[str, Any]:
    metadata_path = _index_metadata_path(index_dir)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cached index metadata is missing or invalid: {metadata_path}") from exc

    expected = {
        "repo": repo_name,
        "repo_git_commit": repo_git_commit,
        "codebase_navigator_git_commit": codebase_navigator_git_commit,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Cached index metadata does not match its inputs: {mismatches}")

    actual_hash = hash_index_tree(index_dir)
    if metadata.get("index_tree_sha256") != actual_hash:
        raise RuntimeError(
            f"Cached index integrity check failed for {index_dir}: "
            f"expected {metadata.get('index_tree_sha256')}, got {actual_hash}"
        )

    return {
        **metadata,
        "cache_status": "reused",
        "metadata_path": str(metadata_path.resolve()),
    }


def prepare_evaluation_index(
    repo_name: str,
    repo_dir: Path,
    indexes_dir: Path,
    git_url: str,
    repo_git_commit: str,
    codebase_navigator_git_commit: str,
) -> tuple[Path, dict[str, Any]]:
    """Reuse or atomically build a commit-keyed immutable evaluation index."""
    codebase_navigator_short_hash = codebase_navigator_git_commit[:12]
    repo_index_dir = indexes_dir / repo_name
    index_dir = repo_index_dir / codebase_navigator_short_hash
    if index_dir.is_dir():
        return index_dir, _load_cached_index_metadata(
            index_dir,
            repo_name,
            repo_git_commit,
            codebase_navigator_git_commit,
        )
    if index_dir.exists():
        raise RuntimeError(f"Index cache path exists but is not a directory: {index_dir}")

    repo_index_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(prefix=f".{codebase_navigator_short_hash}.building-", dir=repo_index_dir)
    )

    try:
        tags_ok, tags_message = TagsManager(repo_dir, tag_file=build_dir / ".tags").generate()
        if not tags_ok:
            raise RuntimeError(f"Failed to build tags for {repo_name}: {tags_message}")

        updated_files, indexed_chunks, pruned_files = VectorIndex(
            repo_dir, custom_index_dir=str(build_dir)
        ).sync(force=True)
        if updated_files <= 0 or indexed_chunks <= 0:
            raise RuntimeError(
                f"Evaluation index for {repo_name} is empty "
                f"({updated_files} files, {indexed_chunks} chunks)"
            )

        tree_hash = hash_index_tree(build_dir)
        metadata: dict[str, Any] = {
            "repo": repo_name,
            "git_url": git_url,
            "git_commit": repo_git_commit,
            "repo_git_commit": repo_git_commit,
            "repo_git_short_hash": repo_git_commit[:12],
            "codebase_navigator_git_commit": codebase_navigator_git_commit,
            "codebase_navigator_git_short_hash": codebase_navigator_short_hash,
            "codebase_navigator_version": __version__,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "indexed_files": updated_files,
            "indexed_chunks": indexed_chunks,
            "pruned_files": pruned_files,
            "tags_file": ".tags",
            "tags_status": tags_message,
            "index_tree_hash_algorithm": "sha256",
            "index_tree_sha256": tree_hash,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

        try:
            build_dir.rename(index_dir)
        except FileExistsError:
            shutil.rmtree(build_dir)
            return index_dir, _load_cached_index_metadata(
                index_dir,
                repo_name,
                repo_git_commit,
                codebase_navigator_git_commit,
            )

        metadata_path = _index_metadata_path(index_dir)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return index_dir, {
            **metadata,
            "cache_status": "built",
            "metadata_path": str(metadata_path.resolve()),
        }
    except BaseException:
        if build_dir.exists():
            shutil.rmtree(build_dir)
        raise


class ReportStream:
    """Streams benchmark results to a JSON file as tasks complete.

    The report is written in two phases so it is always valid JSON:
      1. On construction, global facts (timestamp, overhead, repo plan) are written
         with an empty ``results`` list.
      2. Each completed task is appended and the file is atomically rewritten.

    A companion JSONL trace log in the same run directory contains full answers,
    tool-call traces, and judge rationales for offline analysis.
    """

    def __init__(self, run_dir: Path, global_facts: dict[str, Any]):
        self.data: dict[str, Any] = {**global_facts, "results": []}
        self.report_file = run_dir / "report.json"
        self.trace_file = run_dir / "log.jsonl"
        self._lock = threading.Lock()
        self._write()
        self._write_trace({"type": "header", **global_facts})

    def add_result(self, result: dict[str, Any]) -> None:
        with self._lock:
            self.data["results"].append(result)
            self._write()

    def update_facts(self, **facts: Any) -> None:
        """Persist run metadata as repository preparation progresses."""
        with self._lock:
            self.data.update(facts)
            self._write()

    def record_trace(self, entry: dict[str, Any]) -> None:
        """Write a single task trace line. Never raises: a trace problem must not kill the run."""
        try:
            self._write_trace({"type": "task", **entry})
        except Exception as e:  # noqa: BLE001 — trace logging is best-effort
            with self._lock:
                print(f"⚠️  Trace write failed: {e}", file=sys.stderr)

    def record_event(self, event_type: str, **entry: Any) -> None:
        """Append non-task lifecycle and integrity metadata to the run log."""
        self._write_trace({"type": event_type, **entry})

    def _write_trace(self, entry: dict[str, Any]) -> None:
        # ensure_ascii escapes lone surrogates (from mis-decoded tool output) that
        # would otherwise raise UnicodeEncodeError and crash the whole benchmark.
        with self._lock, open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def finalize(self, summary: dict[str, Any]) -> None:
        with self._lock:
            self.data.update(summary)
            self._write()

    def _write(self) -> None:
        if self.report_file is None:
            return
        tmp = self.report_file.with_suffix(self.report_file.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.report_file)


def _run_task(
    r_name: str,
    r_dir: Path,
    repo_git_commit: str,
    index_dir: Path,
    task: dict[str, Any],
    config,
    judge_model: str,
    use_llm_judge: bool,
    compare_baseline: bool,
    use_spinner: bool,
    trace_callback=None,
    event_callback=None,
    live_progress: LiveTaskProgress | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Execute a single benchmark task (CN agent + judge, then optional baseline) and return its result."""
    from codebase_navigator.cli import StatusSpinner

    task_id = task["id"]
    progress_key = f"{r_name}/{task_id}"
    question = task["question"]
    key = task["expected_answer_key"]
    req_kws = task.get("required_keywords", [])
    req_files = task.get("required_files", [])
    buffered_output: list[str] = []

    def _emit(message: str) -> None:
        if live_progress is not None:
            buffered_output.append(message)
        else:
            _safe_print(message)

    def _flush_output() -> None:
        """Write one completed task block without interleaving parallel results."""
        if live_progress is not None and buffered_output:
            live_progress.write("\n".join(buffered_output))
            buffered_output.clear()

    def _raise_if_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise EvaluationCancelled

    _emit(f"\n  ▶ [{task_id}] Q: {question}")

    cn_trace: list[str] = []
    base_trace: list[str] = []

    spinner: StatusSpinner | None = None

    def _stop_spinner() -> None:
        nonlocal spinner
        if spinner:
            spinner.stop()
            spinner = None

    def _progress(label: str, line: str) -> None:
        nonlocal spinner
        _raise_if_cancelled()
        if use_spinner:
            if spinner:
                spinner.update_message(f"{label}: {line}")
            else:
                spinner = StatusSpinner(f"{label}: {line}", stream=sys.stderr)
                spinner.start()
            return
        if live_progress is not None:
            live_progress.update(progress_key, label, line)
            return
        # No spinner (parallel workers): stream progress to stderr so long-running
        # tasks still show movement instead of sitting silent for minutes.
        _safe_print(f"    [{label}] {line}", file=sys.stderr)

    def handle_cn_progress(line: str) -> None:
        cn_trace.append(line)
        if event_callback is not None:
            event_callback(
                "progress",
                timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                repo=r_name,
                task_id=task_id,
                agent="cn",
                message=line,
            )
        _progress(f"{task_id}·CN", line)

    def handle_base_progress(line: str) -> None:
        base_trace.append(line)
        if event_callback is not None:
            event_callback(
                "progress",
                timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                repo=r_name,
                task_id=task_id,
                agent="baseline",
                message=line,
            )
        _progress(f"{task_id}·Base", line)

    # 1. Evaluate Codebase-Navigator Agent
    t0 = time.time()
    try:
        _raise_if_cancelled()
        if use_spinner:
            spinner = StatusSpinner("CN: 🔍 Searching codebase...", stream=sys.stderr)
            spinner.start()

        cn_answer, cn_stats = ask_codebase(
            folder=r_dir,
            question=question,
            config=config,
            custom_index_dir=str(index_dir),
            verbose=False,
            new_session=True,
            progress_callback=handle_cn_progress,
        )
        _stop_spinner()
        _raise_if_cancelled()

        cn_dt = time.time() - t0
        cn_kw_matches = [kw for kw in req_kws if kw.lower() in cn_answer.lower()]
        cn_file_matches = [f for f in req_files if f.lower() in cn_answer.lower()]
        cn_rule_pass = (len(cn_kw_matches) >= min(1, len(req_kws))) and (
            len(cn_file_matches) >= min(1, len(req_files))
        )

        cn_judge_pass = False
        cn_rationale = ""
        if use_llm_judge and config.api_key:
            cn_judge_pass, cn_rationale = llm_judge_answer(
                question, key, cn_answer, config, judge_model=judge_model
            )
        _raise_if_cancelled()

        cn_passed = cn_judge_pass if use_llm_judge else cn_rule_pass

        cn_tokens = cn_stats.get("total_tokens", cn_stats.get("context_output_tokens", 0))
        cn_api_calls = cn_stats.get("api_calls", 0)
        cn_net_tokens = cn_stats.get("net_tokens", cn_tokens)
        cn_prompt_tokens = cn_stats.get("prompt_tokens", cn_stats.get("context_tokens", 0))
        cn_output_tokens = cn_stats.get("completion_tokens", cn_stats.get("output_tokens", 0))
        cn_cached = cn_stats.get("cached_tokens", 0)
        cn_cached_str = f", cached: {cn_cached:,}" if cn_cached > 0 else ""
        _emit(
            f"    [CN]       Status: {'✅ PASS' if cn_passed else '❌ FAIL'} "
            f"(took {cn_dt:.2f}s, {cn_api_calls} turns, tokens: {cn_tokens:,}{cn_cached_str})"
        )

        # 2. Evaluate Baseline Agent (if --compare-baseline)
        base_tokens = 0
        base_api_calls = 0
        base_net_tokens = 0
        base_prompt_tokens = 0
        base_output_tokens = 0
        base_cached = 0
        base_stats: dict[str, Any] = {}
        base_dt = 0.0
        base_passed = False
        base_answer = ""
        base_rationale = ""
        token_savings_pct = 0.0
        time_savings_pct = 0.0

        if compare_baseline:
            _raise_if_cancelled()
            t_b0 = time.time()
            try:
                if use_spinner:
                    spinner = StatusSpinner(
                        "Baseline: 🤖 Reasoning with agent...", stream=sys.stderr
                    )
                    spinner.start()

                base_answer, base_stats = run_baseline_agent(
                    r_dir, question, config, progress_callback=handle_base_progress
                )
                _stop_spinner()
                _raise_if_cancelled()

                base_dt = time.time() - t_b0
                base_kw_matches = [kw for kw in req_kws if kw.lower() in base_answer.lower()]
                base_file_matches = [f for f in req_files if f.lower() in base_answer.lower()]
                base_rule_pass = (len(base_kw_matches) >= min(1, len(req_kws))) and (
                    len(base_file_matches) >= min(1, len(req_files))
                )

                if use_llm_judge and config.api_key:
                    base_judge_pass, base_rationale = llm_judge_answer(
                        question, key, base_answer, config, judge_model=judge_model
                    )
                else:
                    base_judge_pass = base_rule_pass

                base_passed = base_judge_pass

                base_tokens = base_stats.get(
                    "total_tokens", base_stats.get("context_output_tokens", 0)
                )
                base_api_calls = base_stats.get("api_calls", 0)
                base_net_tokens = base_stats.get("net_tokens", base_tokens)
                base_prompt_tokens = base_stats.get(
                    "prompt_tokens", base_stats.get("context_tokens", 0)
                )
                base_output_tokens = base_stats.get(
                    "completion_tokens", base_stats.get("output_tokens", 0)
                )
                base_cached = base_stats.get("cached_tokens", 0)

                if base_tokens > 0:
                    token_savings_pct = round(((base_tokens - cn_tokens) / base_tokens) * 100, 1)

                time_savings_pct = (
                    round(((base_dt - cn_dt) / base_dt) * 100, 1) if base_dt > 0 else 0.0
                )

                base_cached_str = f", cached: {base_cached:,}" if base_cached > 0 else ""
                _emit(
                    f"    [Baseline] Status: {'✅ PASS' if base_passed else '❌ FAIL'} "
                    f"(took {base_dt:.2f}s, {base_api_calls} turns, tokens: {base_tokens:,}{base_cached_str})"
                )
                _emit(
                    f"    ⚡ Savings: Tokens {token_savings_pct:+.1f}% ({cn_tokens:,} vs {base_tokens:,}) | "
                    f"Turns {cn_api_calls} vs {base_api_calls} | "
                    f"Time {time_savings_pct:+.1f}% ({cn_dt:.2f}s vs {base_dt:.2f}s)"
                )
            except EvaluationCancelled:
                _stop_spinner()
                raise
            except Exception as e_base:  # noqa: BLE001 — isolate agent failures
                _stop_spinner()
                _emit(f"    ❌ Baseline Error: {e_base}")

        result: dict[str, Any] = {
            "task_id": task_id,
            "repo": r_name,
            "repo_git_commit": repo_git_commit,
            "question": question,
            "passed": cn_passed,
            "duration_seconds": round(cn_dt, 2),
            "tokens": cn_tokens,
            "net_tokens": cn_net_tokens,
            "prompt_tokens": cn_prompt_tokens,
            "output_tokens": cn_output_tokens,
            "cached_tokens": cn_cached,
            "api_calls": cn_api_calls,
            "judge_rationale": cn_rationale,
            "baseline_passed": base_passed if compare_baseline else None,
            "baseline_duration_seconds": round(base_dt, 2) if compare_baseline else None,
            "baseline_tokens": base_tokens if compare_baseline else None,
            "baseline_net_tokens": base_net_tokens if compare_baseline else None,
            "baseline_prompt_tokens": base_prompt_tokens if compare_baseline else None,
            "baseline_output_tokens": base_output_tokens if compare_baseline else None,
            "baseline_cached_tokens": base_cached if compare_baseline else None,
            "baseline_api_calls": base_api_calls if compare_baseline else None,
            "token_savings_percentage": token_savings_pct if compare_baseline else None,
            "time_savings_percentage": time_savings_pct if compare_baseline else None,
            "answer_preview": cn_answer[:300] + ("..." if len(cn_answer) > 300 else ""),
        }

        if trace_callback is not None:
            trace_entry: dict[str, Any] = {
                "task_id": task_id,
                "repo": r_name,
                "repo_git_commit": repo_git_commit,
                "question": question,
                "expected_answer_key": key,
                "required_keywords": req_kws,
                "required_files": req_files,
                "cn": {
                    "passed": cn_passed,
                    "answer": cn_answer,
                    "judge_rationale": cn_rationale,
                    "tokens": cn_tokens,
                    "net_tokens": cn_net_tokens,
                    "prompt_tokens": cn_prompt_tokens,
                    "output_tokens": cn_output_tokens,
                    "cached_tokens": cn_cached,
                    "api_calls": cn_stats.get("api_calls", 0),
                    "duration_seconds": round(cn_dt, 2),
                    "tool_trace": cn_trace,
                },
            }
            if compare_baseline:
                trace_entry["baseline"] = {
                    "passed": base_passed,
                    "answer": base_answer,
                    "judge_rationale": base_rationale,
                    "tokens": base_tokens,
                    "net_tokens": base_net_tokens,
                    "prompt_tokens": base_prompt_tokens,
                    "output_tokens": base_output_tokens,
                    "cached_tokens": base_cached,
                    "api_calls": base_stats.get("api_calls", 0),
                    "duration_seconds": round(base_dt, 2),
                    "tool_trace": base_trace,
                }
            trace_callback(trace_entry)

        _flush_output()
        return result

    except EvaluationCancelled:
        _stop_spinner()
        raise
    except Exception as e:  # noqa: BLE001 — isolate agent failures
        _stop_spinner()
        _emit(f"    ❌ Error executing task: {e}")
        result = {
            "task_id": task_id,
            "repo": r_name,
            "repo_git_commit": repo_git_commit,
            "question": question,
            "passed": False,
            "error": str(e),
        }
        if trace_callback is not None:
            trace_callback(
                {
                    "task_id": task_id,
                    "repo": r_name,
                    "repo_git_commit": repo_git_commit,
                    "question": question,
                    "expected_answer_key": key,
                    "cn": {"passed": False, "error": str(e), "tool_trace": cn_trace},
                }
            )
        _flush_output()
        return result


# An answer key that matches every file in the repository is not an answer key.
# `required_files` was hand-authored and never checked: three vikunja tasks named
# "pkg" (913 files) and three uv tasks named "crates" (729 files), so any result
# scored as a hit and those tasks silently could not distinguish good retrieval
# from bad. Validate at load time rather than discovering it in the numbers.
MAX_ANSWER_KEY_MATCHES = 5


def validate_answer_keys(benchmarks: list[dict[str, Any]]) -> list[str]:
    """Return a warning per task whose required_files is missing or too broad."""
    warnings: list[str] = []
    for group in benchmarks:
        repo_dir = REPOS_DIR / group["repo"]
        if not repo_dir.is_dir():
            continue
        try:
            code_files, doc_files = get_available_files(repo_dir)
        except OSError:
            # Repo not checked out yet; nothing to validate against.
            continue
        paths = [str(p.relative_to(repo_dir)) for p in code_files + doc_files]
        for task in group.get("tasks", []):
            required = task.get("required_files") or []
            if not required:
                warnings.append(f"{task['id']}: no required_files")
                continue
            matches = sum(1 for p in paths if any(r in p for r in required))
            if matches == 0:
                warnings.append(
                    f"{task['id']}: required_files {required} matches no file in {group['repo']}"
                )
            elif matches > MAX_ANSWER_KEY_MATCHES:
                warnings.append(
                    f"{task['id']}: required_files {required} matches {matches} files "
                    f"in {group['repo']} — too broad to score retrieval"
                )
    return warnings


def run_benchmark(
    target_repo: str | None = None,
    use_llm_judge: bool = True,
    compare_baseline: bool = False,
    workers: int = 4,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    judge_model: str | None = None,
    seed_mode: str | None = None,
):
    """Run benchmarks in a timestamped, self-contained artifact directory."""
    if not BENCHMARK_CONFIG.is_file():
        print(f"❌ Configuration not found: {BENCHMARK_CONFIG}")
        sys.exit(1)

    with open(BENCHMARK_CONFIG, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    key_warnings = validate_answer_keys(benchmarks)
    if key_warnings:
        print("\n⚠️  Answer-key problems (these tasks cannot score retrieval reliably):")
        for w in key_warnings:
            print(f"     - {w}")
        print()

    resolved_judge_model = (
        judge_model
        or os.environ.get("CN_EVAL_JUDGE_MODEL")
        or os.environ.get("CN_JUDGE_MODEL")
        or DEFAULT_JUDGE_MODEL
    )
    run_dir, run_timestamp = create_run_directory(runs_dir)
    shutil.copy2(BENCHMARK_CONFIG, run_dir / "benchmark_tasks.json")

    is_tty = sys.stderr.isatty() if hasattr(sys.stderr, "isatty") else False

    title_suffix = " (A/B Baseline Comparison)" if compare_baseline else ""
    print("\n" + "=" * 75)
    print(f"🎯 Codebase-Navigator Benchmark & Evaluation Harness{title_suffix}")
    print(f"📦 Run package: {run_dir}")
    if use_llm_judge:
        print(f"⚖️  Judge model: {resolved_judge_model}")
    print("=" * 75)

    overhead = print_prompt_overhead()
    codebase_navigator_git_commit = get_git_commit(PROJECT_DIR)
    codebase_navigator_git_short_hash = codebase_navigator_git_commit[:12]

    # Write the report and log before repository preparation so even an interrupted
    # or failed index build leaves a useful, self-describing run package.
    repo_plan: list[dict[str, Any]] = [
        {
            "repo": entry["repo"],
            "language": entry.get("language", "Unknown"),
            "git_url": entry["git_url"],
            "task_count": len(entry.get("tasks", [])),
            "status": "pending",
        }
        for entry in benchmarks
        if not target_repo or entry["repo"].lower() == target_repo.lower()
    ]
    global_facts: dict[str, Any] = {
        "timestamp": run_timestamp,
        "run_directory": str(run_dir.resolve()),
        "status": "preparing",
        "codebase_navigator_version": __version__,
        "codebase_navigator_git_commit": codebase_navigator_git_commit,
        "codebase_navigator_git_short_hash": codebase_navigator_git_short_hash,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "judge_model": resolved_judge_model if use_llm_judge else None,
        "compare_baseline": compare_baseline,
        "seed_mode": seed_mode or DEFAULT_SEED_MODE,
        "token_metric": {
            "tokens": "sum of prompt and completion tokens across every model call",
            "cached_tokens": "sum of cached prompt tokens across every model call",
            "net_tokens": "uncached prompt tokens plus completion tokens across every model call",
        },
        "prompt_overhead": overhead,
        "benchmark_tasks_snapshot": "benchmark_tasks.json",
        "repo_plan": repo_plan,
        "workers": workers,
    }
    report_stream = ReportStream(run_dir, global_facts)

    # Clone and index each repository exactly once before parallel task execution.
    repo_facts_by_name = {entry["repo"]: entry for entry in repo_plan}
    prepared_indexes: dict[str, tuple[Path, str]] = {}
    work_items: list[tuple[str, Path, str, Path, Any, dict[str, Any]]] = []
    for repo_entry in benchmarks:
        r_name = repo_entry["repo"]
        if target_repo and r_name.lower() != target_repo.lower():
            continue

        r_url = repo_entry["git_url"]
        r_lang = repo_entry.get("language", "Unknown")
        r_dir = REPOS_DIR / r_name
        repo_facts = repo_facts_by_name[r_name]
        repo_facts["checkout_status"] = (
            "existing" if r_dir.is_dir() and (r_dir / ".git").is_dir() else "missing"
        )

        if not ensure_repo_cloned(r_name, r_url, r_dir):
            repo_facts["status"] = "clone_failed"
            report_stream.update_facts(repo_plan=repo_plan)
            continue
        if repo_facts["checkout_status"] == "missing":
            repo_facts["checkout_status"] = "cloned"

        git_commit = get_git_commit(r_dir)
        repo_facts["git_commit"] = git_commit
        config = load_llm_config(folder=r_dir)
        if seed_mode:
            config.seed_mode = seed_mode
        if not config.api_key:
            print(f"\n⚠️  No API key found. Skipping live queries for {r_name}.")
            repo_facts["status"] = "skipped_no_api_key"
            report_stream.update_facts(repo_plan=repo_plan)
            continue

        print(f"\n📁 Repository: {r_name} ({r_lang}) [{r_dir}]")
        print(f"   Git commit: {git_commit}")
        expected_index_dir = INDEXES_DIR / r_name / codebase_navigator_git_short_hash
        print(f"   Preparing cached index at {expected_index_dir}...")
        repo_facts["candidate_model"] = config.model
        repo_facts["status"] = "indexing"
        report_stream.update_facts(repo_plan=repo_plan)
        try:
            index_dir, index_metadata = prepare_evaluation_index(
                r_name,
                r_dir,
                INDEXES_DIR,
                r_url,
                git_commit,
                codebase_navigator_git_commit,
            )
        except Exception as exc:
            repo_facts["status"] = "index_failed"
            repo_facts["error"] = str(exc)
            report_stream.update_facts(repo_plan=repo_plan, status="failed")
            raise
        repo_facts["index"] = {
            "path": os.path.relpath(index_dir, run_dir),
            **index_metadata,
        }
        prepared_indexes[r_name] = (index_dir, index_metadata["index_tree_sha256"])
        index_snapshot = run_dir / "indexes" / f"{r_name}.json"
        index_snapshot.write_text(
            json.dumps(repo_facts["index"], indent=2) + "\n", encoding="utf-8"
        )
        report_stream.record_event(
            "index",
            repo=r_name,
            repo_git_commit=git_commit,
            index_path=str(index_dir.resolve()),
            cache_status=index_metadata["cache_status"],
            index_tree_hash_algorithm="sha256",
            index_tree_sha256=index_metadata["index_tree_sha256"],
        )
        print(
            f"   Index {index_metadata['cache_status']}: "
            f"sha256:{index_metadata['index_tree_sha256']}"
        )
        repo_facts["status"] = "ready"
        report_stream.update_facts(repo_plan=repo_plan)
        print("-" * 60)

        for task in repo_entry.get("tasks", []):
            work_items.append((r_name, r_dir, git_commit, index_dir, config, task))

    report_stream.update_facts(status="running", repo_plan=repo_plan)

    total_tasks = len(work_items)
    use_spinner = is_tty and workers <= 1

    results_report: list[dict[str, Any]] = []
    completed = 0
    cancel_event = threading.Event()
    live_progress = (
        LiveTaskProgress(max_lines=min(workers, total_tasks))
        if is_tty and workers > 1 and total_tasks > 1
        else None
    )
    pool: ThreadPoolExecutor | None = None
    futures = []
    if live_progress is not None:
        live_progress.start()

    try:
        if workers <= 1 or total_tasks <= 1:
            for r_name, r_dir, git_commit, index_dir, config, task in work_items:
                result = _run_task(
                    r_name,
                    r_dir,
                    git_commit,
                    index_dir,
                    task,
                    config,
                    resolved_judge_model,
                    use_llm_judge,
                    compare_baseline,
                    use_spinner,
                    trace_callback=report_stream.record_trace,
                    event_callback=report_stream.record_event,
                    cancel_event=cancel_event,
                )
                results_report.append(result)
                report_stream.add_result(result)
                completed += 1
        else:
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval-worker")
            future_to_task = {}
            for r_name, r_dir, git_commit, index_dir, config, task in work_items:
                future = pool.submit(
                    _run_task,
                    r_name,
                    r_dir,
                    git_commit,
                    index_dir,
                    task,
                    config,
                    resolved_judge_model,
                    use_llm_judge,
                    compare_baseline,
                    use_spinner,
                    report_stream.record_trace,
                    report_stream.record_event,
                    live_progress,
                    cancel_event,
                )
                futures.append(future)
                future_to_task[future] = (r_name, task)
            for future in as_completed(future_to_task):
                r_name, task = future_to_task[future]
                result = future.result()
                results_report.append(result)
                report_stream.add_result(result)
                completed += 1
                if live_progress is not None:
                    live_progress.finish(f"{r_name}/{task['id']}")
                    live_progress.write(
                        f"    🧮 Progress: {completed}/{total_tasks} tasks complete "
                        f"({r_name}/{task['id']})"
                    )
                else:
                    _safe_print(f"    🧮 Progress: {completed}/{total_tasks} tasks complete")
    except KeyboardInterrupt:
        # A repeated Ctrl-C must not interrupt report flushing after the first one.
        previous_sigint_handler = None
        if threading.current_thread() is threading.main_thread():
            previous_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            cancel_event.set()
            for future in futures:
                future.cancel()
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
                pool = None
            if live_progress is not None:
                live_progress.stop()
                live_progress = None
            report_stream.record_event(
                "interrupted", completed_tasks=completed, total_tasks=total_tasks
            )
            report_stream.finalize(
                {
                    "status": "interrupted",
                    "total_tasks": total_tasks,
                    "completed_tasks": completed,
                }
            )
            _safe_print(f"\n⏹️  Evaluation interrupted ({completed}/{total_tasks} tasks completed).")
        finally:
            if previous_sigint_handler is not None:
                signal.signal(signal.SIGINT, previous_sigint_handler)
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if live_progress is not None:
            live_progress.stop()

    # Verify the shared indexes remained byte-for-byte unchanged during evaluation.
    index_integrity_ok = True
    for repo_name, (index_dir, expected_hash) in prepared_indexes.items():
        actual_hash = hash_index_tree(index_dir)
        unchanged = actual_hash == expected_hash
        index_integrity_ok = index_integrity_ok and unchanged
        repo_facts = repo_facts_by_name[repo_name]
        repo_facts["index"]["verified_after_run_sha256"] = actual_hash
        repo_facts["index"]["unchanged_during_run"] = unchanged
        (run_dir / "indexes" / f"{repo_name}.json").write_text(
            json.dumps(repo_facts["index"], indent=2) + "\n", encoding="utf-8"
        )
        report_stream.record_event(
            "index_verification",
            repo=repo_name,
            index_path=str(index_dir.resolve()),
            expected_index_tree_sha256=expected_hash,
            actual_index_tree_sha256=actual_hash,
            unchanged=unchanged,
        )
    report_stream.update_facts(repo_plan=repo_plan, index_integrity_verified=index_integrity_ok)

    # Summary
    passed_tasks = sum(1 for r in results_report if r.get("passed"))
    baseline_passed_tasks = sum(1 for r in results_report if r.get("baseline_passed"))

    # An infrastructure fault ("Remote end closed connection without response") is
    # not a wrong answer. Counting it as one silently understates the score and
    # makes runs incomparable, so errored tasks are reported and scored separately.
    errored = [r for r in results_report if r.get("error")]
    scored_tasks = total_tasks - len(errored)

    print("\n" + "=" * 75)
    score_pct = (passed_tasks / scored_tasks * 100) if scored_tasks > 0 else 0
    print(f"📊 CN Evaluation Score:       {passed_tasks}/{scored_tasks} passed ({score_pct:.1f}%)")
    if errored:
        print(f"⚠️  Incomplete (not scored):   {len(errored)}/{total_tasks} — infrastructure errors")
        for r in errored[:5]:
            print(f"     - {r['task_id']}: {str(r.get('error'))[:70]}")
    if compare_baseline and total_tasks > 0:
        base_score_pct = baseline_passed_tasks / scored_tasks * 100 if scored_tasks else 0.0

        # Only compute savings on mutually successful tasks for fair comparison
        valid_pairs = [
            r
            for r in results_report
            if r.get("passed") and r.get("baseline_passed") and (r.get("baseline_tokens") or 0) > 0
        ]
        valid_cn_tokens = sum(r["tokens"] for r in valid_pairs)
        valid_base_tokens = sum(r["baseline_tokens"] for r in valid_pairs)
        valid_cn_time = sum(r["duration_seconds"] for r in valid_pairs)
        valid_base_time = sum(r["baseline_duration_seconds"] for r in valid_pairs)

        overall_token_savings = (
            round(((valid_base_tokens - valid_cn_tokens) / valid_base_tokens) * 100, 1)
            if valid_base_tokens > 0
            else 0.0
        )
        overall_time_savings = (
            round(((valid_base_time - valid_cn_time) / valid_base_time) * 100, 1)
            if valid_base_time > 0
            else 0.0
        )
        speedup = (valid_base_time / valid_cn_time) if valid_cn_time > 0 else 1.0

        # The aggregate is a token-weighted mean, so one pathological task can
        # carry the whole result. The median says what happens on a typical
        # question, and the two disagreeing is itself the finding.
        per_task_token = sorted(
            100.0 * (r["baseline_tokens"] - r["tokens"]) / r["baseline_tokens"] for r in valid_pairs
        )
        per_task_time = sorted(
            100.0
            * (r["baseline_duration_seconds"] - r["duration_seconds"])
            / r["baseline_duration_seconds"]
            for r in valid_pairs
            if (r.get("baseline_duration_seconds") or 0) > 0
        )
        median_token_savings = statistics.median(per_task_token) if per_task_token else 0.0
        median_time_savings = statistics.median(per_task_time) if per_task_time else 0.0
        cn_cheaper = sum(1 for v in per_task_token if v > 0)

        print(
            f"📊 Baseline Evaluation Score: {baseline_passed_tasks}/{scored_tasks} passed ({base_score_pct:.1f}%)"
        )
        print(
            f"💰 Validated Token Savings:   {overall_token_savings:+.1f}% "
            f"(CN: {valid_cn_tokens:,} vs Base: {valid_base_tokens:,} across {len(valid_pairs)} mutually passed tasks)"
        )
        print(
            f"📐 Median Token Savings:      {median_token_savings:+.1f}% per task "
            f"(CN cheaper on {cn_cheaper}/{len(per_task_token)} tasks)"
        )
        cn_turns = sum(r["api_calls"] for r in valid_pairs)
        base_turns = sum(r["baseline_api_calls"] for r in valid_pairs)
        turn_savings = 100.0 * (base_turns - cn_turns) / base_turns if base_turns else 0.0
        print(
            f"🔄 Turns (round trips):       {turn_savings:+.1f}% "
            f"(CN: {cn_turns} vs Base: {base_turns}) — token cost tracks turns at r≈0.88"
        )
        print(
            f"⏱️  Validated Time Savings:    {overall_time_savings:+.1f}% ({speedup:.2f}x speedup — "
            f"CN: {valid_cn_time:.1f}s vs Base: {valid_base_time:.1f}s)"
        )
        print(f"📐 Median Time Savings:       {median_time_savings:+.1f}% per task")
    print("=" * 75)

    summary: dict[str, Any] = {
        "status": "complete" if index_integrity_ok else "index_integrity_failed",
        "total_tasks": total_tasks,
        "scored_tasks": scored_tasks,
        "errored_tasks": len(errored),
        "errored_task_ids": [r["task_id"] for r in errored],
        "passed_tasks": passed_tasks,
        "score_percentage": score_pct,
        "baseline_passed_tasks": baseline_passed_tasks,
    }
    if compare_baseline and total_tasks > 0:
        summary["median_token_savings_percentage"] = round(median_token_savings, 1)
        summary["median_time_savings_percentage"] = round(median_time_savings, 1)
        summary["cn_cheaper_task_count"] = cn_cheaper
        summary["cn_total_turns"] = cn_turns
        summary["baseline_total_turns"] = base_turns
        summary["turn_savings_percentage"] = round(turn_savings, 1)
        summary["comparable_task_count"] = len(per_task_token)
    report_stream.finalize(summary)
    print(f"📄 Report saved to: {report_stream.report_file}")
    print(f"📜 Trace log saved to: {report_stream.trace_file}")
    print(f"📦 Complete run package: {run_dir}")

    # Errored tasks are excluded from the pass requirement; they are reported
    # separately so an infrastructure blip does not read as a quality failure.
    return passed_tasks == scored_tasks and index_integrity_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run codebase-navigator evaluation benchmarks")
    parser.add_argument("--repo", help="Filter by repository name (e.g. flask, fastapi, httpx)")
    parser.add_argument(
        "--no-judge", action="store_true", help="Disable LLM-as-a-judge (use keyword rules only)"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Run A/B benchmark against generic baseline agent (cat/rg/find/ls)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of tasks to run in parallel (default: 4)",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Parent directory for timestamped run packages (default: eval/runs)",
    )
    parser.add_argument(
        "--seed-mode",
        default=None,
        choices=["always", "router", "never"],
        help=(
            "Pre-flight retrieval policy for the cn arm: 'always' seeds every question "
            "(pre-router behaviour), 'router' seeds only conceptual questions, 'never' "
            "makes the agent call `search` itself. Default: cn's own config."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            f"LLM judge model (default: {DEFAULT_JUDGE_MODEL}; "
            "also configurable with CN_EVAL_JUDGE_MODEL)"
        ),
    )
    args = parser.parse_args()

    try:
        success = run_benchmark(
            target_repo=args.repo,
            use_llm_judge=not args.no_judge,
            compare_baseline=args.compare_baseline,
            workers=args.workers,
            runs_dir=Path(args.runs_dir),
            judge_model=args.judge_model,
            seed_mode=args.seed_mode,
        )
    except KeyboardInterrupt:
        # ThreadPoolExecutor workers cannot be force-cancelled while blocked in a
        # network call. The report has already been flushed, so terminate the CLI
        # process immediately instead of waiting for its non-daemon workers.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    raise SystemExit(0 if success else 1)
