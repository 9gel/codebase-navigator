#!/usr/bin/env python3
"""Automated Multi-Language Benchmark & LLM-as-Judge Evaluation Harness for codebase-navigator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Ensure codebase_navigator from src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codebase_navigator.ask import (
    AGENT_TOOLS_SPEC,
    SYSTEM_PROMPT,
    ask_codebase,
    call_chat_completions,
    load_llm_config,
)
from codebase_navigator.sandbox_bash import bash_tool_spec, run_sandboxed_bash

EXERCISES_DIR = Path(__file__).parent.parent / "exercises"
BENCHMARK_CONFIG = Path(__file__).parent / "benchmark_tasks.json"

_PRINT_LOCK = threading.Lock()


def _safe_print(*args: Any, **kwargs: Any) -> None:
    """Thread-safe print so parallel task workers don't interleave lines."""
    with _PRINT_LOCK:
        print(*args, **kwargs, flush=True)


def ensure_repo_cloned(repo_name: str, git_url: str, target_dir: Path) -> bool:
    """Ensure target repository is cloned and ready for indexing."""
    if target_dir.is_dir() and (target_dir / ".git").is_dir():
        return True

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
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    try:
        resp = call_chat_completions(config.endpoint, config.api_key, payload, timeout=30.0)
        content = (resp["choices"][0]["message"].get("content") or "").strip()
        # Strip code markdown fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        data = json.loads(content)
        return bool(data.get("is_correct", False)), data.get("rationale", "")
    except (RuntimeError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
        return False, f"Judge evaluation error: {e}"


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
            if len(lines) > 200:
                out = "\n".join(lines[:200]) + f"\n... [{len(lines) - 200} lines truncated]"
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
    output_tokens = 0
    context_tokens = 0
    cached_tokens = 0
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
        context_tokens = p_tok
        cached_tokens = cached_tok
        output_tokens += c_tok

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
        stats = {
            "context_tokens": context_tokens,
            "output_tokens": output_tokens,
            "context_output_tokens": context_tokens + output_tokens,
            "cached_tokens": cached_tokens,
            "tool_calls_count": tool_calls_count,
        }
        return content, stats


class ReportStream:
    """Streams benchmark results to a JSON file as tasks complete.

    The report is written in two phases so it is always valid JSON:
      1. On construction, global facts (timestamp, overhead, repo plan) are written
         with an empty ``results`` list.
      2. Each completed task is appended and the file is atomically rewritten.
    """

    def __init__(self, save_report: Path | None, global_facts: dict[str, Any]):
        self.data: dict[str, Any] = {**global_facts, "results": []}
        self.report_file: Path | None = None
        self._lock = threading.Lock()
        if save_report is not None:
            self.report_file = self._resolve_path(save_report)
            self._write()

    @staticmethod
    def _resolve_path(save_report: Path) -> Path:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        if save_report.is_dir() or not save_report.suffix:
            save_report.mkdir(parents=True, exist_ok=True)
            return save_report / f"report_{timestamp_str}.json"
        save_report.parent.mkdir(parents=True, exist_ok=True)
        return save_report

    def add_result(self, result: dict[str, Any]) -> None:
        with self._lock:
            self.data["results"].append(result)
            self._write()

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
    task: dict[str, Any],
    config,
    use_llm_judge: bool,
    compare_baseline: bool,
    use_spinner: bool,
) -> dict[str, Any]:
    """Execute a single benchmark task (CN agent + judge, then optional baseline) and return its result."""
    from codebase_navigator.cli import StatusSpinner

    task_id = task["id"]
    question = task["question"]
    key = task["expected_answer_key"]
    req_kws = task.get("required_keywords", [])
    req_files = task.get("required_files", [])

    _safe_print(f"\n  ▶ [{task_id}] Q: {question}")

    spinner: StatusSpinner | None = None

    def _stop_spinner() -> None:
        nonlocal spinner
        if spinner:
            spinner.stop()
            spinner = None

    def _progress(label: str, line: str) -> None:
        nonlocal spinner
        if not use_spinner:
            return
        if spinner:
            spinner.update_message(f"{label}: {line}")
        else:
            spinner = StatusSpinner(f"{label}: {line}", stream=sys.stderr)
            spinner.start()

    def handle_cn_progress(line: str) -> None:
        _progress("CN", line)

    def handle_base_progress(line: str) -> None:
        _progress("Baseline", line)

    # 1. Evaluate Codebase-Navigator Agent
    t0 = time.time()
    try:
        if use_spinner:
            spinner = StatusSpinner("CN: 🔍 Searching codebase...", stream=sys.stderr)
            spinner.start()

        cn_answer, cn_stats = ask_codebase(
            folder=r_dir,
            question=question,
            config=config,
            verbose=False,
            new_session=True,
            progress_callback=handle_cn_progress,
        )
        _stop_spinner()

        cn_dt = time.time() - t0
        cn_kw_matches = [kw for kw in req_kws if kw.lower() in cn_answer.lower()]
        cn_file_matches = [f for f in req_files if f.lower() in cn_answer.lower()]
        cn_rule_pass = (len(cn_kw_matches) >= min(1, len(req_kws))) and (
            len(cn_file_matches) >= min(1, len(req_files))
        )

        cn_judge_pass = False
        cn_rationale = ""
        if use_llm_judge and config.api_key:
            cn_judge_pass, cn_rationale = llm_judge_answer(question, key, cn_answer, config)

        cn_passed = cn_judge_pass if use_llm_judge else cn_rule_pass

        cn_tokens = cn_stats.get("context_output_tokens", 0)
        cn_cached = cn_stats.get("cached_tokens", 0)
        cn_cached_str = f", cached: {cn_cached:,}" if cn_cached > 0 else ""
        _safe_print(
            f"    [CN]       Status: {'✅ PASS' if cn_passed else '❌ FAIL'} (took {cn_dt:.2f}s, tokens: {cn_tokens:,}{cn_cached_str})"
        )

        # 2. Evaluate Baseline Agent (if --compare-baseline)
        base_tokens = 0
        base_cached = 0
        base_dt = 0.0
        base_passed = False
        token_savings_pct = 0.0
        time_savings_pct = 0.0

        if compare_baseline:
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

                base_dt = time.time() - t_b0
                base_kw_matches = [kw for kw in req_kws if kw.lower() in base_answer.lower()]
                base_file_matches = [f for f in req_files if f.lower() in base_answer.lower()]
                base_rule_pass = (len(base_kw_matches) >= min(1, len(req_kws))) and (
                    len(base_file_matches) >= min(1, len(req_files))
                )

                if use_llm_judge and config.api_key:
                    base_judge_pass, _ = llm_judge_answer(question, key, base_answer, config)
                else:
                    base_judge_pass = base_rule_pass

                base_passed = base_judge_pass

                base_tokens = base_stats.get("context_output_tokens", 0)
                base_cached = base_stats.get("cached_tokens", 0)

                if base_tokens > 0:
                    token_savings_pct = round(((base_tokens - cn_tokens) / base_tokens) * 100, 1)

                time_savings_pct = (
                    round(((base_dt - cn_dt) / base_dt) * 100, 1) if base_dt > 0 else 0.0
                )

                base_cached_str = f", cached: {base_cached:,}" if base_cached > 0 else ""
                _safe_print(
                    f"    [Baseline] Status: {'✅ PASS' if base_passed else '❌ FAIL'} (took {base_dt:.2f}s, tokens: {base_tokens:,}{base_cached_str})"
                )
                _safe_print(
                    f"    ⚡ Savings: Tokens {token_savings_pct:+.1f}% ({cn_tokens:,} vs {base_tokens:,}) | "
                    f"Time {time_savings_pct:+.1f}% ({cn_dt:.2f}s vs {base_dt:.2f}s)"
                )
            except Exception as e_base:  # noqa: BLE001 — isolate agent failures
                _stop_spinner()
                _safe_print(f"    ❌ Baseline Error: {e_base}")

        return {
            "task_id": task_id,
            "repo": r_name,
            "question": question,
            "passed": cn_passed,
            "duration_seconds": round(cn_dt, 2),
            "tokens": cn_tokens,
            "cached_tokens": cn_cached,
            "judge_rationale": cn_rationale,
            "baseline_passed": base_passed if compare_baseline else None,
            "baseline_duration_seconds": round(base_dt, 2) if compare_baseline else None,
            "baseline_tokens": base_tokens if compare_baseline else None,
            "baseline_cached_tokens": base_cached if compare_baseline else None,
            "token_savings_percentage": token_savings_pct if compare_baseline else None,
            "time_savings_percentage": time_savings_pct if compare_baseline else None,
            "answer_preview": cn_answer[:300] + ("..." if len(cn_answer) > 300 else ""),
        }

    except Exception as e:  # noqa: BLE001 — isolate agent failures
        _stop_spinner()
        _safe_print(f"    ❌ Error executing task: {e}")
        return {
            "task_id": task_id,
            "repo": r_name,
            "question": question,
            "passed": False,
            "error": str(e),
        }


def run_benchmark(
    target_repo: str | None = None,
    use_llm_judge: bool = True,
    save_report: Path | None = Path("eval/reports"),
    compare_baseline: bool = False,
    workers: int = 4,
):
    """Run full evaluation suite across configured repositories with optional baseline comparison."""
    if not BENCHMARK_CONFIG.is_file():
        print(f"❌ Configuration not found: {BENCHMARK_CONFIG}")
        sys.exit(1)

    with open(BENCHMARK_CONFIG, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    is_tty = sys.stderr.isatty() if hasattr(sys.stderr, "isatty") else False

    title_suffix = " (A/B Baseline Comparison)" if compare_baseline else ""
    print("\n" + "=" * 75)
    print(f"🎯 Codebase-Navigator Benchmark & Evaluation Harness{title_suffix}")
    print("=" * 75)

    overhead = print_prompt_overhead()

    repo_plan = [
        {
            "repo": r["repo"],
            "language": r.get("language", "Unknown"),
            "git_url": r["git_url"],
            "task_count": len(r.get("tasks", [])),
        }
        for r in benchmarks
        if not target_repo or r["repo"].lower() == target_repo.lower()
    ]

    global_facts: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compare_baseline": compare_baseline,
        "prompt_overhead": overhead,
        "repo_plan": repo_plan,
        "workers": workers,
    }
    report_stream = ReportStream(save_report, global_facts)

    # Build the flat work list, cloning repos and resolving configs up front.
    work_items: list[tuple[str, Path, Any, dict[str, Any]]] = []
    for repo_entry in benchmarks:
        r_name = repo_entry["repo"]
        if target_repo and r_name.lower() != target_repo.lower():
            continue

        r_url = repo_entry["git_url"]
        r_lang = repo_entry.get("language", "Unknown")
        r_dir = EXERCISES_DIR / r_name

        if not ensure_repo_cloned(r_name, r_url, r_dir):
            continue

        config = load_llm_config(folder=r_dir)
        if not config.api_key:
            print(f"\n⚠️  No API key found. Skipping live queries for {r_name}.")
            continue

        print(f"\n📁 Repository: {r_name} ({r_lang}) [{r_dir}]")
        print("-" * 60)

        for task in repo_entry.get("tasks", []):
            work_items.append((r_name, r_dir, config, task))

    total_tasks = len(work_items)
    use_spinner = is_tty and workers <= 1

    results_report: list[dict[str, Any]] = []
    completed = 0

    if workers <= 1 or total_tasks <= 1:
        for r_name, r_dir, config, task in work_items:
            result = _run_task(
                r_name, r_dir, task, config, use_llm_judge, compare_baseline, use_spinner
            )
            results_report.append(result)
            report_stream.add_result(result)
            completed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_task = {
                pool.submit(
                    _run_task,
                    r_name,
                    r_dir,
                    task,
                    config,
                    use_llm_judge,
                    compare_baseline,
                    use_spinner,
                ): (r_name, task)
                for r_name, r_dir, config, task in work_items
            }
            for future in as_completed(future_to_task):
                result = future.result()
                results_report.append(result)
                report_stream.add_result(result)
                completed += 1
                _safe_print(f"    🧮 Progress: {completed}/{total_tasks} tasks complete")

    # Summary
    passed_tasks = sum(1 for r in results_report if r.get("passed"))
    baseline_passed_tasks = sum(1 for r in results_report if r.get("baseline_passed"))

    print("\n" + "=" * 75)
    score_pct = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    print(f"📊 CN Evaluation Score:       {passed_tasks}/{total_tasks} passed ({score_pct:.1f}%)")
    if compare_baseline and total_tasks > 0:
        base_score_pct = baseline_passed_tasks / total_tasks * 100

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

        print(
            f"📊 Baseline Evaluation Score: {baseline_passed_tasks}/{total_tasks} passed ({base_score_pct:.1f}%)"
        )
        print(
            f"💰 Validated Token Savings:   {overall_token_savings:+.1f}% "
            f"(CN: {valid_cn_tokens:,} vs Base: {valid_base_tokens:,} across {len(valid_pairs)} mutually passed tasks)"
        )
        print(
            f"⏱️  Validated Time Savings:    {overall_time_savings:+.1f}% ({speedup:.2f}x speedup — "
            f"CN: {valid_cn_time:.1f}s vs Base: {valid_base_time:.1f}s)"
        )
    print("=" * 75)

    summary: dict[str, Any] = {
        "total_tasks": total_tasks,
        "passed_tasks": passed_tasks,
        "score_percentage": score_pct,
        "baseline_passed_tasks": baseline_passed_tasks,
    }
    report_stream.finalize(summary)
    if report_stream.report_file is not None:
        print(f"📄 Timestamped report saved to: {report_stream.report_file}")

    return passed_tasks == total_tasks


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
        "--report",
        default="eval/reports",
        help="Directory or file path to save report (default: eval/reports/report_<timestamp>.json)",
    )
    args = parser.parse_args()

    run_benchmark(
        target_repo=args.repo,
        use_llm_judge=not args.no_judge,
        save_report=Path(args.report) if args.report else None,
        compare_baseline=args.compare_baseline,
        workers=args.workers,
    )
