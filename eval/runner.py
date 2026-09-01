#!/usr/bin/env python3
"""Automated Multi-Language Benchmark & LLM-as-Judge Evaluation Harness for codebase-navigator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Ensure codebase_navigator from src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codebase_navigator.ask import ask_codebase, call_chat_completions, load_llm_config

EXERCISES_DIR = Path(__file__).parent.parent / "exercises"
BENCHMARK_CONFIG = Path(__file__).parent / "benchmark_tasks.json"


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
    except Exception as e:
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
        content = resp["choices"][0]["message"]["content"].strip()
        # Strip code markdown fences if present
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        data = json.loads(content)
        return bool(data.get("is_correct", False)), data.get("rationale", "")
    except Exception as e:
        return False, f"Judge evaluation error: {e}"


BASELINE_TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read source lines from a file in the repository. Provide start_line and end_line whenever possible to avoid loading entire files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file to read (e.g. 'src/app.py').",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed starting line number (optional, default: 1).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed ending line number (optional, default: 200).",
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
            "description": "Run ripgrep / regex search across repository files to locate symbol names, classes, or patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression or literal text to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional subdirectory or file path to restrict the grep to.",
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
            "description": "Find files matching a glob pattern in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '*.py', '**/*router*').",
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
            "description": "List files and subdirectories inside a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: repository root '.').",
                    }
                },
            },
        },
    },
]


def execute_baseline_tool(folder: Path, name: str, args: dict[str, Any]) -> str:
    """Execute standard baseline tool (cat/rg/find/ls) without specialized codebase-navigator index."""
    if name == "read_file":
        rel_p = args.get("path", "").strip()
        target = folder / rel_p
        if not target.is_file():
            return f"Error: File not found: {rel_p}"
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            s_line = max(1, int(args.get("start_line", 1) or 1))
            e_line = int(args.get("end_line", s_line + 199) or (s_line + 199))
            s_idx = s_line - 1
            e_idx = min(len(lines), e_line)
            selected = lines[s_idx:e_idx]
            out_lines = [f"{i}: {line}" for i, line in enumerate(selected, start=s_line)]
            return "\n".join(out_lines) or "[Empty slice]"
        except Exception as e:
            return f"Error reading {rel_p}: {e}"

    elif name == "grep":
        pattern = args.get("pattern", "")
        sub_path = args.get("path")
        cmd = ["rg", "-n", "-i", pattern]
        if sub_path:
            cmd.append(sub_path)
        try:
            res = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, timeout=10.0)
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
                    except Exception:
                        pass
                if len(matches) >= 200:
                    break
            return "\n".join(matches) or "No matches found."
        except Exception as e:
            return f"Error running grep: {e}"

    elif name == "find_files":
        pattern = args.get("pattern", "*")
        try:
            matches = [str(p.relative_to(folder)) for p in folder.glob(pattern) if p.is_file()]
            if not matches:
                matches = [str(p.relative_to(folder)) for p in folder.rglob(pattern) if p.is_file()]
            return "\n".join(matches[:100]) if matches else "No matching files found."
        except Exception as e:
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
        except Exception as e:
            return f"Error listing directory: {e}"

    return f"Unknown tool: {name}"


def run_baseline_agent(
    folder: Path,
    question: str,
    config,
    progress_callback = None,
) -> tuple[str, dict[str, Any]]:
    """Run typical agent harness with standard tools (read_file, grep, find_files, list_dir)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI coding assistant answering questions about a codebase.\n"
                "Guidelines:\n"
                "1. Use `grep` and `find_files` first to locate relevant definitions, functions, and files.\n"
                "2. When reading code with `read_file`, specify `start_line` and `end_line` to read targeted line ranges rather than reading entire large files.\n"
                "3. Once you locate the primary file and mechanism that answers the question, stop calling tools and synthesize your answer directly."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}",
        },
    ]

    searches_remaining = config.max_searches
    turn_completion_tokens = 0
    last_prompt_tokens = 0
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
        last_prompt_tokens = p_tok
        last_cached_tokens = cached_tok
        turn_completion_tokens += c_tok

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
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except (json.JSONDecodeError, TypeError, ValueError):
                    fn_args = {}

                tool_calls_count += 1
                arg_summary = ", ".join(f"{k}={v!r}" for k, v in list(fn_args.items())[:2])
                if progress_callback:
                    progress_callback(f"🔎 [Baseline Tool {tool_calls_count}: {fn_name}] {arg_summary}...")

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
            "turn_prompt_tokens": last_prompt_tokens,
            "turn_completion_tokens": turn_completion_tokens,
            "turn_total_tokens": last_prompt_tokens + turn_completion_tokens,
            "turn_cached_tokens": last_cached_tokens,
            "tool_calls_count": tool_calls_count,
        }
        return content, stats


def run_benchmark(
    target_repo: str | None = None,
    use_llm_judge: bool = True,
    save_report: Path | None = Path("eval/reports"),
    compare_baseline: bool = False,
):
    """Run full evaluation suite across configured repositories with optional baseline comparison."""
    if not BENCHMARK_CONFIG.is_file():
        print(f"❌ Configuration not found: {BENCHMARK_CONFIG}")
        sys.exit(1)

    with open(BENCHMARK_CONFIG, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    from codebase_navigator.cli import StatusSpinner
    is_tty = sys.stderr.isatty() if hasattr(sys.stderr, "isatty") else False

    title_suffix = " (A/B Baseline Comparison)" if compare_baseline else ""
    print("\n" + "=" * 75)
    print(f"🎯 Codebase-Navigator Benchmark & Evaluation Harness{title_suffix}")
    print("=" * 75)

    results_report: list[dict[str, Any]] = []
    total_tasks = 0
    passed_tasks = 0
    baseline_passed_tasks = 0

    total_cn_tokens = 0
    total_base_tokens = 0

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
            task_id = task["id"]
            question = task["question"]
            key = task["expected_answer_key"]
            req_kws = task.get("required_keywords", [])
            req_files = task.get("required_files", [])

            total_tasks += 1
            print(f"\n  ▶ [{task_id}] Q: {question}")

            spinner: StatusSpinner | None = None

            def handle_cn_progress(line: str):
                nonlocal spinner
                if not is_tty:
                    return
                if spinner:
                    spinner.update_message(f"CN: {line}")
                else:
                    spinner = StatusSpinner(f"CN: {line}", stream=sys.stderr)
                    spinner.start()

            # 1. Evaluate Codebase-Navigator Agent
            t0 = time.time()
            try:
                if is_tty:
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
                if spinner:
                    spinner.stop()
                    spinner = None

                cn_dt = time.time() - t0
                cn_kw_matches = [kw for kw in req_kws if kw.lower() in cn_answer.lower()]
                cn_file_matches = [f for f in req_files if f.lower() in cn_answer.lower()]
                cn_rule_pass = (len(cn_kw_matches) >= min(1, len(req_kws))) and (
                    len(cn_file_matches) >= min(1, len(req_files))
                )

                cn_judge_pass = False
                cn_rationale = ""
                if use_llm_judge and config.api_key:
                    if is_tty:
                        spinner = StatusSpinner("CN: ⚖️ Running LLM Judge evaluation...", stream=sys.stderr)
                        spinner.start()
                    cn_judge_pass, cn_rationale = llm_judge_answer(question, key, cn_answer, config)
                    if spinner:
                        spinner.stop()
                        spinner = None

                cn_passed = cn_judge_pass if use_llm_judge else cn_rule_pass
                if cn_passed:
                    passed_tasks += 1

                cn_tokens = cn_stats.get("turn_total_tokens", 0)
                cn_cached = cn_stats.get("turn_cached_tokens", 0)
                total_cn_tokens += cn_tokens
                cn_cached_str = f", cached: {cn_cached:,}" if cn_cached > 0 else ""
                print(f"    [CN]       Status: {'✅ PASS' if cn_passed else '❌ FAIL'} (took {cn_dt:.2f}s, tokens: {cn_tokens:,}{cn_cached_str})")

                # 2. Evaluate Baseline Agent (if --compare-baseline)
                base_tokens = 0
                base_cached = 0
                base_dt = 0.0
                base_passed = False
                base_rationale = ""
                token_savings_pct = 0.0

                if compare_baseline:
                    def handle_base_progress(line: str):
                        nonlocal spinner
                        if not is_tty:
                            return
                        if spinner:
                            spinner.update_message(f"Baseline: {line}")
                        else:
                            spinner = StatusSpinner(f"Baseline: {line}", stream=sys.stderr)
                            spinner.start()

                    t_b0 = time.time()
                    try:
                        if is_tty:
                            spinner = StatusSpinner("Baseline: 🤖 Reasoning with agent...", stream=sys.stderr)
                            spinner.start()

                        base_answer, base_stats = run_baseline_agent(
                            r_dir, question, config, progress_callback=handle_base_progress
                        )
                        if spinner:
                            spinner.stop()
                            spinner = None

                        base_dt = time.time() - t_b0
                        base_kw_matches = [kw for kw in req_kws if kw.lower() in base_answer.lower()]
                        base_file_matches = [f for f in req_files if f.lower() in base_answer.lower()]
                        base_rule_pass = (len(base_kw_matches) >= min(1, len(req_kws))) and (
                            len(base_file_matches) >= min(1, len(req_files))
                        )

                        if use_llm_judge and config.api_key:
                            if is_tty:
                                spinner = StatusSpinner("Baseline: ⚖️ Running LLM Judge evaluation...", stream=sys.stderr)
                                spinner.start()
                            base_judge_pass, base_rationale = llm_judge_answer(question, key, base_answer, config)
                            if spinner:
                                spinner.stop()
                                spinner = None
                        else:
                            base_judge_pass = base_rule_pass

                        base_passed = base_judge_pass
                        if base_passed:
                            baseline_passed_tasks += 1

                        base_tokens = base_stats.get("turn_total_tokens", 0)
                        base_cached = base_stats.get("turn_cached_tokens", 0)
                        total_base_tokens += base_tokens

                        if base_tokens > 0:
                            token_savings_pct = round(((base_tokens - cn_tokens) / base_tokens) * 100, 1)

                        time_savings_pct = round(((base_dt - cn_dt) / base_dt) * 100, 1) if base_dt > 0 else 0.0

                        base_cached_str = f", cached: {base_cached:,}" if base_cached > 0 else ""
                        print(f"    [Baseline] Status: {'✅ PASS' if base_passed else '❌ FAIL'} (took {base_dt:.2f}s, tokens: {base_tokens:,}{base_cached_str})")
                        print(
                            f"    ⚡ Savings: Tokens {token_savings_pct:+.1f}% ({cn_tokens:,} vs {base_tokens:,}) | "
                            f"Time {time_savings_pct:+.1f}% ({cn_dt:.2f}s vs {base_dt:.2f}s)"
                        )
                    except Exception as e_base:
                        if spinner:
                            spinner.stop()
                            spinner = None
                        print(f"    ❌ Baseline Error: {e_base}")

                results_report.append({
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
                })

            except Exception as e:
                if spinner:
                    spinner.stop()
                    spinner = None
                print(f"    ❌ Error executing task: {e}")
                results_report.append({
                    "task_id": task_id,
                    "repo": r_name,
                    "question": question,
                    "passed": False,
                    "error": str(e),
                })

    # Summary
    print("\n" + "=" * 75)
    score_pct = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    print(f"📊 CN Evaluation Score:       {passed_tasks}/{total_tasks} passed ({score_pct:.1f}%)")
    if compare_baseline and total_tasks > 0:
        base_score_pct = (baseline_passed_tasks / total_tasks * 100)

        # Only compute savings on mutually successful tasks for fair comparison
        valid_pairs = [
            r for r in results_report
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

        print(f"📊 Baseline Evaluation Score: {baseline_passed_tasks}/{total_tasks} passed ({base_score_pct:.1f}%)")
        print(
            f"💰 Validated Token Savings:   {overall_token_savings:+.1f}% "
            f"(CN: {valid_cn_tokens:,} vs Base: {valid_base_tokens:,} across {len(valid_pairs)} mutually passed tasks)"
        )
        print(
            f"⏱️  Validated Time Savings:    {overall_time_savings:+.1f}% ({speedup:.2f}x speedup — "
            f"CN: {valid_cn_time:.1f}s vs Base: {valid_base_time:.1f}s)"
        )
    print("=" * 75)

    if save_report:
        save_path = Path(save_report)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        if save_path.is_dir() or not save_path.suffix:
            save_path.mkdir(parents=True, exist_ok=True)
            report_file = save_path / f"report_{timestamp_str}.json"
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            report_file = save_path

        report_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_tasks": total_tasks,
            "passed_tasks": passed_tasks,
            "score_percentage": score_pct,
            "compare_baseline": compare_baseline,
            "results": results_report,
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"📄 Timestamped report saved to: {report_file}")

    return passed_tasks == total_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run codebase-navigator evaluation benchmarks")
    parser.add_argument("--repo", help="Filter by repository name (e.g. flask, fastapi, httpx)")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM-as-a-judge (use keyword rules only)")
    parser.add_argument("--compare-baseline", action="store_true", help="Run A/B benchmark against generic baseline agent (cat/rg/find/ls)")
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
    )

