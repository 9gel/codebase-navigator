#!/usr/bin/env python3
"""Render a codebase-navigator benchmark report JSON as TTY-like human-readable output.

Usage:
    uv run eval/report_viewer.py eval/runs/run_YYYYMMDD_HHMMSSffffff/report.json
    uv run eval/report_viewer.py --watch eval/runs/run_YYYYMMDD_HHMMSSffffff/report.json

The report is streamed by ``eval/runner.py`` as tasks complete, so ``--watch``
re-renders the file in place as new results are appended.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

WIDTH = 75

COL_GAP = "  │  "


def _wrap(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=width, replace_whitespace=False) or [""])
    return lines


def render_diff(left_title: str, left: str, right_title: str, right: str) -> str:
    """Render two texts side by side, line-aligned, for A/B comparison."""
    col_w = (WIDTH - len(COL_GAP)) // 2
    left_lines = _wrap(left, col_w)
    right_lines = _wrap(right, col_w)
    n = max(len(left_lines), len(right_lines), 1)

    out = [f"  {left_title:<{col_w}}{COL_GAP}{right_title}"]
    out.append("  " + "-" * (WIDTH - 2))
    for i in range(n):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        out.append(f"  {l:<{col_w}}{COL_GAP}{r}")
    out.append("  " + "-" * (WIDTH - 2))
    return "\n".join(out)


def _banner(title: str) -> str:
    return (
        "\n"
        + "=" * WIDTH
        + f"\n🎯 Codebase-Navigator Benchmark & Evaluation Harness{title}\n"
        + "=" * WIDTH
    )


def render_overhead(oh: dict[str, int] | None) -> list[str]:
    if not oh:
        return []
    return [
        "=" * WIDTH,
        "📏 Prompt overhead (est. tokens, ~4 chars/token)",
        "-" * WIDTH,
        f"  CN system prompt:      {oh.get('cn_system_prompt_tokens', 0):>6,} tokens",
        f"  Baseline system prompt:{oh.get('baseline_system_prompt_tokens', 0):>6,} tokens",
        f"  CN tools spec:         {oh.get('cn_tools_spec_tokens', 0):>6,} tokens",
        f"  Baseline tools spec:   {oh.get('baseline_tools_spec_tokens', 0):>6,} tokens",
        f"  CN per-turn overhead:  {oh.get('cn_per_turn_overhead_tokens', 0):>6,} tokens",
        f"  Base per-turn overhead:{oh.get('baseline_per_turn_overhead_tokens', 0):>6,} tokens",
        "=" * WIDTH,
    ]


def render_result(r: dict[str, Any], compare_baseline: bool) -> list[str]:
    lines: list[str] = []
    lines.append(f"  ▶ [{r.get('task_id', '?')}] Q: {r.get('question', '')}")

    if "error" in r and not r.get("passed"):
        lines.append(f"    ❌ Error executing task: {r.get('error')}")
        return lines

    cn_status = "✅ PASS" if r.get("passed") else "❌ FAIL"
    cn_cached = r.get("cached_tokens") or 0
    cn_cached_str = f", cached: {cn_cached:,}" if cn_cached else ""
    lines.append(
        f"    [CN]       Status: {cn_status} (took {r.get('duration_seconds', 0):.2f}s, "
        f"tokens: {r.get('tokens', 0):,}{cn_cached_str})"
    )

    if compare_baseline and r.get("baseline_passed") is not None:
        base_status = "✅ PASS" if r.get("baseline_passed") else "❌ FAIL"
        base_cached = r.get("baseline_cached_tokens") or 0
        base_cached_str = f", cached: {base_cached:,}" if base_cached else ""
        lines.append(
            f"    [Baseline] Status: {base_status} (took {r.get('baseline_duration_seconds', 0):.2f}s, "
            f"tokens: {r.get('baseline_tokens', 0):,}{base_cached_str})"
        )
        lines.append(
            f"    ⚡ Savings: Tokens {r.get('token_savings_percentage', 0):+.1f}% "
            f"({r.get('tokens', 0):,} vs {r.get('baseline_tokens', 0):,}) | "
            f"Time {r.get('time_savings_percentage', 0):+.1f}% "
            f"({r.get('duration_seconds', 0):.2f}s vs {r.get('baseline_duration_seconds', 0):.2f}s)"
        )
    return lines


def _expected_total(report: dict[str, Any]) -> int:
    """Number of tasks the benchmark expects, used to flag aborted/partial runs."""
    if report.get("total_tasks"):
        return report["total_tasks"]
    plan = report.get("repo_plan") or []
    planned = sum(rp.get("task_count", 0) for rp in plan)
    return planned or len(report.get("results", []))


def render_summary(report: dict[str, Any]) -> list[str]:
    results: list[dict[str, Any]] = report.get("results", [])
    completed = len(results)
    expected = _expected_total(report)
    passed = sum(1 for r in results if r.get("passed"))
    base_passed = sum(1 for r in results if r.get("baseline_passed"))
    compare_baseline = report.get("compare_baseline", False)

    aborted = "total_tasks" not in report or completed < expected

    lines = ["\n" + "=" * WIDTH]
    status = f" (aborted/incomplete: {completed}/{expected})" if aborted else ""
    denom = expected if expected else 1
    score = passed / denom * 100
    lines.append(f"📊 CN Evaluation Score:       {passed}/{denom} passed ({score:.1f}%){status}")

    if compare_baseline and completed > 0:
        base_pct = base_passed / denom * 100
        lines.append(
            f"📊 Baseline Evaluation Score: {base_passed}/{denom} passed ({base_pct:.1f}%)"
        )

        # Overall savings over mutually-passed tasks (matches eval/runner.py).
        valid_pairs = [
            r
            for r in results
            if r.get("passed") and r.get("baseline_passed") and (r.get("baseline_tokens") or 0) > 0
        ]
        cn_tokens = sum(r.get("tokens", 0) for r in valid_pairs)
        base_tokens = sum(r.get("baseline_tokens", 0) for r in valid_pairs)
        cn_time = sum(r.get("duration_seconds", 0) for r in valid_pairs)
        base_time = sum(r.get("baseline_duration_seconds", 0) for r in valid_pairs)

        token_savings = (
            round((base_tokens - cn_tokens) / base_tokens * 100, 1) if base_tokens > 0 else 0.0
        )
        time_savings = round((base_time - cn_time) / base_time * 100, 1) if base_time > 0 else 0.0
        speedup = (base_time / cn_time) if cn_time > 0 else 1.0

        # How many tasks regressed vs baseline (CN used more tokens / took longer).
        compared = [
            r
            for r in results
            if r.get("baseline_tokens") is not None and r.get("tokens") is not None
        ]
        token_worse = sum(
            1 for r in compared if (r.get("tokens") or 0) > (r.get("baseline_tokens") or 0)
        )
        timed = [
            r
            for r in results
            if r.get("baseline_duration_seconds") is not None
            and r.get("duration_seconds") is not None
        ]
        time_worse = sum(
            1
            for r in timed
            if (r.get("duration_seconds") or 0) > (r.get("baseline_duration_seconds") or 0)
        )

        prefix = "~" if aborted else ""
        lines.append(
            f"💰 Validated Token Savings:   {prefix}{token_savings:+.1f}% "
            f"(CN: {cn_tokens:,} vs Base: {base_tokens:,} across {len(valid_pairs)} mutually passed tasks)"
        )
        lines.append(
            f"⏱️  Validated Time Savings:    {prefix}{time_savings:+.1f}% ({speedup:.2f}x speedup — "
            f"CN: {cn_time:.1f}s vs Base: {base_time:.1f}s)"
        )
        lines.append(
            f"📉 Token regressions:         {token_worse}/{len(compared)} tasks worse "
            f"(CN used more tokens than baseline)"
        )
        lines.append(
            f"🕒 Time regressions:          {time_worse}/{len(timed)} tasks worse "
            f"(CN slower than baseline)"
        )
    lines.append("=" * WIDTH)
    return lines


def render(report: dict[str, Any]) -> str:
    compare_baseline = report.get("compare_baseline", False)
    suffix = " (A/B Baseline Comparison)" if compare_baseline else ""

    out: list[str] = [_banner(suffix)]
    if report.get("judge_model"):
        out.append(f"⚖️  Judge model: {report['judge_model']}")
    if report.get("embedding_model"):
        out.append(f"🧠 Embedding model: {report['embedding_model']}")
    out += render_overhead(report.get("prompt_overhead"))

    results = report.get("results", [])
    repo_plan = {rp["repo"]: rp for rp in report.get("repo_plan", [])}

    last_repo: str | None = None
    for r in results:
        repo = r.get("repo", "?")
        if repo != last_repo:
            plan = repo_plan.get(repo, {})
            lang = plan.get("language", "Unknown")
            commit = plan.get("git_commit", "unknown")
            out.append(f"\n📁 Repository: {repo} ({lang}) @ {commit}")
            out.append("-" * 60)
            last_repo = repo
        out += render_result(r, compare_baseline)

    out += render_summary(report)
    return "\n".join(out)


def load_trace(path: Path) -> dict[str, dict[str, Any]]:
    """Load a JSONL trace log, keyed by task_id. Returns {} if missing/invalid."""
    entries: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return entries
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "task":
                continue
            tid = obj.get("task_id")
            if tid is not None:
                entries[str(tid)] = obj
    except (json.JSONDecodeError, OSError):
        return {}
    return entries


def render_diff_view(trace_entries: dict[str, dict[str, Any]], mismatch_only: bool) -> str:
    """Render full CN-vs-baseline answers side by side for each traced task."""
    out: list[str] = []
    for tid, t in trace_entries.items():
        cn = t.get("cn") or {}
        base = t.get("baseline")
        if mismatch_only:
            cn_pass = bool(cn.get("passed"))
            base_pass = bool(base.get("passed")) if base is not None else True
            if cn_pass and base_pass:
                continue
        out.append(f"\n  ▶ [{tid}] Q: {t.get('question', '')}")
        if cn.get("error"):
            out.append(f"    ❌ CN error: {cn.get('error')}")
        out.append(
            render_diff(
                f"CN {'✅' if cn.get('passed') else '❌'}",
                cn.get("answer", "(no answer)"),
                "Baseline " + ("✅" if base and base.get("passed") else ("❌" if base else "—")),
                base.get("answer", "(no answer)") if base is not None else "(no baseline)",
            )
        )
    return "\n".join(out) if out else "\nNo traced tasks found.\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to report JSON file")
    parser.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help="Poll the report file and re-render as new results are appended",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Watch poll interval (s)")
    parser.add_argument(
        "--trace",
        default=None,
        help="Path to a trace log (*.jsonl) to render full answers side by side",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Render full CN-vs-baseline answers side by side (requires --trace)",
    )
    parser.add_argument(
        "--mismatch-only",
        action="store_true",
        help="With --diff, only show tasks where CN or baseline did not pass",
    )
    args = parser.parse_args()

    path = Path(args.report)
    if not path.is_file():
        print(f"❌ Report not found: {path}", file=sys.stderr)
        sys.exit(1)

    if args.diff:
        trace_path = Path(args.trace) if args.trace else None
        if trace_path is None:
            print("❌ --diff requires --trace <trace.jsonl>", file=sys.stderr)
            sys.exit(1)
        entries = load_trace(trace_path)
        print(render_diff_view(entries, args.mismatch_only))
        return

    if not args.watch:
        report = json.loads(path.read_text(encoding="utf-8"))
        print(render(report))
        return

    last_size = -1
    last_mtime = -1.0
    while True:
        try:
            stat = path.stat()
            if stat.st_size != last_size or stat.st_mtime != last_mtime:
                report = json.loads(path.read_text(encoding="utf-8"))
                last_size = stat.st_size
                last_mtime = stat.st_mtime
                os.system("clear" if os.name == "posix" else "cls")
                print(render(report))
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
