#!/usr/bin/env python3
"""Render a codebase-navigator benchmark report JSON as TTY-like human-readable output.

Usage:
    uv run eval/report_viewer.py eval/reports/report_YYYYMMDD_HHMMSS.json
    uv run eval/report_viewer.py --watch eval/reports/report_YYYYMMDD_HHMMSS.json

The report is streamed by ``eval/runner.py`` as tasks complete, so ``--watch``
re-renders the file in place as new results are appended.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

WIDTH = 75


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


def render_summary(report: dict[str, Any]) -> list[str]:
    total = report.get("total_tasks", 0)
    passed = report.get("passed_tasks", 0)
    score = report.get("score_percentage", 0) or 0
    compare_baseline = report.get("compare_baseline", False)

    lines = ["\n" + "=" * WIDTH]
    lines.append(f"📊 CN Evaluation Score:       {passed}/{total} passed ({score:.1f}%)")

    if compare_baseline and total > 0:
        base_passed = report.get("baseline_passed_tasks", 0)
        base_pct = base_passed / total * 100
        lines.append(
            f"📊 Baseline Evaluation Score: {base_passed}/{total} passed ({base_pct:.1f}%)"
        )
    lines.append("=" * WIDTH)
    return lines


def render(report: dict[str, Any]) -> str:
    compare_baseline = report.get("compare_baseline", False)
    suffix = " (A/B Baseline Comparison)" if compare_baseline else ""

    out: list[str] = [_banner(suffix)]
    out += render_overhead(report.get("prompt_overhead"))

    results = report.get("results", [])
    repo_plan = {rp["repo"]: rp for rp in report.get("repo_plan", [])}

    last_repo: str | None = None
    for r in results:
        repo = r.get("repo", "?")
        if repo != last_repo:
            lang = repo_plan.get(repo, {}).get("language", "Unknown")
            out.append(f"\n📁 Repository: {repo} ({lang})")
            out.append("-" * 60)
            last_repo = repo
        out += render_result(r, compare_baseline)

    out += render_summary(report)
    return "\n".join(out)


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
    args = parser.parse_args()

    path = Path(args.report)
    if not path.is_file():
        print(f"❌ Report not found: {path}", file=sys.stderr)
        sys.exit(1)

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
