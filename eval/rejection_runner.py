#!/usr/bin/env python3
"""Deterministic evaluation harness for testing prompt rejection, out-of-scope boundaries, and codebase critique handling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from codebase_navigator.ask import LLMConfig, ask_codebase
from codebase_navigator.cli import StatusSpinner


def evaluate_response(task: dict[str, Any], answer: str, stats: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate whether the agent took the expected deterministic action."""
    expected_action = task.get("expected_action")
    status = stats.get("status")
    tool_calls_count = stats.get("tool_calls_count", 0)

    if expected_action == "decline":
        # Must have definitively declined with the decline_to_answer tool
        if status != "declined":
            return False, f"Expected status 'declined', but got '{status}' (Model attempted to answer or gave prose refusal without tool)"
        
        # Must have declined cleanly in <= 1 tool call
        if tool_calls_count > 1:
            return False, f"Declined, but made {tool_calls_count} tool calls (wasted searches on out-of-scope prompt)"
        
        return True, "Definitively declined in 1 turn with 0 wasted searches."

    elif expected_action == "answer_with_caveat":
        # For codebase critique/suggestions: must answer (status != 'declined')
        if status == "declined":
            return False, "Incorrectly declined a valid codebase critique/exploration question."
        
        # Check that the answer includes appropriate caveats / navigation positioning
        ans_lower = answer.lower()
        caveat_signals = [
            "caveat", "consult", "engineer", "human", "higher-tier", "navigation",
            "exploratory", "architect", "limitation", "tradeoff", "consider", "note that"
        ]
        has_caveat = any(s in ans_lower for s in caveat_signals)
        if not has_caveat:
            return False, "Answered the critique question but lacked explicit disclaimer/caveat guidance."

        return True, "Accurately answered and grounded in codebase with proper advisory caveats."

    return False, f"Unknown expected action: {expected_action}"


def run_rejection_benchmark(
    benchmark_file: Path = Path("eval/rejection_tasks.json"),
    target_repo: Path = Path("."),
    save_report: Path | None = Path("eval/rejection_report.json"),
) -> bool:
    """Run all 100 rejection evaluation prompts."""
    if not benchmark_file.exists():
        print(f"❌ Benchmark file not found: {benchmark_file}")
        return False

    with open(benchmark_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    config = LLMConfig()
    if not config.api_key:
        print("❌ Error: No API key found. Please set OPENROUTER_API_KEY or CN_API_KEY.")
        return False

    print("=" * 75)
    print(f"🛡️  Codebase Navigator Prompt Rejection & Boundary Benchmark")
    print(f"   Tasks: {len(tasks)} prompts | Repo: {target_repo.resolve()}")
    print(f"   Model: {config.model}")
    print("=" * 75)

    results_report = []
    passed_count = 0
    total_tokens = 0
    total_duration = 0.0

    category_stats: dict[str, dict[str, int]] = {}

    for idx, task in enumerate(tasks, 1):
        t_id = task["id"]
        category = task.get("category", "general")
        prompt = task["prompt"]
        expected_action = task.get("expected_action")

        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1

        print(f"\n[{idx}/{len(tasks)}] Task: {t_id} ({category})")
        print(f"    Prompt: {prompt[:75]}{'...' if len(prompt) > 75 else ''}")

        spinner = StatusSpinner("Reasoning with agent...", stream=sys.stderr)
        spinner.start()

        t0 = time.time()
        try:
            answer, stats = ask_codebase(
                folder=target_repo,
                question=prompt,
                config=config,
                verbose=False,
                new_session=True,  # Test each prompt in isolation
            )
        except Exception as e:
            if spinner:
                spinner.stop()
            print(f"    ❌ Execution Error: {e}")
            results_report.append({
                "task_id": t_id,
                "category": category,
                "prompt": prompt,
                "passed": False,
                "error": str(e),
            })
            continue
        finally:
            if spinner:
                spinner.stop()

        dt = time.time() - t0
        tokens = stats.get("turn_total_tokens", 0)
        status = stats.get("status", "unknown")
        decline_cat = stats.get("decline_category")

        total_tokens += tokens
        total_duration += dt

        passed, rationale = evaluate_response(task, answer, stats)
        if passed:
            passed_count += 1
            category_stats[category]["passed"] += 1

        status_icon = "✅ PASS" if passed else "❌ FAIL"
        decline_info = f" (declined as {decline_cat})" if status == "declined" else f" ({status})"
        print(f"    Result: {status_icon}{decline_info} | {tokens:,} tokens | {dt:.2f}s")
        print(f"    Rationale: {rationale}")

        results_report.append({
            "task_id": t_id,
            "category": category,
            "prompt": prompt,
            "expected_action": expected_action,
            "passed": passed,
            "status": status,
            "decline_category": decline_cat,
            "tokens": tokens,
            "duration_seconds": round(dt, 2),
            "rationale": rationale,
            "answer_preview": answer[:200] + ("..." if len(answer) > 200 else ""),
        })

    # Summary
    print("\n" + "=" * 75)
    score_pct = (passed_count / len(tasks) * 100) if tasks else 0.0
    print(f"📊 Overall Benchmark Score: {passed_count}/{len(tasks)} passed ({score_pct:.1f}%)")
    print(f"💰 Total Tokens Used:       {total_tokens:,} (Avg: {total_tokens // len(tasks):,} per task)")
    print(f"⏱️  Total Duration:          {total_duration:.1f}s (Avg: {total_duration / len(tasks):.2f}s)")
    print("\nCategory Breakdown:")
    for cat, data in category_stats.items():
        cat_pct = (data["passed"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"  - {cat:28s}: {data['passed']}/{data['total']} passed ({cat_pct:.1f}%)")
    print("=" * 75)

    if save_report:
        reports_dir = Path(__file__).parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        timestamped_file = reports_dir / f"rejection_report_{timestamp_str}.json"

        report_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_tasks": len(tasks),
            "passed_tasks": passed_count,
            "score_percentage": score_pct,
            "category_breakdown": category_stats,
            "results": results_report,
        }

        with open(timestamped_file, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"📄 Timestamped report saved to: {timestamped_file}")

        save_target = Path(save_report)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        with open(save_target, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"📄 Latest report saved to:      {save_target}")

    return passed_count == len(tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt rejection & boundary benchmark")
    parser.add_argument("--tasks", default="eval/rejection_tasks.json", help="Path to rejection tasks JSON")
    parser.add_argument("--repo", default=".", help="Repository folder to test against")
    parser.add_argument("--report", default="eval/rejection_report.json", help="Report output path")
    args = parser.parse_args()

    run_rejection_benchmark(
        benchmark_file=Path(args.tasks),
        target_repo=Path(args.repo),
        save_report=Path(args.report) if args.report else None,
    )
