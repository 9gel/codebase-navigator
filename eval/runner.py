#!/usr/bin/env python3
"""Automated Multi-Language Benchmark & LLM-as-Judge Evaluation Harness for codebase-navigator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

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


def run_benchmark(
    target_repo: str | None = None,
    use_llm_judge: bool = True,
    save_report: Path | None = None,
):
    """Run full evaluation suite across configured repositories."""
    if not BENCHMARK_CONFIG.is_file():
        print(f"❌ Configuration not found: {BENCHMARK_CONFIG}")
        sys.exit(1)

    with open(BENCHMARK_CONFIG, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print("\n" + "=" * 75)
    print("🎯 Codebase-Navigator Benchmark & Multi-Language Evaluation Harness")
    print("=" * 75)

    results_report: list[dict[str, Any]] = []
    total_tasks = 0
    passed_tasks = 0

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

            t0 = time.time()
            try:
                answer, stats = ask_codebase(
                    folder=r_dir,
                    question=question,
                    config=config,
                    verbose=False,
                    new_session=True,  # Fresh session for clean benchmark
                )
                dt = time.time() - t0

                # 1. Rule-based keyword/file verification
                kw_matches = [kw for kw in req_kws if kw.lower() in answer.lower()]
                file_matches = [f for f in req_files if f.lower() in answer.lower()]
                rule_pass = (len(kw_matches) >= min(1, len(req_kws))) and (
                    len(file_matches) >= min(1, len(req_files))
                )

                # 2. LLM-as-a-Judge verification (if enabled)
                judge_pass = False
                judge_rationale = ""
                if use_llm_judge and config.api_key:
                    judge_pass, judge_rationale = llm_judge_answer(question, key, answer, config)

                task_passed = judge_pass if use_llm_judge else rule_pass

                if task_passed:
                    passed_tasks += 1
                    status_icon = "✅ PASS"
                else:
                    status_icon = "❌ FAIL"

                tokens = stats.get("turn_total_tokens", 0)
                print(f"    Status: {status_icon} (took {dt:.2f}s, tokens: {tokens:,})")
                print(f"    Keyword matches: {kw_matches}/{req_kws}")
                if use_llm_judge:
                    print(f"    Judge Verdict: {'Approved' if judge_pass else 'Rejected'} — {judge_rationale}")

                results_report.append({
                    "task_id": task_id,
                    "repo": r_name,
                    "question": question,
                    "passed": task_passed,
                    "duration_seconds": round(dt, 2),
                    "tokens": tokens,
                    "keyword_matches": kw_matches,
                    "judge_rationale": judge_rationale,
                    "answer_preview": answer[:300] + ("..." if len(answer) > 300 else ""),
                })

            except Exception as e:
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
    print(f"📊 Evaluation Score: {passed_tasks}/{total_tasks} passed ({score_pct:.1f}%)")
    print("=" * 75)

    if save_report:
        with open(save_report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_tasks": total_tasks,
                    "passed_tasks": passed_tasks,
                    "score_percentage": score_pct,
                    "results": results_report,
                },
                f,
                indent=2,
            )
        print(f"📄 Full benchmark report saved to: {save_report}")

    return passed_tasks == total_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run codebase-navigator evaluation benchmarks")
    parser.add_argument("--repo", help="Filter by repository name (e.g. flask, fastapi, httpx)")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM-as-a-judge (use keyword rules only)")
    parser.add_argument("--report", default="eval/report.json", help="Path to save JSON evaluation report")
    args = parser.parse_args()

    run_benchmark(
        target_repo=args.repo,
        use_llm_judge=not args.no_judge,
        save_report=Path(args.report) if args.report else None,
    )
