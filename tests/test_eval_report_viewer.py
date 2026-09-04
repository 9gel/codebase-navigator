"""Tests for rendering self-contained evaluation run packages."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.report_viewer import render, resolve_report_path, resolve_trace_path


def test_render_shows_run_revision_index_and_cumulative_tokens():
    commit = "a" * 40
    report = {
        "timestamp": "2026-09-04T12:00:00Z",
        "status": "complete",
        "codebase_navigator_version": "0.3.60",
        "judge_model": "deepseek/deepseek-v4-pro",
        "embedding_model": "test-embedding-model",
        "token_metric": {"tokens": "sum of prompt and completion tokens across every model call"},
        "compare_baseline": True,
        "repo_plan": [
            {
                "repo": "demo",
                "language": "Python",
                "git_commit": commit,
                "task_count": 1,
                "status": "ready",
                "index": {"path": "indexes/demo"},
            }
        ],
        "results": [
            {
                "task_id": "demo-task",
                "repo": "demo",
                "question": "Where is it?",
                "passed": True,
                "duration_seconds": 1.0,
                "tokens": 330,
                "prompt_tokens": 300,
                "output_tokens": 30,
                "cached_tokens": 160,
                "net_tokens": 170,
                "api_calls": 2,
                "baseline_passed": True,
                "baseline_duration_seconds": 2.0,
                "baseline_tokens": 500,
                "baseline_prompt_tokens": 450,
                "baseline_output_tokens": 50,
                "baseline_cached_tokens": 100,
                "baseline_net_tokens": 400,
                "baseline_api_calls": 3,
                "token_savings_percentage": 34.0,
                "time_savings_percentage": 50.0,
            }
        ],
        "total_tasks": 1,
    }

    output = render(report)

    assert "Run status: complete" in output
    assert "codebase-navigator: 0.3.60" in output
    assert "Judge model: deepseek/deepseek-v4-pro" in output
    assert f"demo (Python) @ {commit} [ready, index=indexes/demo]" in output
    assert (
        "Tokens (cumulative): total=330, prompt=300, output=30, cached=160, net=170, calls=2"
        in output
    )
    assert "cumulative API tokens +34.0%" in output


def test_run_directory_resolves_bundled_report_and_trace(tmp_path: Path):
    run_dir = tmp_path / "run_20260904_120000_000000"
    run_dir.mkdir()

    report_path = resolve_report_path(run_dir)

    assert report_path == run_dir / "report.json"
    assert resolve_trace_path(report_path, None) == run_dir / "log.jsonl"
    assert resolve_trace_path(report_path, str(run_dir)) == run_dir / "log.jsonl"
    assert resolve_report_path(run_dir / "legacy.json") == run_dir / "legacy.json"


def test_render_exposes_repository_index_failure_without_task_results():
    report = {
        "status": "failed",
        "repo_plan": [
            {
                "repo": "broken",
                "git_commit": "b" * 40,
                "task_count": 1,
                "status": "index_failed",
                "error": "ctags failed",
            }
        ],
        "results": [],
    }

    output = render(report)

    assert "Run status: failed" in output
    assert "index_failed" in output
    assert "error=ctags failed" in output
    assert "aborted/incomplete: 0/1" in output
