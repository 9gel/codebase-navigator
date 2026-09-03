from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.rejection_runner import run_rejection_benchmark
from eval.runner import (
    DEFAULT_JUDGE_MODEL,
    llm_judge_answer,
    prepare_evaluation_index,
    run_baseline_agent,
    run_benchmark,
)


def test_eval_runner_creates_self_contained_run_package(tmp_path: Path):
    runs_dir = tmp_path / "runs"

    with patch("eval.runner.BENCHMARK_CONFIG", tmp_path / "benchmarks.json"):
        (tmp_path / "benchmarks.json").write_text("[]", encoding="utf-8")
        result = run_benchmark(runs_dir=runs_dir)

    assert result is True
    run_dirs = list(runs_dir.glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "log.jsonl").is_file()
    assert (run_dir / "benchmark_tasks.json").is_file()
    assert (run_dir / "indexes").is_dir()

    data = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert data["total_tasks"] == 0
    assert data["judge_model"] == DEFAULT_JUDGE_MODEL
    assert data["token_metric"]["tokens"].startswith("sum of prompt")


def test_eval_runner_captures_repo_commit_and_index_metadata(tmp_path: Path):
    benchmark_file = tmp_path / "benchmarks.json"
    benchmark_file.write_text(
        json.dumps(
            [
                {
                    "repo": "demo",
                    "git_url": "https://example.invalid/demo.git",
                    "language": "Python",
                    "tasks": [
                        {
                            "id": "demo-task",
                            "question": "Where is demo?",
                            "expected_answer_key": "In demo.py",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    repo_dir = tmp_path / "exercises" / "demo"
    repo_dir.mkdir(parents=True)
    commit = "a" * 40

    def fake_prepare(repo_name, _repo_dir, run_dir, git_url, git_commit):
        index_dir = run_dir / "indexes" / repo_name
        index_dir.mkdir()
        metadata = {
            "repo": repo_name,
            "git_url": git_url,
            "git_commit": git_commit,
            "indexed_files": 1,
            "indexed_chunks": 2,
        }
        (index_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return index_dir, metadata

    task_result = {
        "task_id": "demo-task",
        "repo": "demo",
        "repo_git_commit": commit,
        "question": "Where is demo?",
        "passed": True,
    }

    with (
        patch("eval.runner.BENCHMARK_CONFIG", benchmark_file),
        patch("eval.runner.EXERCISES_DIR", tmp_path / "exercises"),
        patch("eval.runner.ensure_repo_cloned", return_value=True),
        patch("eval.runner.get_git_commit", return_value=commit),
        patch(
            "eval.runner.load_llm_config",
            return_value=SimpleNamespace(api_key="test-key", model="candidate-model"),
        ),
        patch("eval.runner.prepare_evaluation_index", side_effect=fake_prepare),
        patch("eval.runner._run_task", return_value=task_result) as run_task,
    ):
        result = run_benchmark(runs_dir=tmp_path / "runs", workers=1, judge_model="judge-model")

    assert result is True
    report_file = next((tmp_path / "runs").glob("run_*/report.json"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["repo_plan"][0]["git_commit"] == commit
    assert report["repo_plan"][0]["index"]["git_commit"] == commit
    assert report["results"][0]["repo_git_commit"] == commit
    assert (report_file.parent / "indexes" / "demo" / "metadata.json").is_file()
    assert report["judge_model"] == "judge-model"
    task_args = run_task.call_args.args
    assert task_args[2] == commit
    assert task_args[3] == report_file.parent / "indexes" / "demo"
    assert task_args[6] == "judge-model"


def test_eval_runner_preserves_report_when_index_build_fails(tmp_path: Path):
    benchmark_file = tmp_path / "benchmarks.json"
    benchmark_file.write_text(
        json.dumps(
            [
                {
                    "repo": "demo",
                    "git_url": "https://example.invalid/demo.git",
                    "tasks": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    repo_dir = tmp_path / "exercises" / "demo"
    repo_dir.mkdir(parents=True)

    with (
        patch("eval.runner.BENCHMARK_CONFIG", benchmark_file),
        patch("eval.runner.EXERCISES_DIR", tmp_path / "exercises"),
        patch("eval.runner.ensure_repo_cloned", return_value=True),
        patch("eval.runner.get_git_commit", return_value="a" * 40),
        patch(
            "eval.runner.load_llm_config",
            return_value=SimpleNamespace(api_key="test-key", model="candidate-model"),
        ),
        patch("eval.runner.prepare_evaluation_index", side_effect=RuntimeError("index broke")),
        pytest.raises(RuntimeError, match="index broke"),
    ):
        run_benchmark(runs_dir=tmp_path / "runs", workers=1)

    report_file = next((tmp_path / "runs").glob("run_*/report.json"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["repo_plan"][0]["status"] == "index_failed"
    assert report["repo_plan"][0]["error"] == "index broke"
    assert (report_file.parent / "log.jsonl").is_file()


def test_judge_uses_dedicated_default_model():
    response = {"choices": [{"message": {"content": '{"is_correct": true, "rationale": "ok"}'}}]}
    config = SimpleNamespace(endpoint="https://example.invalid", api_key="key")
    with patch("eval.runner.call_chat_completions", return_value=response) as call:
        passed, rationale = llm_judge_answer("q", "key", "answer", config)

    assert passed is True
    assert rationale == "ok"
    assert call.call_args.args[2]["model"] == DEFAULT_JUDGE_MODEL


def test_prepare_evaluation_index_captures_vector_and_ctags_metadata(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    run_dir = tmp_path / "run_20260904_120000_000000"
    (run_dir / "indexes").mkdir(parents=True)
    commit = "b" * 40

    with (
        patch("eval.runner.TagsManager") as tags_manager,
        patch("eval.runner.VectorIndex") as vector_index,
    ):
        tags_manager.return_value.generate.return_value = (True, "tags ready")
        vector_index.return_value.sync.return_value = (3, 7, 0)
        index_dir, metadata = prepare_evaluation_index(
            "demo", repo_dir, run_dir, "https://example.invalid/demo.git", commit
        )

    assert index_dir == run_dir / "indexes" / "demo"
    assert metadata["git_commit"] == commit
    assert metadata["indexed_files"] == 3
    assert metadata["indexed_chunks"] == 7
    tags_manager.assert_called_once_with(repo_dir, tag_file=index_dir / ".tags")
    vector_index.assert_called_once_with(repo_dir, custom_index_dir=str(index_dir))
    vector_index.return_value.sync.assert_called_once_with(force=True)
    saved = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved == metadata


def test_baseline_token_metrics_accumulate_every_model_call(tmp_path: Path):
    (tmp_path / "demo.py").write_text("value = 1\n", encoding="utf-8")
    first = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "one",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "demo.py"}),
                            },
                        }
                    ],
                }
            }
        ],
    }
    second = {
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 120},
        },
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
    }
    config = SimpleNamespace(model="candidate", endpoint="endpoint", api_key="key", max_searches=2)

    with patch("eval.runner.call_chat_completions", side_effect=[first, second]):
        answer, stats = run_baseline_agent(tmp_path, "question", config)

    assert answer == "done"
    assert stats["prompt_tokens"] == 300
    assert stats["completion_tokens"] == 30
    assert stats["cached_tokens"] == 160
    assert stats["total_tokens"] == 330
    assert stats["net_tokens"] == 170
    assert stats["api_calls"] == 2


def test_rejection_runner_saves_report_to_directory(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    tasks_file = tmp_path / "rejection_tasks.json"
    tasks_file.write_text("[]", encoding="utf-8")

    result = run_rejection_benchmark(
        benchmark_file=tasks_file,
        target_repo=tmp_path,
        save_report=reports_dir,
        api_key="test-key",
    )

    assert result is True
    assert reports_dir.is_dir()
    created_files = list(reports_dir.glob("rejection_report_*.json"))
    assert len(created_files) == 1
    assert (tmp_path / "reports.json").exists() is False
    assert (tmp_path / "report.json").exists() is False
    assert (tmp_path / "rejection_report.json").exists() is False
