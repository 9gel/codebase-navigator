from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.rejection_runner import run_rejection_benchmark
from eval.runner import (
    DEFAULT_JUDGE_MODEL,
    LiveTaskProgress,
    _display_width,
    _run_task,
    ensure_repo_cloned,
    hash_index_tree,
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


def test_existing_eval_repo_is_never_fetched_or_updated(tmp_path: Path):
    repo_dir = tmp_path / "eval" / "repos" / "demo"
    (repo_dir / ".git").mkdir(parents=True)

    with patch("eval.runner.subprocess.run") as run:
        assert ensure_repo_cloned("demo", "https://example.invalid/demo.git", repo_dir) is True

    run.assert_not_called()


def test_existing_non_git_repo_is_not_replaced(tmp_path: Path):
    repo_dir = tmp_path / "eval" / "repos" / "demo"
    repo_dir.mkdir(parents=True)

    with patch("eval.runner.subprocess.run") as run:
        assert ensure_repo_cloned("demo", "https://example.invalid/demo.git", repo_dir) is False

    run.assert_not_called()


def test_live_progress_renders_bounded_in_place_line_per_worker():
    stream = io.StringIO()
    progress = LiveTaskProgress(stream=stream, max_lines=4)
    for number in range(4):
        progress.update(
            str(number),
            f"task-{number}·CN",
            f"Worker phase {number} with action output that would otherwise wrap...",
        )
    with patch("eval.runner.shutil.get_terminal_size", return_value=SimpleNamespace(columns=120)):
        progress.start()
        time.sleep(0.12)
        progress.write("one task completed")
        progress.stop()

    output = stream.getvalue()
    assert "\r\033[2K" in output
    assert "\033[4A" in output
    for number in range(4):
        assert f"task-{number}·CN" in output
    assert output.rfind("task-0·CN") > output.find("one task completed")
    assert "\033[M" in output
    rendered_rows = [line[line.index("(") :] for line in output.splitlines() if "·CN" in line]
    assert rendered_rows
    assert all(row.startswith("(0.") for row in rendered_rows)
    assert all(_display_width(row) <= 80 for row in rendered_rows)


def test_task_progress_phases_are_logged_before_completion(tmp_path: Path):
    events: list[tuple[str, dict]] = []

    def fake_ask_codebase(**kwargs):
        kwargs["progress_callback"]("🔍 Opening local index...")
        kwargs["progress_callback"]("🔍 Embedding search query...")
        return "answer", {"total_tokens": 1}

    with patch("eval.runner.ask_codebase", side_effect=fake_ask_codebase):
        result = _run_task(
            "demo",
            tmp_path,
            "a" * 40,
            tmp_path / "index",
            {"id": "task", "question": "Question?", "expected_answer_key": "answer"},
            SimpleNamespace(api_key="key"),
            "judge",
            False,
            False,
            False,
            event_callback=lambda event_type, **entry: events.append((event_type, entry)),
        )

    assert result["passed"] is True
    assert [entry[1]["message"] for entry in events] == [
        "🔍 Opening local index...",
        "🔍 Embedding search query...",
    ]
    assert all(event_type == "progress" for event_type, _entry in events)
    assert all(entry["task_id"] == "task" for _event_type, entry in events)


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
    repo_dir = tmp_path / "repos" / "demo"
    repo_dir.mkdir(parents=True)
    commit = "a" * 40

    def fake_prepare(
        repo_name,
        _repo_dir,
        indexes_dir,
        git_url,
        git_commit,
        codebase_navigator_git_commit,
    ):
        index_dir = indexes_dir / repo_name / codebase_navigator_git_commit[:12]
        index_dir.mkdir(parents=True)
        metadata = {
            "repo": repo_name,
            "git_url": git_url,
            "git_commit": git_commit,
            "indexed_files": 1,
            "indexed_chunks": 2,
            "index_tree_sha256": hash_index_tree(index_dir),
            "cache_status": "built",
        }
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
        patch("eval.runner.REPOS_DIR", tmp_path / "repos"),
        patch("eval.runner.INDEXES_DIR", tmp_path / "repos" / "_indexes"),
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
    assert report["repo_plan"][0]["index"]["cache_status"] == "built"
    assert report["repo_plan"][0]["index"]["unchanged_during_run"] is True
    assert report["index_integrity_verified"] is True
    assert report["results"][0]["repo_git_commit"] == commit
    assert (report_file.parent / "indexes" / "demo.json").is_file()
    assert report["judge_model"] == "judge-model"
    task_args = run_task.call_args.args
    assert task_args[2] == commit
    assert task_args[3] == tmp_path / "repos" / "_indexes" / "demo" / commit[:12]
    assert task_args[6] == "judge-model"
    log_entries = [
        json.loads(line)
        for line in (report_file.parent / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    index_event = next(entry for entry in log_entries if entry["type"] == "index")
    verification_event = next(
        entry for entry in log_entries if entry["type"] == "index_verification"
    )
    assert index_event["index_tree_sha256"] == report["repo_plan"][0]["index"]["index_tree_sha256"]
    assert verification_event["unchanged"] is True


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
    repo_dir = tmp_path / "repos" / "demo"
    repo_dir.mkdir(parents=True)

    with (
        patch("eval.runner.BENCHMARK_CONFIG", benchmark_file),
        patch("eval.runner.REPOS_DIR", tmp_path / "repos"),
        patch("eval.runner.INDEXES_DIR", tmp_path / "repos" / "_indexes"),
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


def test_parallel_eval_ctrl_c_cancels_futures_and_preserves_report(tmp_path: Path):
    benchmark_file = tmp_path / "benchmarks.json"
    benchmark_file.write_text(
        json.dumps(
            [
                {
                    "repo": "demo",
                    "git_url": "https://example.invalid/demo.git",
                    "tasks": [
                        {"id": "one", "question": "One?", "expected_answer_key": "one"},
                        {"id": "two", "question": "Two?", "expected_answer_key": "two"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    repo_dir = tmp_path / "repos" / "demo"
    repo_dir.mkdir(parents=True)
    commit = "2" * 40

    def fake_prepare(repo_name, _repo_dir, indexes_dir, _url, _repo_commit, nav_commit):
        index_dir = indexes_dir / repo_name / nav_commit[:12]
        index_dir.mkdir(parents=True)
        return index_dir, {
            "git_commit": commit,
            "index_tree_sha256": hash_index_tree(index_dir),
            "cache_status": "built",
        }

    futures = [MagicMock(), MagicMock()]
    pool = MagicMock()
    pool.submit.side_effect = futures

    with (
        patch("eval.runner.BENCHMARK_CONFIG", benchmark_file),
        patch("eval.runner.REPOS_DIR", tmp_path / "repos"),
        patch("eval.runner.INDEXES_DIR", tmp_path / "repos" / "_indexes"),
        patch("eval.runner.ensure_repo_cloned", return_value=True),
        patch("eval.runner.get_git_commit", return_value=commit),
        patch(
            "eval.runner.load_llm_config",
            return_value=SimpleNamespace(api_key="test-key", model="candidate-model"),
        ),
        patch("eval.runner.prepare_evaluation_index", side_effect=fake_prepare),
        patch("eval.runner.ThreadPoolExecutor", return_value=pool),
        patch("eval.runner.as_completed", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        run_benchmark(runs_dir=tmp_path / "runs", workers=2)

    pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    for future in futures:
        future.cancel.assert_called_once_with()
    report_file = next((tmp_path / "runs").glob("run_*/report.json"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["completed_tasks"] == 0
    log_entries = [
        json.loads(line)
        for line in (report_file.parent / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["type"] == "interrupted" for entry in log_entries)


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
    indexes_dir = tmp_path / "repos" / "_indexes"
    repo_commit = "b" * 40
    navigator_commit = "c" * 40

    with (
        patch("eval.runner.TagsManager") as tags_manager,
        patch("eval.runner.VectorIndex") as vector_index,
    ):
        tags_manager.return_value.generate.return_value = (True, "tags ready")
        vector_index.return_value.sync.return_value = (3, 7, 0)
        index_dir, metadata = prepare_evaluation_index(
            "demo",
            repo_dir,
            indexes_dir,
            "https://example.invalid/demo.git",
            repo_commit,
            navigator_commit,
        )

    assert index_dir == indexes_dir / "demo" / navigator_commit[:12]
    assert metadata["repo_git_commit"] == repo_commit
    assert metadata["codebase_navigator_git_commit"] == navigator_commit
    assert metadata["indexed_files"] == 3
    assert metadata["indexed_chunks"] == 7
    assert metadata["cache_status"] == "built"
    assert metadata["index_tree_sha256"] == hash_index_tree(index_dir)
    tag_build_path = tags_manager.call_args.kwargs["tag_file"]
    vector_build_path = Path(vector_index.call_args.kwargs["custom_index_dir"])
    assert tag_build_path.parent == vector_build_path
    assert tag_build_path.name == ".tags"
    vector_index.return_value.sync.assert_called_once_with(force=True)
    sidecar = index_dir.parent / f"{index_dir.name}.metadata.json"
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved["index_tree_sha256"] == metadata["index_tree_sha256"]


def test_prepare_evaluation_index_reuses_verified_cache(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    indexes_dir = tmp_path / "indexes"
    repo_commit = "d" * 40
    navigator_commit = "e" * 40

    with (
        patch("eval.runner.TagsManager") as tags_manager,
        patch("eval.runner.VectorIndex") as vector_index,
    ):
        tags_manager.return_value.generate.return_value = (True, "tags ready")
        vector_index.return_value.sync.return_value = (1, 2, 0)
        first_dir, first_metadata = prepare_evaluation_index(
            "demo",
            repo_dir,
            indexes_dir,
            "https://example.invalid/demo.git",
            repo_commit,
            navigator_commit,
        )
        second_dir, second_metadata = prepare_evaluation_index(
            "demo",
            repo_dir,
            indexes_dir,
            "https://example.invalid/demo.git",
            repo_commit,
            navigator_commit,
        )

    assert first_dir == second_dir
    assert first_metadata["cache_status"] == "built"
    assert second_metadata["cache_status"] == "reused"
    assert second_metadata["index_tree_sha256"] == first_metadata["index_tree_sha256"]
    tags_manager.return_value.generate.assert_called_once()
    vector_index.return_value.sync.assert_called_once()


def test_prepare_evaluation_index_rejects_modified_cache(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    indexes_dir = tmp_path / "indexes"
    repo_commit = "f" * 40
    navigator_commit = "1" * 40

    with (
        patch("eval.runner.TagsManager") as tags_manager,
        patch("eval.runner.VectorIndex") as vector_index,
    ):
        tags_manager.return_value.generate.return_value = (True, "tags ready")
        vector_index.return_value.sync.return_value = (1, 2, 0)
        index_dir, _ = prepare_evaluation_index(
            "demo",
            repo_dir,
            indexes_dir,
            "https://example.invalid/demo.git",
            repo_commit,
            navigator_commit,
        )

    (index_dir / "tampered").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        prepare_evaluation_index(
            "demo",
            repo_dir,
            indexes_dir,
            "https://example.invalid/demo.git",
            repo_commit,
            navigator_commit,
        )


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
