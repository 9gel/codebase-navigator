from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.rejection_runner import run_rejection_benchmark
from eval.runner import run_benchmark


def test_eval_runner_saves_report_to_directory(tmp_path: Path):
    reports_dir = tmp_path / "reports"

    with patch("eval.runner.BENCHMARK_CONFIG", tmp_path / "benchmarks.json"):
        (tmp_path / "benchmarks.json").write_text("[]", encoding="utf-8")
        result = run_benchmark(save_report=reports_dir)

    assert result is True
    assert reports_dir.is_dir()
    created_files = list(reports_dir.glob("report_*.json"))
    assert len(created_files) == 1
    assert (tmp_path / "reports.json").exists() is False
    assert (tmp_path / "report.json").exists() is False

    data = json.loads(created_files[0].read_text(encoding="utf-8"))
    assert data["total_tasks"] == 0


def test_eval_runner_saves_report_to_explicit_file(tmp_path: Path):
    custom_file = tmp_path / "custom_out.json"

    with patch("eval.runner.BENCHMARK_CONFIG", tmp_path / "benchmarks.json"):
        (tmp_path / "benchmarks.json").write_text("[]", encoding="utf-8")
        result = run_benchmark(save_report=custom_file)

    assert result is True
    assert custom_file.is_file()
    assert (tmp_path / "reports.json").exists() is False
    assert (tmp_path / "report.json").exists() is False


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
