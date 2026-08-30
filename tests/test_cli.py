import pytest
from pathlib import Path
from devel_tools.cli import format_search_results, format_tag_results


def test_format_search_results(tmp_path: Path):
    results = [
        {
            "score": 0.89,
            "path": "GLOSSARY.md",
            "abs_path": str(tmp_path / "GLOSSARY.md"),
            "doc_type": "markdown",
            "title": "GLOSSARY.md > Retail destination",
            "start_line": 15,
            "end_line": 20,
            "content": "**Retail destination** — a place that holds shops.",
        }
    ]
    output = format_search_results(results, tmp_path)
    assert "### 1. [GLOSSARY.md:L15-L20]" in output
    assert "file://" in output
    assert "(Match: 89%)" in output


def test_format_tag_results(tmp_path: Path):
    results = [
        {
            "symbol": "classify_destination",
            "kind": "function",
            "path": "src/policy.py",
            "abs_path": str(tmp_path / "src/policy.py"),
            "line": 42,
            "preview": "def classify_destination():",
        }
    ]
    output = format_tag_results(results)
    assert "1. `classify_destination` (function)" in output
    assert "[src/policy.py:L42]" in output
