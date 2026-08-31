import pytest
from pathlib import Path

from codebase_navigator.tools import (
    read_code,
    grep_search,
    find_references,
    get_call_tree,
    check_ripgrep_installed,
)
from codebase_navigator.tags import TagsManager


def test_read_code_bounds(tmp_path: Path):
    sample_file = tmp_path / "sample.py"
    lines = [f"line_{i} = {i}" for i in range(1, 51)]
    sample_file.write_text("\n".join(lines), encoding="utf-8")

    # Read slice
    res = read_code(tmp_path, "sample.py", start_line=10, end_line=15)
    assert "error" not in res
    assert res["start_line"] == 10
    assert res["end_line"] == 15
    assert res["total_lines"] == 50
    assert "10 | line_10 = 10" in res["content"]
    assert "15 | line_15 = 15" in res["content"]
    assert "16 | line_16 = 16" not in res["content"]


def test_grep_search_and_references(tmp_path: Path):
    src = tmp_path / "module.py"
    src.write_text(
        """def calculate_total(a, b):
    return a + b

def process_order(x):
    total = calculate_total(x, 10)
    return total
""",
        encoding="utf-8",
    )

    # Generate tags
    tags = TagsManager(tmp_path)
    tags.generate()

    # Find references
    refs = find_references(tmp_path, "calculate_total")
    assert len(refs) >= 2
    types = [r["type"] for r in refs]
    assert "definition" in types
    assert "reference" in types

    # Call tree
    tree = get_call_tree(tmp_path, "calculate_total", path="module.py")
    assert tree["symbol"] == "calculate_total"
    assert len(tree["callers"]) >= 1
    assert any(c.get("caller_function") == "process_order" or "process_order" in c.get("preview", "") for c in tree["callers"])
