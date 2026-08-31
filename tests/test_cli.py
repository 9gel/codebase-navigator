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


def test_dt_parser_subcommands():
    from devel_tools.cli import build_parser

    parser = build_parser()
    assert parser.prog == "dt"

    # search
    args = parser.parse_args(["search", "test query", "/some/path", "--limit", "10", "--type", "md"])
    assert args.command == "search"
    assert args.query == "test query"
    assert args.folder == "/some/path"
    assert args.limit == 10
    assert args.type == "md"

    # tags
    args = parser.parse_args(["tags", "MyClass", "--exact", "--limit", "5"])
    assert args.command == "tags"
    assert args.symbol == "MyClass"
    assert args.folder == "."
    assert args.exact is True
    assert args.limit == 5

    # status
    args = parser.parse_args(["status", "--index-dir", "/tmp/idx"])
    assert args.command == "status"
    assert args.folder == "."
    assert args.index_dir == "/tmp/idx"

    # sync
    args = parser.parse_args(["sync", "my_repo", "--force"])
    assert args.command == "sync"
    assert args.folder == "my_repo"
    assert args.force is True

    # watch
    args = parser.parse_args(["watch", "--debounce", "500"])
    assert args.command == "watch"
    assert args.folder == "."
    assert args.debounce == 500


def test_dt_main_dispatch(monkeypatch, tmp_path: Path):
    from devel_tools.cli import main
    import devel_tools.cli as cli_mod

    called = {}

    def mock_status(folder, custom_index_dir=None):
        called["status"] = (folder, custom_index_dir)

    def mock_sync(folder, force=False, custom_index_dir=None):
        called["sync"] = (folder, force, custom_index_dir)

    def mock_watch(folder, debounce_ms=1000, custom_index_dir=None):
        called["watch"] = (folder, debounce_ms, custom_index_dir)

    def mock_search(folder, query, limit=5, doc_type="all", custom_index_dir=None):
        called["search"] = (folder, query, limit, doc_type, custom_index_dir)

    def mock_tags(folder, symbol, exact=False, limit=20):
        called["tags"] = (folder, symbol, exact, limit)

    monkeypatch.setattr(cli_mod, "_run_status", mock_status)
    monkeypatch.setattr(cli_mod, "_run_sync", mock_sync)
    monkeypatch.setattr(cli_mod, "_run_watch", mock_watch)
    monkeypatch.setattr(cli_mod, "_run_search", mock_search)
    monkeypatch.setattr(cli_mod, "_run_tags", mock_tags)

    main(["status", str(tmp_path)])
    assert called["status"] == (tmp_path.resolve(), None)

    main(["sync", str(tmp_path), "--force"])
    assert called["sync"] == (tmp_path.resolve(), True, None)

    main(["watch", str(tmp_path), "--debounce", "2000"])
    assert called["watch"] == (tmp_path.resolve(), 2000, None)

    main(["search", "hello world", str(tmp_path), "--limit", "3", "--type", "code"])
    assert called["search"] == (tmp_path.resolve(), "hello world", 3, "code", None)

    main(["tags", "MySymbol", str(tmp_path), "--exact"])
    assert called["tags"] == (tmp_path.resolve(), "MySymbol", True, 20)
