from pathlib import Path

from codebase_navigator.cli import format_search_results, format_tag_results


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


def test_cn_parser_subcommands():
    from codebase_navigator.cli import build_parser

    parser = build_parser()
    assert parser.prog == "cn"

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

    # ask
    args = parser.parse_args([
        "ask", "How does indexing work?", "/my/repo",
        "--model", "test-model",
        "--endpoint", "https://api.openai.com/v1",
        "--api-key", "test-key",
        "--limit", "15",
        "--max-searches", "4",
        "--index-dir", "/custom/idx",
        "-q",
    ])
    assert args.command == "ask"
    assert args.question == "How does indexing work?"
    assert args.folder == "/my/repo"
    assert args.model == "test-model"
    assert args.endpoint == "https://api.openai.com/v1"
    assert args.api_key == "test-key"
    assert args.limit == 15
    assert args.max_searches == 4
    assert args.index_dir == "/custom/idx"
    assert args.quiet is True


def test_cn_main_dispatch(monkeypatch, tmp_path: Path):
    import codebase_navigator.cli as cli_mod
    from codebase_navigator.cli import main

    called = {}

    def mock_status(folder, custom_index_dir=None):
        called["status"] = (folder, custom_index_dir)

    def mock_sync(folder, force=False, custom_index_dir=None):
        called["sync"] = (folder, force, custom_index_dir)

    def mock_watch(folder, debounce_ms=1000, custom_index_dir=None):
        called["watch"] = (folder, debounce_ms, custom_index_dir)

    def mock_search(folder, query, limit=5, doc_type="all", links="auto", theme="auto", wrap=None, width=None, custom_index_dir=None):
        called["search"] = (folder, query, limit, doc_type, links, theme, wrap, width, custom_index_dir)

    def mock_tags(folder, symbol, exact=False, limit=20):
        called["tags"] = (folder, symbol, exact, limit)

    def mock_ask(folder, question, model=None, endpoint=None, api_key=None, limit=None, max_searches=None, links="auto", theme="auto", wrap=None, width=None, custom_index_dir=None, quiet=False):
        called["ask"] = (folder, question, model, endpoint, api_key, limit, max_searches, links, theme, wrap, width, custom_index_dir, quiet)

    monkeypatch.setattr(cli_mod, "_run_status", mock_status)
    monkeypatch.setattr(cli_mod, "_run_sync", mock_sync)
    monkeypatch.setattr(cli_mod, "_run_watch", mock_watch)
    monkeypatch.setattr(cli_mod, "_run_search", mock_search)
    monkeypatch.setattr(cli_mod, "_run_tags", mock_tags)
    monkeypatch.setattr(cli_mod, "_run_ask", mock_ask)

    main(["status", str(tmp_path)])
    assert called["status"] == (tmp_path.resolve(), None)

    main(["sync", str(tmp_path), "--force"])
    assert called["sync"] == (tmp_path.resolve(), True, None)

    main(["watch", str(tmp_path), "--debounce", "2000"])
    assert called["watch"] == (tmp_path.resolve(), 2000, None)

    main(["search", "hello world", str(tmp_path), "--limit", "3", "--type", "code", "--links", "osc8", "--theme", "dark", "--no-wrap", "--width", "80"])
    assert called["search"] == (tmp_path.resolve(), "hello world", 3, "code", "osc8", "dark", False, 80, None)

    main(["tags", "MySymbol", str(tmp_path), "--exact"])
    assert called["tags"] == (tmp_path.resolve(), "MySymbol", True, 20)

    main(["ask", "What is the architecture?", str(tmp_path), "--model", "custom-model", "--limit", "12", "--links", "terminal", "--theme", "light", "--wrap", "--width", "90"])
    assert called["ask"] == (tmp_path.resolve(), "What is the architecture?", "custom-model", None, None, 12, None, "terminal", "light", True, 90, None, False)


def test_format_output_links_and_wrapping():
    from codebase_navigator.cli import format_output_links

    raw = (
        "The `classify_retail_destination` function in "
        "[src/policy.py:86-123](file:///home/user/repo/src/policy.py#L86-L123) "
        "is indeed the code that returns the `retail_destination` output."
    )

    # Markdown mode: preserves full markdown link without forced wrapping
    assert format_output_links(raw, mode="markdown") == raw

    # Terminal / clean mode with wrapping at 50 cols (no color)
    wrapped_term = format_output_links(raw, mode="terminal", wrap=True, width=50, color=False)
    assert "src/policy.py:86-123" in wrapped_term
    for line in wrapped_term.splitlines():
        assert len(line) <= 55

    # Color mode (dark theme): backticks light blue and file links green
    colored_dark = format_output_links(raw, mode="terminal", wrap=True, width=60, color=True, theme="dark")
    assert "\033[38;5;75m`classify_retail_destination`\033[0m" in colored_dark
    assert "\033[32msrc/policy.py:86-123\033[0m" in colored_dark

    # Color mode (light theme): backticks dark blue and file links dark green
    colored_light = format_output_links(raw, mode="terminal", wrap=True, width=60, color=True, theme="light")
    assert "\033[38;5;26m`classify_retail_destination`\033[0m" in colored_light
    assert "\033[38;5;28msrc/policy.py:86-123\033[0m" in colored_light

    # OSC 8 mode with wrapping at 50 cols and color
    wrapped_osc8 = format_output_links(raw, mode="osc8", wrap=True, width=50, color=True, theme="dark")
    assert "\033[32m\033]8;;file:///home/user/repo/src/policy.py#L86-L123\033\\" in wrapped_osc8
    assert "src/policy.py:86-123" in wrapped_osc8


def test_wrap_terminal_text_structures():
    from codebase_navigator.cli import wrap_terminal_text

    sample = (
        "- Item one has very long text that will wrap around to the next line nicely with indentation.\n"
        "```python\n"
        "long_code_line_that_must_not_be_wrapped_under_any_circumstances = 123456789\n"
        "```\n"
        "### Header Title\n"
        "> Blockquote with long text that also wraps with blockquote prefix."
    )

    wrapped = wrap_terminal_text(sample, width=40)
    # Check code block was untouched
    assert "long_code_line_that_must_not_be_wrapped_under_any_circumstances = 123456789" in wrapped
    # Check header was untouched
    assert "### Header Title" in wrapped
    # Check bullet item indented subsequent lines
    assert "  " in wrapped


def test_theme_code_blocks_formatting():
    from codebase_navigator.cli import colorize_terminal_text

    sample = "```python\nprint('hello')\n```"

    dark_out = colorize_terminal_text(sample, theme="dark")
    assert "\033[91m```python\033[0m" in dark_out
    assert "\033[36mprint('hello')\033[0m" in dark_out

    light_out = colorize_terminal_text(sample, theme="light")
    assert "\033[31m```python\033[0m" in light_out
    assert "\033[38;5;30mprint('hello')\033[0m" in light_out





