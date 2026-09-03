from pathlib import Path
from watchfiles import Change
from codebase_navigator.watcher import SourceFilter


def test_source_filter(tmp_path: Path):
    sf = SourceFilter(tmp_path)

    # Valid code/doc files
    assert sf(Change.added, str(tmp_path / "main.py")) is True
    assert sf(Change.modified, str(tmp_path / "docs" / "README.md")) is True

    # Ignored files/directories
    assert sf(Change.added, str(tmp_path / ".git" / "config")) is False
    assert sf(Change.added, str(tmp_path / ".codebase-navigator" / "watch.sock")) is False
    assert sf(Change.added, str(tmp_path / ".devel-tools" / "watch.sock")) is False
    assert sf(Change.added, str(tmp_path / "image.png")) is False


def test_directory_watcher_headless(tmp_path: Path, capsys):
    from codebase_navigator.watcher import DirectoryWatcher
    import time

    watcher = DirectoryWatcher(tmp_path)
    # A captured pytest stream is non-TTY, so interactive=True must still use
    # the headless path and print startup messages rather than entering the TUI.
    import threading

    t = threading.Thread(target=watcher.start, kwargs={"interactive": True}, daemon=True)
    t.start()
    time.sleep(0.1)

    try:
        assert watcher.running is True
        assert watcher.ipc_server is not None
    finally:
        watcher.stop()
        t.join(timeout=2.0)
        assert watcher.running is False

    output = capsys.readouterr().out
    assert "🚀 Starting cn watch" in output
    assert "👀 Watching for file changes" in output


def test_watcher_tui_layout_and_methods(tmp_path: Path):
    from codebase_navigator.tui import WatcherTUI

    submitted = []
    tui = WatcherTUI(
        folder=tmp_path,
        model_name="deepseek/deepseek-v4-flash-0731",
        on_submit=lambda q: submitted.append(q),
        on_reset=lambda: None,
        on_status=lambda: "status ok",
        on_exit=lambda: None,
    )

    assert tui.folder == tmp_path
    assert tui.tokens_total == 0
    tui.update_stats(total=1500, prompt=1200, completion=300, tool_count=2)
    assert tui.tokens_total == 1500
    assert tui.turn_count == 1
    assert tui.last_tool_count == 2


def test_watcher_tui_transcript_does_not_deadlock(tmp_path: Path, capsys):
    from codebase_navigator.tui import WatcherTUI

    tui = WatcherTUI(
        folder=tmp_path,
        model_name="test-model",
        on_submit=lambda _query: None,
        on_reset=lambda: None,
        on_status=lambda: "status ok",
        on_exit=lambda: None,
    )

    tui.write_transcript("startup complete")

    output = capsys.readouterr().out
    assert "startup complete" in output
    assert "Keep `cn watch` up to ensure `cn search` work efficiently elsewhere." in output
    assert f"[{tmp_path.name}] ❯" not in output


def test_watcher_tui_renders_startup_logs_after_screen_setup(tmp_path: Path, monkeypatch, capsys):
    import io

    from codebase_navigator.tui import WatcherTUI

    tui = WatcherTUI(
        folder=tmp_path,
        model_name="test-model",
        on_submit=lambda _query: None,
        on_reset=lambda: None,
        on_status=lambda: "status ok",
        on_exit=lambda: None,
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    tui.run_loop(initial_logs=[".tags: updated", "LanceDB: ready"])

    output = capsys.readouterr().out
    assert output.index(".tags: updated") > output.index("\033[?1049h")
    assert "LanceDB: ready" in output


def test_watcher_tui_exit_keys_show_notice_then_exit(tmp_path: Path, capsys):
    from codebase_navigator.tui import WatcherTUI

    tui = WatcherTUI(
        folder=tmp_path,
        model_name="test-model",
        on_submit=lambda _query: None,
        on_reset=lambda: None,
        on_status=lambda: "status ok",
        on_exit=lambda: None,
    )

    assert tui._handle_exit_key("Ctrl-C") is False
    assert tui.exit_notice == " Press Ctrl-C to exit"
    assert tui._handle_exit_key("Ctrl-C") is True
    assert tui._handle_exit_key("Ctrl-D") is False
    assert tui._handle_exit_key("Ctrl-Q") is False
    assert "Press Ctrl-C to exit" in capsys.readouterr().out


def test_watcher_tui_spinner_is_transient(tmp_path: Path, capsys):
    import time

    from codebase_navigator.tui import WatcherTUI

    tui = WatcherTUI(
        folder=tmp_path,
        model_name="test-model",
        on_submit=lambda _query: None,
        on_reset=lambda: None,
        on_status=lambda: "status ok",
        on_exit=lambda: None,
    )

    tui.write_transcript("latest output")
    capsys.readouterr()
    tui.start_spinner()
    assert tui.spinner_active is True
    time.sleep(0.15)
    tui.stop_spinner()
    spinner_output = capsys.readouterr().out

    assert tui.spinner_active is False
    assert tui._spinner_thread is None
    assert "Agent is working..." in spinner_output
    assert "\033[1;1H" not in spinner_output
    assert "Keep `cn watch` up to ensure `cn search` work efficiently elsewhere." in spinner_output


def test_watcher_tui_prompt_echo_has_theme_colored_dividers(tmp_path: Path, capsys):
    from codebase_navigator.tui import WatcherTUI

    tui = WatcherTUI(
        folder=tmp_path,
        model_name="test-model",
        on_submit=lambda _query: None,
        on_reset=lambda: None,
        on_status=lambda: "status ok",
        on_exit=lambda: None,
    )

    tui._submit_query("why is this here?")

    output = capsys.readouterr().out
    assert "👤 You: why is this here?" in output
    assert "\033[38;5;75m" in output or "\033[34m" in output
    prompt_index = next(
        i for i, line in enumerate(tui.transcript_lines) if "👤 You: why is this here?" in line
    )
    assert prompt_index > 0
    assert prompt_index + 1 < len(tui.transcript_lines)
    assert "─" in tui.transcript_lines[prompt_index - 1]
    assert "─" in tui.transcript_lines[prompt_index + 1]


def test_watcher_tui_mouse_scroll_moves_in_small_steps(tmp_path: Path, capsys):
    from codebase_navigator.tui import WatcherTUI

    tui = WatcherTUI(
        folder=tmp_path,
        model_name="test-model",
        on_submit=lambda _query: None,
        on_reset=lambda: None,
        on_status=lambda: "status ok",
        on_exit=lambda: None,
    )
    for number in range(40):
        tui.write_transcript(f"line {number}")
    capsys.readouterr()

    tui.scroll_transcript(1)

    assert tui.scroll_offset == 1


def test_watcher_tui_status_bar_uses_theme_colors(tmp_path: Path, monkeypatch, capsys):
    from codebase_navigator.tui import WatcherTUI

    monkeypatch.setenv("CN_THEME", "dark")
    tui = WatcherTUI(
        folder=tmp_path,
        model_name="test-model",
        on_submit=lambda _query: None,
        on_reset=lambda: None,
        on_status=lambda: "status ok",
        on_exit=lambda: None,
    )

    tui.render_bottom_chrome()

    output = capsys.readouterr().out
    assert "\033[38;5;252m\033[48;5;238m" in output
    assert "\033[7m" not in output
