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


def test_directory_watcher_headless(tmp_path: Path):
    from codebase_navigator.watcher import DirectoryWatcher
    import time

    watcher = DirectoryWatcher(tmp_path)
    # Start headless (interactive=False) on background thread
    import threading
    t = threading.Thread(target=watcher.start, kwargs={"interactive": False}, daemon=True)
    t.start()
    time.sleep(0.1)

    try:
        assert watcher.running is True
        assert watcher.ipc_server is not None
    finally:
        watcher.stop()
        t.join(timeout=2.0)
        assert watcher.running is False


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

    assert "startup complete" in capsys.readouterr().out
