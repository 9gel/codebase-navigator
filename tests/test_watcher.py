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
