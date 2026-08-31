from pathlib import Path
from watchfiles import Change
from devel_tools.watcher import SourceFilter


def test_source_filter(tmp_path: Path):
    sf = SourceFilter(tmp_path)
    
    # Valid code/doc files
    assert sf(Change.added, str(tmp_path / "main.py")) is True
    assert sf(Change.modified, str(tmp_path / "docs" / "README.md")) is True
    
    # Ignored files/directories
    assert sf(Change.added, str(tmp_path / ".git" / "config")) is False
    assert sf(Change.added, str(tmp_path / ".devel-tools" / "watch.sock")) is False
    assert sf(Change.added, str(tmp_path / ".tags")) is False
    assert sf(Change.added, str(tmp_path / "image.png")) is False
