import time
from pathlib import Path
from devel_tools.config import get_cache_dir, get_socket_path
from devel_tools.index import VectorIndex
from devel_tools.ipc import IPCServer, ping_socket, query_socket
from devel_tools.cli import _run_search


def test_default_cache_dir_and_socket_path(tmp_path: Path):
    cache_dir = get_cache_dir(tmp_path)
    assert cache_dir == tmp_path / ".devel-tools"
    assert cache_dir.exists()

    sock_path = get_socket_path(tmp_path)
    assert sock_path == tmp_path / ".devel-tools" / "watch.sock"


def test_ipc_server_and_socket_query(tmp_path: Path, capsys):
    # Setup test file
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    f1 = doc_dir / "guide.md"
    f1.write_text("# Deployment Guide\n\nInstructions on how to deploy the application service.")

    idx = VectorIndex(tmp_path)
    idx.sync()

    sock_path = get_socket_path(tmp_path)
    server = IPCServer(sock_path, idx)
    server.start()

    try:
        # Give server thread a moment to bind
        time.sleep(0.05)
        assert sock_path.exists()

        # Ping status
        status = ping_socket(sock_path)
        assert status is not None
        assert status.get("status") == "ok"
        assert status.get("files_count") == 1

        # Query via socket
        results = query_socket(sock_path, "deploy application service", limit=3)
        assert results is not None
        assert len(results) >= 1
        assert "guide.md" in results[0]["path"]

        # Test CLI run search using socket
        _run_search(tmp_path, "deploy application service", limit=1)
        captured = capsys.readouterr()
        assert "guide.md" in captured.out
        assert "Deployment Guide" in captured.out
    finally:
        server.stop()
        assert not sock_path.exists()


def test_socket_fallback_when_server_dead(tmp_path: Path, capsys):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    f1 = doc_dir / "faq.md"
    f1.write_text("# FAQ Section\n\nAnswers to frequently asked questions.")

    idx = VectorIndex(tmp_path)
    idx.sync()

    sock_path = get_socket_path(tmp_path)
    # Ensure socket does not exist
    if sock_path.exists():
        sock_path.unlink()

    # Querying a nonexistent socket returns None
    assert query_socket(sock_path, "faq") is None
    assert ping_socket(sock_path) is None

    # CLI search should seamlessly fallback to in-process search
    _run_search(tmp_path, "frequently asked questions", limit=1)
    captured = capsys.readouterr()
    assert "faq.md" in captured.out


def test_duplicate_ipc_server_rejected(tmp_path: Path):
    idx = VectorIndex(tmp_path)
    sock_path = get_socket_path(tmp_path)

    server1 = IPCServer(sock_path, idx)
    server1.start()
    time.sleep(0.05)

    try:
        server2 = IPCServer(sock_path, idx)
        import pytest
        with pytest.raises(RuntimeError, match="Another dt watch instance is already running"):
            server2.start()
    finally:
        server1.stop()


def test_stale_socket_recovery(tmp_path: Path):
    idx = VectorIndex(tmp_path)
    sock_path = get_socket_path(tmp_path)
    sock_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a dummy stale socket/file
    import socket
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(sock_path))
    s.close()  # Closed/dead socket file leftover on disk

    assert sock_path.exists()
    # ping_socket should detect it is dead, return None, and unlink it
    assert ping_socket(sock_path) is None
    assert not sock_path.exists()

    # Now starting a new server should succeed without errors
    server = IPCServer(sock_path, idx)
    server.start()
    time.sleep(0.05)
    try:
        assert sock_path.exists()
        assert ping_socket(sock_path) is not None
    finally:
        server.stop()
