"""Unix domain socket IPC server and client for fast semantic querying via dt watch."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import socketserver
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .index import VectorIndex


class _IPCRequestHandler(socketserver.StreamRequestHandler):
    """Handles single client connection requests over Unix Domain Socket."""

    server: _IPCUnixStreamServer

    def handle(self):
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line.decode("utf-8"))
                action = req.get("action")
                if action == "ping":
                    resp = {"status": "ok", "pong": True}
                elif action == "search":
                    query = req.get("query", "")
                    limit = int(req.get("limit", 5))
                    doc_type = req.get("type", "all")
                    with self.server.lock:
                        results = self.server.index.search(query, limit=limit, doc_type=doc_type)
                    resp = {"status": "ok", "results": results}
                elif action == "status":
                    with self.server.lock:
                        meta = self.server.index.load_meta()
                    chunk_count = sum(m.get("chunks", 0) for m in meta.values())
                    resp = {
                        "status": "ok",
                        "files_count": len(meta),
                        "chunk_count": chunk_count,
                        "cache_dir": str(self.server.index.cache_dir),
                    }
                else:
                    resp = {"status": "error", "error": f"Unknown action: {action}"}
            except Exception as e:
                resp = {"status": "error", "error": str(e)}

            response_bytes = json.dumps(resp).encode("utf-8") + b"\n"
            self.wfile.write(response_bytes)
            self.wfile.flush()


class _IPCUnixStreamServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: str, RequestHandlerClass, index: VectorIndex, lock: threading.Lock):
        self.index = index
        self.lock = lock
        super().__init__(server_address, RequestHandlerClass)


class IPCServer:
    """Unix Domain Socket server hosting in-memory index for fast search queries."""

    def __init__(self, socket_path: Path, index: VectorIndex, lock: threading.Lock | None = None):
        self.socket_path = socket_path
        self.index = index
        self.lock = lock or threading.Lock()
        self._server: _IPCUnixStreamServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        """Start socket server in a background thread.

        Raises RuntimeError if another active daemon is already listening on this socket.
        Cleans up stale socket files from prior crashes automatically.
        """
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            # Check if another process is actively listening on this socket
            status = ping_socket(self.socket_path, timeout=0.5)
            if status is not None:
                raise RuntimeError(
                    f"Another dt watch instance is already running on {self.socket_path}"
                )
            # Socket file exists but no process is listening -> stale socket from prior crash
            try:
                self.socket_path.unlink()
            except OSError:
                pass

        self._server = _IPCUnixStreamServer(
            str(self.socket_path),
            _IPCRequestHandler,
            index=self.index,
            lock=self.lock,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop socket server and clean up socket file."""
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass


def query_socket(
    socket_path: Path,
    query: str,
    limit: int = 5,
    doc_type: str = "all",
    timeout: float = 3.0,
) -> list[dict[str, Any]] | None:
    """Query the running dt watch daemon via Unix Domain Socket.

    Returns search results if successful, or None if socket is unavailable/unresponsive.
    Automatically unlinks stale socket files from dead daemons.
    """
    if not socket_path.exists():
        return None

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path))
        req = {
            "action": "search",
            "query": query,
            "limit": limit,
            "type": doc_type,
        }
        payload = json.dumps(req).encode("utf-8") + b"\n"
        sock.sendall(payload)

        # Read response line
        buffer = b""
        while b"\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk

        if not buffer:
            return None

        line = buffer.split(b"\n")[0]
        resp = json.loads(line.decode("utf-8"))
        if resp.get("status") == "ok":
            return resp.get("results")
        return None
    except ConnectionRefusedError:
        # Socket file exists but no process is listening -> stale socket from prior crash
        try:
            socket_path.unlink()
        except OSError:
            pass
        return None
    except (OSError, socket.error, json.JSONDecodeError, TimeoutError):
        return None
    finally:
        sock.close()


def ping_socket(socket_path: Path, timeout: float = 0.5) -> dict[str, Any] | None:
    """Check if dt watch daemon is active and return its status info.

    Automatically unlinks stale socket files from dead daemons.
    """
    if not socket_path.exists():
        return None

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path))
        req = {"action": "status"}
        payload = json.dumps(req).encode("utf-8") + b"\n"
        sock.sendall(payload)

        buffer = b""
        while b"\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk

        if not buffer:
            return None

        line = buffer.split(b"\n")[0]
        resp = json.loads(line.decode("utf-8"))
        if resp.get("status") == "ok":
            return resp
        return None
    except ConnectionRefusedError:
        # Socket file exists but no process is listening -> stale socket from prior crash
        try:
            socket_path.unlink()
        except OSError:
            pass
        return None
    except (OSError, socket.error, json.JSONDecodeError, TimeoutError):
        return None
    finally:
        sock.close()
