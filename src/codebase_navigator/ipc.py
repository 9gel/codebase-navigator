"""Unix domain socket IPC server and client for fast semantic querying and agent session hosting."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import socketserver
import threading
from typing import TYPE_CHECKING, Any

from . import __version__

if TYPE_CHECKING:
    from .index import VectorIndex
    from .watcher import DirectoryWatcher


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
                client_ver = req.get("version")

                # Validate client version (unless ping)
                if action != "ping" and client_ver and client_ver != __version__:
                    resp = {
                        "type": "final",
                        "status": "version_mismatch",
                        "error": (
                            f"Version mismatch! cn client ({client_ver}) != cn watch daemon ({__version__}).\n"
                            f"👉 Please restart 'cn watch' so both client and daemon run version {client_ver}."
                        ),
                        "server_version": __version__,
                        "client_version": client_ver,
                    }
                elif action == "ping":
                    resp = {"status": "ok", "pong": True, "version": __version__}
                elif action == "search":
                    query = req.get("query", "")
                    limit = int(req.get("limit", 5))
                    doc_type = req.get("type", "all")
                    with self.server.lock:
                        results = self.server.index.search(query, limit=limit, doc_type=doc_type)
                    resp = {"status": "ok", "results": results}
                elif action == "ask":
                    question = req.get("question", "")
                    cfg_data = req.get("config", {})
                    new_session = req.get("new_session", False)
                    verbose = req.get("verbose", True)
                    if self.server.watcher:
                        def progress_cb(msg_text: str):
                            try:
                                prog_payload = json.dumps({"type": "progress", "message": msg_text}).encode("utf-8") + b"\n"
                                self.wfile.write(prog_payload)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                pass

                        answer, stats = self.server.watcher.handle_ask(
                            question,
                            cfg_data,
                            new_session=new_session,
                            verbose=verbose,
                            progress_callback=progress_cb if verbose else None,
                        )
                        resp = {"type": "final", "status": "ok", "answer": answer, "stats": stats}
                    else:
                        resp = {"type": "final", "status": "error", "error": "Watcher daemon not configured for ask"}
                elif action == "reset_session":
                    if self.server.watcher and self.server.watcher.session:
                        self.server.watcher.session.reset()
                    resp = {"status": "ok", "reset": True}
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

            try:
                response_bytes = json.dumps(resp).encode("utf-8") + b"\n"
                self.wfile.write(response_bytes)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break


class _IPCUnixStreamServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: str,
        RequestHandlerClass,
        index: VectorIndex,
        lock: threading.Lock,
        watcher: DirectoryWatcher | None = None,
    ):
        self.index = index
        self.lock = lock
        self.watcher = watcher
        super().__init__(server_address, RequestHandlerClass)


class IPCServer:
    """Unix Domain Socket server hosting in-memory index and agent sessions for fast queries."""

    def __init__(
        self,
        socket_path: Path,
        index: VectorIndex,
        lock: threading.Lock | None = None,
        watcher: DirectoryWatcher | None = None,
    ):
        self.socket_path = socket_path
        self.index = index
        self.lock = lock or threading.Lock()
        self.watcher = watcher
        self._server: _IPCUnixStreamServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        """Start socket server in a background thread."""
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            status = ping_socket(self.socket_path, timeout=0.5)
            if status is not None:
                raise RuntimeError(
                    f"Another cn watch instance is already running on {self.socket_path}"
                )
            try:
                self.socket_path.unlink()
            except OSError:
                pass

        self._server = _IPCUnixStreamServer(
            str(self.socket_path),
            _IPCRequestHandler,
            index=self.index,
            lock=self.lock,
            watcher=self.watcher,
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


def send_socket_command(
    socket_path: Path,
    action: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 180.0,
    progress_callback = None,
) -> dict[str, Any] | None:
    """Send arbitrary command to cn watch daemon and return response dict with streaming support."""
    if not socket_path.exists():
        return None

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path))
        req = {"action": action, "version": __version__, **(payload or {})}
        payload_bytes = json.dumps(req).encode("utf-8") + b"\n"
        sock.sendall(payload_bytes)

        buffer = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line_str = line_bytes.decode("utf-8").strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    if data.get("type") == "progress":
                        if progress_callback:
                            progress_callback(data.get("message", ""))
                    else:
                        return data
                except json.JSONDecodeError:
                    continue

        if not buffer:
            return None

        return None
    except ConnectionRefusedError:
        try:
            socket_path.unlink()
        except OSError:
            pass
        return None
    except (OSError, socket.error, json.JSONDecodeError, TimeoutError):
        return None
    finally:
        sock.close()


def query_socket(
    socket_path: Path,
    query: str,
    limit: int = 5,
    doc_type: str = "all",
    timeout: float = 3.0,
) -> list[dict[str, Any]] | None:
    """Query the running cn watch daemon via Unix Domain Socket for fast vector search."""
    res = send_socket_command(
        socket_path,
        action="search",
        payload={"query": query, "limit": limit, "type": doc_type},
        timeout=timeout,
    )
    if res and res.get("status") == "ok":
        return res.get("results")
    return None


def ping_socket(socket_path: Path, timeout: float = 0.5) -> dict[str, Any] | None:
    """Check if cn watch daemon is active and return its status info."""
    return send_socket_command(socket_path, action="status", timeout=timeout)
