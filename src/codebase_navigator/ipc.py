"""Unix domain socket and TCP loopback IPC server and client for fast semantic querying and agent session hosting."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import socketserver
import threading
from typing import TYPE_CHECKING, Any

from . import __version__
from .config import get_default_tcp_port, get_port_path, get_socket_path

if TYPE_CHECKING:
    from .index import VectorIndex
    from .watcher import DirectoryWatcher


class _IPCRequestHandler(socketserver.StreamRequestHandler):
    """Handles single client connection requests over Unix Domain Socket or TCP loopback."""

    server: _IPCServerMixin

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
                        "version": __version__,
                    }
                else:
                    resp = {"status": "error", "error": f"Unknown action: {action}"}
            except Exception as e:
                resp = {"status": "error", "error": str(e)}

            try:
                response_bytes = json.dumps(resp).encode("utf-8") + b"\n"
                self.wfile.write(response_bytes)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                break


class _IPCServerMixin:
    index: VectorIndex
    lock: threading.Lock
    watcher: DirectoryWatcher | None


class _IPCUnixStreamServer(socketserver.ThreadingUnixStreamServer, _IPCServerMixin):
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


class _IPCTCPServer(socketserver.ThreadingTCPServer, _IPCServerMixin):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
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
    """IPC server hosting in-memory index and agent sessions over Unix Domain Socket and/or TCP loopback."""

    def __init__(
        self,
        socket_path: Path,
        index: VectorIndex,
        lock: threading.Lock | None = None,
        watcher: DirectoryWatcher | None = None,
        folder: Path | None = None,
        custom_index_dir: str | None = None,
    ):
        self.socket_path = socket_path
        self.index = index
        self.lock = lock or threading.Lock()
        self.watcher = watcher
        self.folder = folder or (watcher.folder if watcher else socket_path.parent.parent)
        self.custom_index_dir = custom_index_dir

        self.port_path = get_port_path(self.folder, custom_index_dir)
        self.tcp_port: int | None = None

        self._unix_server: _IPCUnixStreamServer | None = None
        self._tcp_server: _IPCTCPServer | None = None
        self._threads: list[threading.Thread] = []

    def start(self):
        """Start Unix and/or TCP servers in background threads."""
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Pre-check if an active instance is already running via socket or port
        existing_target = discover_daemon_target(self.folder, self.custom_index_dir)
        if existing_target is not None:
            raise RuntimeError(
                f"Another cn watch instance is already running for {self.folder} (target: {existing_target})"
            )

        # 1. Start Unix domain socket server if supported
        try:
            if self.socket_path.exists():
                try:
                    self.socket_path.unlink()
                except OSError:
                    pass

            self._unix_server = _IPCUnixStreamServer(
                str(self.socket_path),
                _IPCRequestHandler,
                index=self.index,
                lock=self.lock,
                watcher=self.watcher,
            )
            t_unix = threading.Thread(target=self._unix_server.serve_forever, daemon=True)
            t_unix.start()
            self._threads.append(t_unix)
        except OSError:
            # Fall back gracefully if Unix domain sockets are blocked in sandboxed container
            self._unix_server = None

        # 2. Start TCP loopback server on 127.0.0.1 (deterministic hash port or fallback)
        base_port = get_default_tcp_port(self.folder)
        tcp_server = None

        # Probe port range around deterministic hash port: [base_port .. base_port + 50]
        for port in range(base_port, min(65535, base_port + 50)):
            try:
                tcp_server = _IPCTCPServer(
                    ("127.0.0.1", port),
                    _IPCRequestHandler,
                    index=self.index,
                    lock=self.lock,
                    watcher=self.watcher,
                )
                self.tcp_port = port
                break
            except OSError:
                continue

        # If none in range bound, try OS ephemeral unprivileged port (port 0)
        if tcp_server is None:
            try:
                tcp_server = _IPCTCPServer(
                    ("127.0.0.1", 0),
                    _IPCRequestHandler,
                    index=self.index,
                    lock=self.lock,
                    watcher=self.watcher,
                )
                self.tcp_port = tcp_server.server_address[1]
            except Exception:
                tcp_server = None

        if tcp_server:
            self._tcp_server = tcp_server
            try:
                self.port_path.write_text(str(self.tcp_port))
            except Exception:
                pass
            t_tcp = threading.Thread(target=self._tcp_server.serve_forever, daemon=True)
            t_tcp.start()
            self._threads.append(t_tcp)

        if not self._unix_server and not self._tcp_server:
            raise RuntimeError("Failed to bind either Unix Domain Socket or TCP loopback port for cn watch.")

    def stop(self):
        """Stop servers and clean up socket/port files."""
        if self._unix_server:
            try:
                self._unix_server.shutdown()
                self._unix_server.server_close()
            except Exception:
                pass
            self._unix_server = None

        if self._tcp_server:
            try:
                self._tcp_server.shutdown()
                self._tcp_server.server_close()
            except Exception:
                pass
            self._tcp_server = None

        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

        if self.port_path.exists():
            try:
                self.port_path.unlink()
            except OSError:
                pass


def _connect_to_daemon(target: Path | int | tuple[str, int], timeout: float) -> socket.socket:
    """Establish connection to daemon via Unix socket path or TCP (host, port)."""
    if isinstance(target, Path):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(str(target))
        return sock
    elif isinstance(target, int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(("127.0.0.1", target))
        return sock
    elif isinstance(target, tuple):
        host, port = target
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        return sock
    else:
        raise ValueError(f"Invalid connection target: {target}")


def discover_daemon_target(folder: Path, custom_index_dir: str | None = None) -> Path | int | None:
    """Discover reachable daemon connection target (Unix socket or TCP port)."""
    # 1. Try port file first (TCP loopback)
    port_path = get_port_path(folder, custom_index_dir)
    if port_path.exists():
        try:
            port_val = int(port_path.read_text().strip())
            # Quick ping to verify alive
            if ping_target(port_val, timeout=0.3) is not None:
                return port_val
        except (ValueError, OSError):
            pass

    # 2. Try Unix domain socket
    sock_path = get_socket_path(folder, custom_index_dir)
    if sock_path.exists():
        if ping_target(sock_path, timeout=0.3) is not None:
            return sock_path

    # 3. Try deterministic hashed port (useful when file mounts in sandboxes are isolated)
    default_port = get_default_tcp_port(folder)
    if ping_target(default_port, timeout=0.3) is not None:
        return default_port

    return None


def send_target_command(
    target: Path | int | tuple[str, int],
    action: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 180.0,
    progress_callback = None,
) -> dict[str, Any] | None:
    """Send command to daemon target and return parsed response with streaming support."""
    try:
        sock = _connect_to_daemon(target, timeout)
    except Exception:
        # Clean up stale files if target was a dead path/port
        if isinstance(target, Path) and target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        return None

    try:
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
    except (OSError, socket.error, json.JSONDecodeError, TimeoutError):
        return None
    finally:
        sock.close()


def send_socket_command(
    socket_path: Path,
    action: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 180.0,
    progress_callback = None,
) -> dict[str, Any] | None:
    """Backward compatibility alias: send command via target or socket path."""
    return send_target_command(
        socket_path,
        action=action,
        payload=payload,
        timeout=timeout,
        progress_callback=progress_callback,
    )


def ping_target(target: Path | int | tuple[str, int], timeout: float = 0.5) -> dict[str, Any] | None:
    """Check if target daemon is active and return its status info."""
    return send_target_command(target, action="status", timeout=timeout)


def ping_socket(socket_path: Path, timeout: float = 0.5) -> dict[str, Any] | None:
    """Check if cn watch daemon is active via socket path."""
    return ping_target(socket_path, timeout=timeout)


def query_target(
    target: Path | int | tuple[str, int],
    query: str,
    limit: int = 5,
    doc_type: str = "all",
    timeout: float = 3.0,
) -> list[dict[str, Any]] | None:
    """Query running daemon for fast vector search."""
    res = send_target_command(
        target,
        action="search",
        payload={"query": query, "limit": limit, "type": doc_type},
        timeout=timeout,
    )
    if res and res.get("status") == "ok":
        return res.get("results")
    return None


def query_socket(
    socket_path: Path,
    query: str,
    limit: int = 5,
    doc_type: str = "all",
    timeout: float = 3.0,
) -> list[dict[str, Any]] | None:
    """Query running cn watch daemon via socket path."""
    return query_target(socket_path, query, limit=limit, doc_type=doc_type, timeout=timeout)
