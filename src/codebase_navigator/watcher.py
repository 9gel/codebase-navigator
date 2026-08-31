"""Filesystem watcher for live ctags, LanceDB vector re-indexing, and agent session hosting."""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any

from watchfiles import Change, DefaultFilter, watch

from .ask import AgentSession, LLMConfig
from .config import CODE_EXTENSIONS, DOC_EXTENSIONS, IGNORE_DIR_NAMES, get_socket_path
from .index import VectorIndex
from .ipc import IPCServer, ping_socket
from .tags import TagsManager


class SourceFilter(DefaultFilter):
    """Filter that includes only recognized source/doc extensions and ignores build/git artifacts."""

    def __init__(self, folder: Path):
        super().__init__(
            ignore_dirs=tuple(IGNORE_DIR_NAMES),
            ignore_entity_patterns=(r"^\..*", r".*\.tags$", r".*tags$", r".*\.sock$"),
        )
        self.folder = folder

    def __call__(self, change: Change, path: str) -> bool:
        if not super().__call__(change, path):
            return False

        p = Path(path)
        if p.name.startswith("."):
            return False

        for part in p.parts:
            if part in IGNORE_DIR_NAMES or part.startswith("."):
                return False

        ext = p.suffix.lower()
        return ext in CODE_EXTENSIONS or ext in DOC_EXTENSIONS


class DirectoryWatcher:
    """Watches folder, serves IPC Unix socket, keeps LanceDB and .tags up to date, and hosts agent sessions."""

    def __init__(
        self,
        folder: Path,
        debounce_ms: int = 1000,
        custom_index_dir: str | None = None,
    ):
        self.folder = folder
        self.debounce_ms = debounce_ms
        self.custom_index_dir = custom_index_dir
        self.tags_mgr = TagsManager(folder)
        self.index = VectorIndex(folder, custom_index_dir)
        self.socket_path = get_socket_path(folder, custom_index_dir)
        self.index_lock = threading.Lock()
        self.session: AgentSession | None = None
        self.session_lock = threading.Lock()
        self.ipc_server = IPCServer(self.socket_path, self.index, lock=self.index_lock, watcher=self)

    def handle_ask(self, question: str, cfg_data: dict[str, Any], new_session: bool = False) -> str:
        """Execute or continue an LLM agent reasoning session within the daemon."""
        with self.session_lock:
            config = LLMConfig(
                endpoint=cfg_data.get("endpoint", "https://openrouter.ai/api/v1"),
                api_key=cfg_data.get("api_key"),
                model=cfg_data.get("model", "google/gemini-2.5-flash"),
                max_searches=int(cfg_data.get("max_searches", 15)),
                initial_limit=int(cfg_data.get("initial_limit", 10)),
            )
            if self.session is None or new_session:
                self.session = AgentSession(self.folder, config, custom_index_dir=self.custom_index_dir)
            else:
                self.session.config = config

            return self.session.ask(question, verbose=False)

    def start(self):
        """Run blocking live watcher loop and IPC server."""
        if self.socket_path.exists():
            active = ping_socket(self.socket_path, timeout=0.5)
            if active is not None:
                print(f"⚠️  Another cn watch instance is already running for: {self.folder}")
                print(f"   Active socket: {self.socket_path}")
                return

        print(f"🚀 Starting cn watch for: {self.folder}")
        print("  Performing initial sync...")
        ok, msg = self.tags_mgr.generate()
        print(f"  .tags: {msg}")

        with self.index_lock:
            u_files, u_chunks, p_files = self.index.sync()
        print(f"  LanceDB: {u_files} files updated ({u_chunks} chunks), {p_files} pruned.")
        print(f"  Index location: {self.index.cache_dir}")

        self.ipc_server.start()
        print(f"  🔌 IPC Socket: {self.socket_path}")
        print("  🧠 Agent Session Daemon: Ready (KV prompt caching enabled)")
        print("👀 Watching for file changes (Ctrl+C to stop)...\n")

        source_filter = SourceFilter(self.folder)

        try:
            for changes in watch(
                self.folder,
                watch_filter=source_filter,
                debounce=self.debounce_ms,
                step=50,
            ):
                code_changed = False
                doc_changed = False
                affected_files: list[Path] = []

                for change_type, change_path in changes:
                    p = Path(change_path)
                    ext = p.suffix.lower()
                    if ext in CODE_EXTENSIONS:
                        code_changed = True
                        affected_files.append(p)
                    elif ext in DOC_EXTENSIONS:
                        doc_changed = True
                        affected_files.append(p)

                if not affected_files:
                    continue

                t0 = time.time()
                ts_str = time.strftime("%H:%M:%S")

                if code_changed:
                    ok, msg = self.tags_mgr.generate()
                    print(f"[{ts_str}] 🏷️  Tags updated: {msg}")

                total_chunks = 0
                for fpath in affected_files:
                    try:
                        with self.index_lock:
                            n_chunks = self.index.update_single_file(fpath)
                        total_chunks += n_chunks
                    except Exception as e:
                        print(f"[{ts_str}] ⚠️ Error updating embeddings for {fpath.name}: {e}")

                dt = (time.time() - t0) * 1000
                print(f"[{ts_str}] ⚡ Synced {len(affected_files)} file(s) ({total_chunks} chunks) in {dt:.0f}ms")

        except KeyboardInterrupt:
            print("\n👋 cn watch stopped.")
        finally:
            self.ipc_server.stop()
