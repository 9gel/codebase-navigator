"""Filesystem watcher for live ctags, LanceDB vector re-indexing, and agent session hosting."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
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

        self.lifetime_turn_tokens = 0
        self.lifetime_prompt_tokens = 0
        self.lifetime_completion_tokens = 0
        self.turn_count = 0
        self.running = False
        self._stop_event = threading.Event()
        self._watch_thread: threading.Thread | None = None

    def handle_ask(
        self,
        question: str,
        cfg_data: dict[str, Any],
        new_session: bool = False,
        verbose: bool = False,
        progress_callback = None,
    ) -> tuple[str, dict[str, Any]]:
        """Execute or continue an LLM agent reasoning session within the daemon."""
        with self.session_lock:
            config = LLMConfig(
                endpoint=cfg_data.get("endpoint", "https://openrouter.ai/api/v1"),
                api_key=cfg_data.get("api_key"),
                model=cfg_data.get("model", "deepseek/deepseek-v4-flash-0731"),
                max_searches=int(cfg_data.get("max_searches", 15)),
                initial_limit=int(cfg_data.get("initial_limit", 5)),
                system_prompt=cfg_data.get("system_prompt"),
            )
            if self.session is None or new_session:
                self.session = AgentSession(self.folder, config, custom_index_dir=self.custom_index_dir)
            else:
                self.session.config = config

            # Always pass verbose=False to AgentSession in daemon so watcher terminal stays clean,
            # while progress_callback streams updates back to the requesting client over the socket/TCP.
            answer, stats = self.session.ask(
                question,
                verbose=False,
                progress_callback=progress_callback,
            )
            if stats:
                self.lifetime_prompt_tokens = stats.get("turn_prompt_tokens", 0)
                self.lifetime_completion_tokens = stats.get("lifetime_completion_tokens", 0)
                self.lifetime_turn_tokens = self.lifetime_prompt_tokens + self.lifetime_completion_tokens
                self.turn_count += 1
            return answer, stats

    def _watch_loop(self):
        """Background filesystem watcher thread."""
        source_filter = SourceFilter(self.folder)

        try:
            for changes in watch(
                self.folder,
                watch_filter=source_filter,
                debounce=self.debounce_ms,
                step=50,
                stop_event=self._stop_event,
            ):
                if self._stop_event.is_set():
                    break

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
                    self._log_event(f"[{ts_str}] 🏷️  Tags updated: {msg}")

                total_chunks = 0
                for fpath in affected_files:
                    try:
                        with self.index_lock:
                            n_chunks = self.index.update_single_file(fpath)
                        total_chunks += n_chunks
                    except Exception as e:
                        self._log_event(f"[{ts_str}] ⚠️ Error updating embeddings for {fpath.name}: {e}")

                dt = (time.time() - t0) * 1000
                self._log_event(f"[{ts_str}] ⚡ Synced {len(affected_files)} file(s) ({total_chunks} chunks) in {dt:.0f}ms")

        except Exception as e:
            if not self._stop_event.is_set():
                self._log_event(f"⚠️ Watcher error: {e}")

    def _log_event(self, text: str):
        """Print an asynchronous event cleanly above the interactive prompt."""
        # \r\033[2K clears current prompt line before printing
        sys.stderr.write(f"\r\033[2K{text}\n")
        sys.stderr.flush()

    def start(self, interactive: bool = True):
        """Run watcher loop and IPC server, optionally hosting an interactive console."""
        from .ipc import discover_daemon_target

        existing_target = discover_daemon_target(self.folder, self.custom_index_dir)
        if existing_target is not None:
            print(f"⚠️  Another cn watch instance is already running for: {self.folder}")
            print(f"   Active target: {existing_target}")
            return

        is_tty = sys.stdin.isatty() and sys.stdout.isatty() if hasattr(sys.stdin, "isatty") else False
        use_interactive = interactive and is_tty

        print(f"🚀 Starting cn watch for: {self.folder}")
        print("  Performing initial sync...")
        ok, msg = self.tags_mgr.generate()
        print(f"  .tags: {msg}")

        with self.index_lock:
            u_files, u_chunks, p_files = self.index.sync()
        print(f"  LanceDB: {u_files} files updated ({u_chunks} chunks), {p_files} pruned.")
        print(f"  Index location: {self.index.cache_dir}")

        self.ipc_server.start()
        if self.ipc_server._unix_server:
            print(f"  🔌 IPC Socket: {self.socket_path}")
        if self.ipc_server.tcp_port:
            print(f"  🌐 IPC Loopback TCP: 127.0.0.1:{self.ipc_server.tcp_port} (port file: {self.ipc_server.port_path})")
        print("  🧠 Agent Session Daemon: Ready (KV prompt caching enabled)")

        # Start watcher in background thread
        self.running = True
        self._stop_event.clear()
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()

        if not use_interactive:
            print("👀 Watching for file changes (Ctrl+C to stop)...\n")
            try:
                while self.running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                self.stop()
            return

        # Interactive Console Mode
        self._run_interactive_console()

    def _run_interactive_console(self):
        """Host rich interactive REPL console directly inside cn watch."""
        from .ask import load_llm_config
        from .cli import StatusSpinner, detect_terminal_theme, format_output_links

        config = load_llm_config(folder=self.folder)

        print("\n" + "=" * 75)
        print("🧭 Codebase-Navigator Interactive Console")
        print("   Type your question to ask about this codebase.")
        print("   Commands: /reset (clear context), /status (index info), /exit (quit)")
        print("=" * 75 + "\n")

        while self.running:
            try:
                # Render clean prompt with context stats
                short_folder = self.folder.name or str(self.folder)
                model_name = config.model.split("/")[-1]
                stats_str = (
                    f" \033[2m[{self.lifetime_turn_tokens:,} tokens]\033[0m"
                    if self.lifetime_turn_tokens > 0
                    else ""
                )

                prompt_str = f"\033[36m[{short_folder}]\033[0m ❯ "
                user_input = input(prompt_str).strip()

                if not user_input:
                    continue

                if user_input.lower() in ("/exit", "exit", "quit", ":q"):
                    break

                if user_input.lower() in ("/reset", "reset", "/clear", "clear"):
                    with self.session_lock:
                        if self.session:
                            self.session.reset()
                        self.lifetime_turn_tokens = 0
                        self.lifetime_prompt_tokens = 0
                        self.lifetime_completion_tokens = 0
                        self.turn_count = 0
                    print("🧹 Conversation context reset. Started fresh session.\n")
                    continue

                if user_input.lower() in ("/status", "status"):
                    code_files, doc_files = self.index.get_available_files() if hasattr(self.index, "get_available_files") else ([], [])
                    meta = self.index.load_meta()
                    chunks = sum(m.get("chunks", 0) for m in meta.values())
                    print(f"📊 Status: {len(meta)} files indexed ({chunks} chunks). Model: {config.model}")
                    print(f"   Session Context: {self.lifetime_turn_tokens:,} tokens across {self.turn_count} turns.\n")
                    continue

                # Run in-process query with spinner
                spinner: StatusSpinner | None = None

                def handle_progress(line: str):
                    nonlocal spinner
                    if spinner:
                        spinner.update_message(line)
                    else:
                        spinner = StatusSpinner(line, stream=sys.stderr)
                        spinner.start()

                try:
                    spinner = StatusSpinner("🔍 Searching codebase...", stream=sys.stderr)
                    spinner.start()

                    answer, stats = self.handle_ask(
                        user_input,
                        cfg_data=config.to_dict(),
                        new_session=False,
                        verbose=False,
                        progress_callback=handle_progress,
                    )
                finally:
                    if spinner:
                        spinner.stop()

                # Render formatted markdown answer
                term_cols = shutil.get_terminal_size((80, 24)).columns if "shutil" in globals() else 80
                divider_width = min(term_cols, 80)
                print(f"\n\033[36m{'─' * divider_width}\033[0m\n")
                print(
                    format_output_links(
                        answer,
                        mode="auto",
                        wrap=True,
                        theme="auto",
                    )
                )
                print(f"\n\033[36m{'─' * divider_width}\033[0m")

                # Footer stats
                turn_in = stats.get("turn_prompt_tokens", 0)
                turn_out = stats.get("turn_completion_tokens", 0)
                turn_total = stats.get("turn_total_tokens", 0)
                tool_count = stats.get("tool_calls_count", 0)
                tool_suffix = f" | {tool_count} tool call{'s' if tool_count != 1 else ''}" if tool_count > 0 else ""
                print(
                    f"\033[2mTokens: {turn_total:,} (prompt: {turn_in:,}, completion: {turn_out:,}){tool_suffix} | "
                    f"Model: {model_name}\033[0m\n"
                )

            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n(To quit cn watch, type /exit or press Ctrl+D)")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

        self.stop()

    def stop(self):
        """Cleanly stop watcher background thread and IPC servers."""
        self.running = False
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2.0)
        self.ipc_server.stop()
        print("\n👋 cn watch stopped.")
