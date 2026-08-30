"""Filesystem watcher for live ctags and LanceDB vector re-indexing."""

from __future__ import annotations

from pathlib import Path
import time

from watchfiles import Change, DefaultFilter, watch

from .config import CODE_EXTENSIONS, DOC_EXTENSIONS, IGNORE_DIR_NAMES
from .index import VectorIndex
from .tags import TagsManager


class SourceFilter(DefaultFilter):
    """Filter that includes only recognized source/doc extensions and ignores build/git artifacts."""

    def __init__(self, folder: Path):
        super().__init__(
            ignore_dirs=tuple(IGNORE_DIR_NAMES),
            ignore_entity_patterns=(r"^\..*", r".*\.tags$", r".*tags$"),
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
    """Watches folder and keeps .tags and LanceDB vector embeddings up to date."""

    def __init__(
        self,
        folder: Path,
        debounce_ms: int = 1000,
        custom_index_dir: str | None = None,
    ):
        self.folder = folder
        self.debounce_ms = debounce_ms
        self.tags_mgr = TagsManager(folder)
        self.index = VectorIndex(folder, custom_index_dir)

    def start(self):
        """Run blocking live watcher loop."""
        print(f"🚀 Starting devel-watch for: {self.folder}")
        print("  Performing initial sync...")
        ok, msg = self.tags_mgr.generate()
        print(f"  .tags: {msg}")

        u_files, u_chunks, p_files = self.index.sync()
        print(f"  LanceDB: {u_files} files updated ({u_chunks} chunks), {p_files} pruned.")
        print(f"  Index location: {self.index.cache_dir}")
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

                # Update tags if source code changed
                if code_changed:
                    ok, msg = self.tags_mgr.generate()
                    print(f"[{ts_str}] 🏷️  Tags updated: {msg}")

                # Update LanceDB embeddings incrementally
                total_chunks = 0
                for fpath in affected_files:
                    try:
                        n_chunks = self.index.update_single_file(fpath)
                        total_chunks += n_chunks
                    except Exception as e:
                        print(f"[{ts_str}] ⚠️ Error updating embeddings for {fpath.name}: {e}")

                dt = (time.time() - t0) * 1000
                print(f"[{ts_str}] ⚡ Synced {len(affected_files)} file(s) ({total_chunks} chunks) in {dt:.0f}ms")

        except KeyboardInterrupt:
            print("\n👋 devel-watch stopped.")
