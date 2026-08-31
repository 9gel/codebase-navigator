"""CLI command line interfaces and formatting for devel-tools."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import get_socket_path
from .ipc import ping_socket, query_socket
from .tags import TagsManager, get_available_files


def format_search_results(results: list[dict], base_folder: Path) -> str:
    """Format results into GitHub markdown with clickable links."""
    if not results:
        return "No matching documentation or code comments found."

    lines = []
    for idx, r in enumerate(results, start=1):
        rel_p = r["path"]
        abs_p = r["abs_path"]
        s_line = r["start_line"]
        e_line = r["end_line"]
        title = r["title"]
        score_pct = int(r["score"] * 100)
        content = r["content"]

        lines.append(
            f"### {idx}. [{rel_p}:L{s_line}-L{e_line}](file://{abs_p}#L{s_line}-L{e_line}) — {title} (Match: {score_pct}%)"
        )
        lines.append("```")
        c_lines = content.splitlines()
        if len(c_lines) > 8:
            lines.extend(c_lines[:6])
            lines.append("...")
        else:
            lines.extend(c_lines)
        lines.append("```\n")

    return "\n".join(lines)


def format_tag_results(results: list[dict]) -> str:
    """Format ctags symbol results."""
    if not results:
        return "No symbols found."

    lines = []
    for idx, r in enumerate(results, start=1):
        sym = r["symbol"]
        kind = r["kind"]
        rel_p = r["path"]
        abs_p = r["abs_path"]
        line_no = r["line"]
        preview = r["preview"]

        lines.append(
            f"{idx}. `{sym}` ({kind}) -> [{rel_p}:L{line_no}](file://{abs_p}#L{line_no})"
        )
        if preview:
            lines.append(f"   `{preview}`")

    return "\n".join(lines)


def main_nav():
    """Main devel-nav entrypoint with subcommands."""
    parser = argparse.ArgumentParser(
        prog="devel-nav",
        description="Generic Code & Documentation Navigation Engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Check indexing status")
    p_status.add_argument("folder", nargs="?", default=".", help="Target folder (default: current directory)")
    p_status.add_argument("--index-dir", default=None, help="Custom LanceDB directory")

    # sync
    p_sync = subparsers.add_parser("sync", help="Synchronize .tags and LanceDB index")
    p_sync.add_argument("folder", nargs="?", default=".", help="Target folder (default: current directory)")
    p_sync.add_argument("--force", action="store_true", help="Force complete re-indexing")
    p_sync.add_argument("--index-dir", default=None, help="Custom LanceDB directory")

    # watch
    p_watch = subparsers.add_parser("watch", help="Watch folder and continuously update indexes")
    p_watch.add_argument("folder", nargs="?", default=".", help="Target folder (default: current directory)")
    p_watch.add_argument("--debounce", type=int, default=1000, help="Debounce milliseconds (default: 1000)")
    p_watch.add_argument("--index-dir", default=None, help="Custom LanceDB directory")

    # search
    p_search = subparsers.add_parser("search", help="Semantic search in docs and code")
    p_search.add_argument("query", help="Semantic query string")
    p_search.add_argument("folder", nargs="?", default=".", help="Target folder (default: current directory)")
    p_search.add_argument("--limit", type=int, default=5, help="Maximum results (default: 5)")
    p_search.add_argument(
        "--type",
        choices=["all", "md", "code_doc", "markdown", "code"],
        default="all",
        help="Filter document types (default: all)",
    )
    p_search.add_argument("--index-dir", default=None, help="Custom LanceDB directory")

    # tags
    p_tags = subparsers.add_parser("tags", help="Lookup symbol definition in .tags")
    p_tags.add_argument("symbol", help="Symbol name or regex pattern")
    p_tags.add_argument("folder", nargs="?", default=".", help="Target folder (default: current directory)")
    p_tags.add_argument("--exact", action="store_true", help="Match exact symbol name")
    p_tags.add_argument("--limit", type=int, default=20, help="Maximum results (default: 20)")

    args = parser.parse_args()
    folder = Path(args.folder).resolve()

    if args.command == "status":
        _run_status(folder, custom_index_dir=args.index_dir)
    elif args.command == "sync":
        _run_sync(folder, force=args.force, custom_index_dir=args.index_dir)
    elif args.command == "watch":
        _run_watch(folder, debounce_ms=args.debounce, custom_index_dir=args.index_dir)
    elif args.command == "search":
        _run_search(folder, args.query, limit=args.limit, doc_type=args.type, custom_index_dir=args.index_dir)
    elif args.command == "tags":
        _run_tags(folder, args.symbol, exact=args.exact, limit=args.limit)


def main_watch():
    parser = argparse.ArgumentParser(prog="devel-watch", description="Live file watcher for tags & vector indexing")
    parser.add_argument("folder", nargs="?", default=".", help="Folder to watch (default: .)")
    parser.add_argument("--debounce", type=int, default=1000, help="Debounce milliseconds")
    parser.add_argument("--index-dir", default=None, help="Custom index directory")
    args = parser.parse_args()
    _run_watch(Path(args.folder).resolve(), debounce_ms=args.debounce, custom_index_dir=args.index_dir)


def main_sync():
    parser = argparse.ArgumentParser(prog="devel-sync", description="Synchronize .tags and LanceDB embeddings")
    parser.add_argument("folder", nargs="?", default=".", help="Folder to index (default: .)")
    parser.add_argument("--force", action="store_true", help="Force complete re-indexing")
    parser.add_argument("--index-dir", default=None, help="Custom index directory")
    args = parser.parse_args()
    _run_sync(Path(args.folder).resolve(), force=args.force, custom_index_dir=args.index_dir)


def main_search():
    parser = argparse.ArgumentParser(prog="devel-search", description="Semantic search in markdown docs & code comments")
    parser.add_argument("query", help="Semantic query string")
    parser.add_argument("folder", nargs="?", default=".", help="Folder context (default: .)")
    parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--type",
        choices=["all", "md", "code_doc", "markdown", "code"],
        default="all",
        help="Filter document types (default: all)",
    )
    parser.add_argument("--index-dir", default=None, help="Custom index directory")
    args = parser.parse_args()
    _run_search(Path(args.folder).resolve(), args.query, limit=args.limit, doc_type=args.type, custom_index_dir=args.index_dir)


def main_tags():
    parser = argparse.ArgumentParser(prog="devel-tags", description="Symbol lookup in .tags")
    parser.add_argument("symbol", help="Symbol name or regex pattern")
    parser.add_argument("folder", nargs="?", default=".", help="Folder context (default: .)")
    parser.add_argument("--exact", action="store_true", help="Match exact symbol name")
    parser.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    args = parser.parse_args()
    _run_tags(Path(args.folder).resolve(), args.symbol, exact=args.exact, limit=args.limit)


def main_status():
    parser = argparse.ArgumentParser(prog="devel-status", description="Check tags and index status")
    parser.add_argument("folder", nargs="?", default=".", help="Folder context (default: .)")
    parser.add_argument("--index-dir", default=None, help="Custom index directory")
    args = parser.parse_args()
    _run_status(Path(args.folder).resolve(), custom_index_dir=args.index_dir)


def _run_status(folder: Path, custom_index_dir: str | None = None):
    print(f"📊 Navigation Status for: {folder}")
    code_files, doc_files = get_available_files(folder)
    print(f"  Available files: {len(code_files)} source code files, {len(doc_files)} doc files")

    mgr = TagsManager(folder)
    tf = mgr.find_tag_file()
    if tf and tf.exists():
        sz = tf.stat().st_size / (1024 * 1024)
        print(f"  🏷️  Tags file: {tf} ({sz:.2f} MB)")
    else:
        print("  🏷️  Tags file: Not found (run devel-sync)")

    socket_path = get_socket_path(folder, custom_index_dir)
    daemon_status = ping_socket(socket_path)
    if daemon_status:
        print(f"  🟢 devel-watch daemon: ACTIVE (socket: {socket_path})")
    else:
        print(f"  ⚪ devel-watch daemon: NOT RUNNING (socket: {socket_path})")

    from .index import VectorIndex
    idx = VectorIndex(folder, custom_index_dir)
    meta = idx.load_meta()
    chunk_count = sum(m.get("chunks", 0) for m in meta.values())
    print(f"  🧠 Vector index: {idx.cache_dir}")
    print(f"     Indexed files: {len(meta)}, Total chunks: {chunk_count}")


def _run_sync(folder: Path, force: bool = False, custom_index_dir: str | None = None):
    print(f"Discovering git/source files in {folder}...")
    code_files, doc_files = get_available_files(folder)
    print(f"  Found {len(code_files)} source files, {len(doc_files)} doc files.")

    print(f"Updating {folder / '.tags'}...")
    mgr = TagsManager(folder)
    ok, msg = mgr.generate()
    print(f"  .tags generation: {msg if ok else 'FAILED: ' + msg}")

    print("Syncing LanceDB embeddings...")
    from .index import VectorIndex
    idx = VectorIndex(folder, custom_index_dir)
    u_files, u_chunks, p_files = idx.sync(force=force)
    print(f"✓ Complete: {u_files} files updated ({u_chunks} chunks indexed), {p_files} deleted files pruned.")
    print(f"📦 Embedding index location: {idx.cache_dir}")


def _run_watch(folder: Path, debounce_ms: int = 1000, custom_index_dir: str | None = None):
    from .watcher import DirectoryWatcher
    watcher = DirectoryWatcher(folder, debounce_ms=debounce_ms, custom_index_dir=custom_index_dir)
    watcher.start()


def _run_search(folder: Path, query: str, limit: int = 5, doc_type: str = "all", custom_index_dir: str | None = None):
    socket_path = get_socket_path(folder, custom_index_dir)
    results = query_socket(socket_path, query, limit=limit, doc_type=doc_type)
    if results is not None:
        print(format_search_results(results, folder))
        return

    # Fallback to direct in-process search
    from .index import VectorIndex
    idx = VectorIndex(folder, custom_index_dir)
    results = idx.search(query, limit=limit, doc_type=doc_type)
    print(format_search_results(results, folder))


def _run_tags(folder: Path, symbol: str, exact: bool = False, limit: int = 20):
    mgr = TagsManager(folder)
    results = mgr.lookup_symbol(symbol, exact=exact, limit=limit)
    print(format_tag_results(results))
