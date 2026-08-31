"""CLI command line interfaces and formatting for codebase-navigator."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from .config import get_socket_path
from .ipc import ping_socket, query_socket
from .tags import TagsManager, get_available_files


def supports_osc8() -> bool:
    """Check if the current terminal environment supports OSC 8 hyperlinks."""
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_HYPERLINK") in ("1", "true", "yes"):
        return True

    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if term_program in ("vscode", "iterm.app", "wezterm", "hyper", "warp", "ghostty", "rio"):
        return True
    if "TMUX" in os.environ:
        return True
    if os.environ.get("VTE_VERSION"):
        try:
            return int(os.environ["VTE_VERSION"]) >= 5000
        except ValueError:
            return True
    return any(k in os.environ for k in ("KITTY_PID", "ALACRITTY_LOG", "WT_SESSION", "DOMTERM"))


def format_output_links(text: str, mode: str = "auto") -> str:
    """Format markdown links for terminal or markdown output.

    Modes:
    - 'auto': OSC 8 if TTY + supported, clean path if TTY without OSC 8, markdown if non-TTY
    - 'osc8': Embed OSC 8 terminal hyperlinks (\x1b]8;;url\x1b\\label\x1b]8;;\x1b\\)
    - 'terminal' / 'clean': Strip markdown link syntax, leaving clean path:line
    - 'markdown': Preserve [label](file://...) markdown link syntax
    """
    if mode == "markdown":
        return text

    is_tty = sys.stdout.isatty()
    if mode == "auto" and not is_tty:
        return text

    pat = re.compile(r"\[([^\]]+)\]\((file://[^\)]+)\)")
    use_osc8 = (mode == "osc8") or (mode == "auto" and supports_osc8())

    if use_osc8:
        return pat.sub(lambda m: f"\033]8;;{m.group(2)}\033\\{m.group(1)}\033]8;;\033\\", text)
    else:
        return pat.sub(lambda m: m.group(1), text)


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


def build_parser() -> argparse.ArgumentParser:
    """Build the unified cn CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="cn",
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
    p_search.add_argument(
        "--links",
        choices=["auto", "markdown", "terminal", "osc8"],
        default="auto",
        help="Link formatting: auto (detect terminal), markdown, terminal (clean path:line), osc8 (default: auto)",
    )
    p_search.add_argument("--index-dir", default=None, help="Custom LanceDB directory")

    # tags
    p_tags = subparsers.add_parser("tags", help="Lookup symbol definition in .tags")
    p_tags.add_argument("symbol", help="Symbol name or regex pattern")
    p_tags.add_argument("folder", nargs="?", default=".", help="Target folder (default: current directory)")
    p_tags.add_argument("--exact", action="store_true", help="Match exact symbol name")
    p_tags.add_argument("--limit", type=int, default=20, help="Maximum results (default: 20)")

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask an LLM questions about the codebase using iterative semantic search")
    p_ask.add_argument("question", help="Question about the codebase")
    p_ask.add_argument("folder", nargs="?", default=".", help="Target folder (default: current directory)")
    p_ask.add_argument("--model", default=None, help="LLM model name (default: google/gemini-2.5-flash)")
    p_ask.add_argument("--endpoint", "--base-url", dest="endpoint", default=None, help="OpenAI-compatible LLM endpoint (default: https://openrouter.ai/api/v1)")
    p_ask.add_argument("--api-key", default=None, help="LLM API key")
    p_ask.add_argument("--limit", type=int, default=None, help="Initial search results count (default: 10)")
    p_ask.add_argument("--max-searches", type=int, default=None, help="Max additional LLM-driven searches (default: 5)")
    p_ask.add_argument(
        "--links",
        choices=["auto", "markdown", "terminal", "osc8"],
        default="auto",
        help="Link formatting: auto (detect terminal), markdown, terminal (clean path:line), osc8 (default: auto)",
    )
    p_ask.add_argument("--index-dir", default=None, help="Custom LanceDB directory")
    p_ask.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    return parser


def main(argv: list[str] | None = None):
    """Main cn entrypoint with subcommands."""
    parser = build_parser()
    args = parser.parse_args(argv)
    folder = Path(args.folder).resolve()

    if args.command == "status":
        _run_status(folder, custom_index_dir=args.index_dir)
    elif args.command == "sync":
        _run_sync(folder, force=args.force, custom_index_dir=args.index_dir)
    elif args.command == "watch":
        _run_watch(folder, debounce_ms=args.debounce, custom_index_dir=args.index_dir)
    elif args.command == "search":
        _run_search(folder, args.query, limit=args.limit, doc_type=args.type, links=args.links, custom_index_dir=args.index_dir)
    elif args.command == "tags":
        _run_tags(folder, args.symbol, exact=args.exact, limit=args.limit)
    elif args.command == "ask":
        _run_ask(
            folder,
            args.question,
            model=args.model,
            endpoint=args.endpoint,
            api_key=args.api_key,
            limit=args.limit,
            max_searches=args.max_searches,
            links=args.links,
            custom_index_dir=args.index_dir,
            quiet=args.quiet,
        )


# Backward compatibility aliases
main_nav = main


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
        print("  🏷️  Tags file: Not found (run cn sync)")

    socket_path = get_socket_path(folder, custom_index_dir)
    daemon_status = ping_socket(socket_path)
    if daemon_status:
        print(f"  🟢 cn watch daemon: ACTIVE (socket: {socket_path})")
    else:
        print(f"  ⚪ cn watch daemon: NOT RUNNING (socket: {socket_path})")

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


def _run_search(
    folder: Path,
    query: str,
    limit: int = 5,
    doc_type: str = "all",
    links: str = "auto",
    custom_index_dir: str | None = None,
):
    socket_path = get_socket_path(folder, custom_index_dir)
    results = query_socket(socket_path, query, limit=limit, doc_type=doc_type)
    if results is not None:
        raw_out = format_search_results(results, folder)
        print(format_output_links(raw_out, mode=links))
        return

    # Fallback to direct in-process search
    from .index import VectorIndex
    idx = VectorIndex(folder, custom_index_dir)
    results = idx.search(query, limit=limit, doc_type=doc_type)
    raw_out = format_search_results(results, folder)
    print(format_output_links(raw_out, mode=links))


def _run_tags(folder: Path, symbol: str, exact: bool = False, limit: int = 20):
    mgr = TagsManager(folder)
    results = mgr.lookup_symbol(symbol, exact=exact, limit=limit)
    print(format_tag_results(results))


def _run_ask(
    folder: Path,
    question: str,
    model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    limit: int | None = None,
    max_searches: int | None = None,
    links: str = "auto",
    custom_index_dir: str | None = None,
    quiet: bool = False,
):
    from .ask import ask_codebase, load_llm_config

    config = load_llm_config(
        folder=folder,
        cli_overrides={
            "model": model,
            "endpoint": endpoint,
            "api_key": api_key,
            "limit": limit,
            "max_searches": max_searches,
        },
    )

    try:
        answer = ask_codebase(
            folder=folder,
            question=question,
            config=config,
            custom_index_dir=custom_index_dir,
            verbose=not quiet,
        )
        print(format_output_links(answer, mode=links))
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
