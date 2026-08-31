"""CLI command line interfaces and formatting for codebase-navigator."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import textwrap
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


def wrap_terminal_text(text: str, width: int | None = None) -> str:
    """Wrap markdown-like text for terminal display, preserving code blocks, bullet indentation, and headers."""
    if width is None:
        term_cols = shutil.get_terminal_size((80, 24)).columns
        width = max(40, min(term_cols, 100))

    lines = text.splitlines()
    wrapped_lines = []
    in_code_block = False

    bullet_re = re.compile(r"^(\s*[-*+]|\s*\d+\.)\s+(.*)$")
    header_re = re.compile(r"^(#{1,6})\s+(.*)$")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            wrapped_lines.append(line)
            continue

        if in_code_block or not stripped:
            wrapped_lines.append(line)
            continue

        # Check for bullet point (e.g. "* item", "- item", "1. item")
        bm = bullet_re.match(line)
        if bm:
            prefix = bm.group(1)
            content = bm.group(2)
            indent_len = len(prefix) + 1
            w = textwrap.TextWrapper(
                width=width,
                initial_indent=f"{prefix} ",
                subsequent_indent=" " * indent_len,
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines.extend(w.wrap(content))
            continue

        # Check for blockquote (e.g. "> text")
        if stripped.startswith(">"):
            content = stripped[1:].strip()
            w = textwrap.TextWrapper(
                width=width,
                initial_indent="> ",
                subsequent_indent="> ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines.extend(w.wrap(content))
            continue

        # Check for markdown header (e.g. "## Header")
        if header_re.match(stripped):
            wrapped_lines.append(line)
            continue

        # Regular paragraph line
        w = textwrap.TextWrapper(
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(w.wrap(line))

    return "\n".join(wrapped_lines)


def colorize_terminal_text(text: str) -> str:
    """Colorize inline markdown for terminal display: backticks in light blue, bold in bold."""
    lines = text.splitlines()
    res = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            res.append(f"\033[36m{line}\033[0m")
            continue
        if in_code:
            res.append(line)
            continue

        # Color inline backticks `code` in light blue (\033[38;5;75m)
        line = re.sub(r"(`[^`\n]+`)", r"\033[38;5;75m\1\033[0m", line)
        # Bold **text**
        line = re.sub(r"(\*\*[^*\n]+\*\*)", r"\033[1m\1\033[0m", line)
        res.append(line)
    return "\n".join(res)


def format_output_links(
    text: str,
    mode: str = "auto",
    wrap: bool | None = None,
    width: int | None = None,
    color: bool | None = None,
) -> str:
    """Format markdown links, wrap lines, and colorize output for terminal display.

    Modes:
    - 'auto': OSC 8 if TTY + supported, clean path if TTY without OSC 8, markdown if non-TTY
    - 'osc8': Embed OSC 8 terminal hyperlinks (\x1b]8;;url\x1b\\label\x1b]8;;\x1b\\)
    - 'terminal' / 'clean': Strip markdown link syntax, leaving clean path:line
    - 'markdown': Preserve [label](file://...) markdown link syntax
    """
    if mode == "markdown":
        return text

    is_tty = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
    if mode == "auto" and not is_tty:
        return text

    should_wrap = wrap if wrap is not None else (is_tty or mode in ("osc8", "terminal"))
    should_color = color if color is not None else (is_tty and not os.environ.get("NO_COLOR"))

    pat = re.compile(r"\[([^\]]+)\]\((file://[^\)]+)\)")
    use_osc8 = (mode == "osc8") or (mode == "auto" and supports_osc8())

    # Extract links into indexed placeholders so wrapping measures exact visible width
    extracted_links: list[tuple[str, str]] = []

    def save_link(m: re.Match) -> str:
        idx = len(extracted_links)
        label, url = m.group(1), m.group(2)
        extracted_links.append((label, url))
        return f"__CN_LINK_{idx}__"

    tokenized = pat.sub(save_link, text)

    def restore_visible(t: str) -> str:
        for idx, (label, _) in enumerate(extracted_links):
            t = t.replace(f"__CN_LINK_{idx}__", label)
        return t

    if should_wrap:
        visible_text = restore_visible(tokenized)
        processed = wrap_terminal_text(visible_text, width=width)
    else:
        processed = restore_visible(tokenized)

    if should_color:
        processed = colorize_terminal_text(processed)

    for label, url in extracted_links:
        if use_osc8 and should_color:
            replacement = f"\033[32m\033]8;;{url}\033\\{label}\033]8;;\033\\\033[0m"
        elif use_osc8:
            replacement = f"\033]8;;{url}\033\\{label}\033]8;;\033\\"
        elif should_color:
            replacement = f"\033[32m{label}\033[0m"
        else:
            replacement = label
        processed = processed.replace(label, replacement, 1)

    return processed


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
    p_search.add_argument("--wrap", action=argparse.BooleanOptionalAction, default=None, help="Wrap terminal lines (default: enabled on TTY)")
    p_search.add_argument("--width", type=int, default=None, help="Wrap line width (default: terminal width)")
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
    p_ask.add_argument("--wrap", action=argparse.BooleanOptionalAction, default=None, help="Wrap terminal lines (default: enabled on TTY)")
    p_ask.add_argument("--width", type=int, default=None, help="Wrap line width (default: terminal width)")
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
        _run_search(
            folder,
            args.query,
            limit=args.limit,
            doc_type=args.type,
            links=args.links,
            wrap=args.wrap,
            width=args.width,
            custom_index_dir=args.index_dir,
        )
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
            wrap=args.wrap,
            width=args.width,
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
    wrap: bool | None = None,
    width: int | None = None,
    custom_index_dir: str | None = None,
):
    socket_path = get_socket_path(folder, custom_index_dir)
    results = query_socket(socket_path, query, limit=limit, doc_type=doc_type)
    if results is not None:
        raw_out = format_search_results(results, folder)
        print(format_output_links(raw_out, mode=links, wrap=wrap, width=width))
        return

    # Fallback to direct in-process search
    from .index import VectorIndex
    idx = VectorIndex(folder, custom_index_dir)
    results = idx.search(query, limit=limit, doc_type=doc_type)
    raw_out = format_search_results(results, folder)
    print(format_output_links(raw_out, mode=links, wrap=wrap, width=width))


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
    wrap: bool | None = None,
    width: int | None = None,
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
        if not quiet:
            term_w = min(width or shutil.get_terminal_size((80, 24)).columns, 80)
            print(f"\n\033[36m{'=' * term_w}\033[0m\n")
        print(format_output_links(answer, mode=links, wrap=wrap, width=width))
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
