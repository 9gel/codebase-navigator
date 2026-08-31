"""CLI command line interfaces and formatting for codebase-navigator."""

from __future__ import annotations

import argparse
import os
import re
import select
import shutil
import sys
import textwrap
from pathlib import Path

from .config import get_display_config, get_socket_path
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


def detect_terminal_theme(theme_override: str | None = None) -> str:
    """Detect terminal background theme ('dark' or 'light')."""
    if theme_override and theme_override != "auto":
        return "light" if "light" in theme_override.lower() else "dark"

    # 1. Environment variables
    env_theme = (
        os.environ.get("CN_THEME")
        or os.environ.get("CODEBASE_NAVIGATOR_THEME")
        or os.environ.get("TERMINAL_THEME")
    )
    if env_theme:
        return "light" if "light" in env_theme.lower() else "dark"

    # 2. COLORFGBG check (format: "fg;bg" or "fg;bg;...")
    colorfgbg = os.environ.get("COLORFGBG")
    if colorfgbg:
        parts = colorfgbg.split(";")
        try:
            bg = int(parts[-1])
            return "light" if bg in (7, 11, 14, 15) or bg > 8 else "dark"
        except ValueError:
            pass

    for k in ["BAT_THEME", "ITERM_PROFILE"]:
        v = os.environ.get(k, "").lower()
        if "light" in v:
            return "light"
        if "dark" in v:
            return "dark"

    # 3. Interactive TTY OSC 11 background query
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                sys.stdout.write("\033]11;?\033\\")
                sys.stdout.flush()
                r, _, _ = select.select([sys.stdin], [], [], 0.02)
                if r:
                    resp = ""
                    while True:
                        ch = sys.stdin.read(1)
                        resp += ch
                        if ch in ("\a", "\\") or len(resp) > 30:
                            break
                    if "rgb:" in resp:
                        rgb_part = resp.split("rgb:")[1].rstrip("\a\033\\")
                        components = rgb_part.split("/")
                        if len(components) == 3:
                            r_val = int(components[0][:2], 16) / 255.0
                            g_val = int(components[1][:2], 16) / 255.0
                            b_val = int(components[2][:2], 16) / 255.0
                            luminance = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
                            return "light" if luminance >= 0.5 else "dark"
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:  # noqa: BLE001, S110
            pass

    return "dark"


def wrap_terminal_text(text: str, width: int | None = None) -> str:
    """Wrap markdown-like text for terminal display, preserving code blocks, bullet indentation, and headers."""
    term_cols = shutil.get_terminal_size((80, 24)).columns
    if width is not None and width > 0:
        target_width = min(width, term_cols) if term_cols > 0 else width
        target_width = max(20, target_width)
    else:
        target_width = max(40, min(term_cols, 100))

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
                width=target_width,
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
                width=target_width,
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
            width=target_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(w.wrap(line))

    return "\n".join(wrapped_lines)


def colorize_terminal_text(text: str, theme: str = "dark") -> str:
    """Colorize inline markdown for terminal display based on theme."""
    if theme == "light":
        color_fence = "\033[31m"        # Red
        color_code = "\033[38;5;30m"    # Dark Cyan
        color_inline = "\033[38;5;26m"  # Dark Blue
    else:  # dark
        color_fence = "\033[91m"        # Bright Red
        color_code = "\033[36m"         # Cyan
        color_inline = "\033[38;5;75m"  # Light Blue

    reset = "\033[0m"
    bold = "\033[1m"

    lines = text.splitlines()
    res = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            res.append(f"{color_fence}{line}{reset}")
            continue
        if in_code:
            res.append(f"{color_code}{line}{reset}")
            continue

        # Color inline backticks `code` in light blue / dark blue
        line = re.sub(r"(`[^`\n]+`)", f"{color_inline}\\1{reset}", line)
        # Bold **text**
        line = re.sub(r"(\*\*[^*\n]+\*\*)", f"{bold}\\1{reset}", line)
        res.append(line)
    return "\n".join(res)


def format_output_links(
    text: str,
    mode: str = "auto",
    wrap: bool | None = None,
    width: int | None = None,
    color: bool | None = None,
    theme: str = "auto",
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
    detected_theme = detect_terminal_theme(theme)

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
        processed = colorize_terminal_text(processed, theme=detected_theme)

    link_color = "\033[38;5;28m" if detected_theme == "light" else "\033[32m"

    for label, url in extracted_links:
        if use_osc8 and should_color:
            replacement = f"{link_color}\033]8;;{url}\033\\{label}\033]8;;\033\\\033[0m"
        elif use_osc8:
            replacement = f"\033]8;;{url}\033\\{label}\033]8;;\033\\"
        elif should_color:
            replacement = f"{link_color}{label}\033[0m"
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
    from . import __version__
    parser = argparse.ArgumentParser(
        prog="cn",
        description="Generic Code & Documentation Navigation Engine",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program version and exit",
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
    p_search.add_argument(
        "--theme",
        choices=["auto", "dark", "light"],
        default="auto",
        help="Terminal color theme: auto (detect background), dark, light (default: auto)",
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
    p_ask.add_argument("--model", default=None, help="LLM model name (default: deepseek/deepseek-v4-flash-0731)")
    p_ask.add_argument("--endpoint", "--base-url", dest="endpoint", default=None, help="OpenAI-compatible LLM endpoint (default: https://openrouter.ai/api/v1)")
    p_ask.add_argument("--api-key", default=None, help="LLM API key")
    p_ask.add_argument("--system-prompt", default=None, help="Additional custom system prompt / persona instructions")
    p_ask.add_argument("--limit", type=int, default=None, help="Initial search results count (default: 10)")
    p_ask.add_argument("--max-searches", type=int, default=None, help="Max additional LLM-driven searches (default: 5)")
    p_ask.add_argument(
        "--links",
        choices=["auto", "markdown", "terminal", "osc8"],
        default="auto",
        help="Link formatting: auto (detect terminal), markdown, terminal (clean path:line), osc8 (default: auto)",
    )
    p_ask.add_argument(
        "--theme",
        choices=["auto", "dark", "light"],
        default="auto",
        help="Terminal color theme: auto (detect background), dark, light (default: auto)",
    )
    p_ask.add_argument("--wrap", action=argparse.BooleanOptionalAction, default=None, help="Wrap terminal lines (default: enabled on TTY)")
    p_ask.add_argument("--width", type=int, default=None, help="Wrap line width (default: terminal width)")
    p_ask.add_argument("--index-dir", default=None, help="Custom LanceDB directory")
    p_ask.add_argument("-n", "--new-session", action="store_true", help="Start a fresh conversation session with the daemon")
    p_ask.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # mcp
    p_mcp = subparsers.add_parser("mcp", help="Run Model Context Protocol (MCP) server over stdio")
    p_mcp.add_argument("folder", nargs="?", default=None, help="Initial default workspace repository root")
    p_mcp.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="MCP transport (default: stdio)")

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
            theme=args.theme,
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
            theme=args.theme,
            wrap=args.wrap,
            width=args.width,
            custom_index_dir=args.index_dir,
            quiet=args.quiet,
            new_session=args.new_session,
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

    from .ipc import discover_daemon_target, ping_target, query_target

    target = discover_daemon_target(folder, custom_index_dir)
    daemon_status = ping_target(target) if target else None
    if daemon_status:
        target_desc = f"socket: {target}" if isinstance(target, Path) else f"127.0.0.1:{target}"
        print(f"  🟢 cn watch daemon: ACTIVE ({target_desc})")
    else:
        socket_path = get_socket_path(folder, custom_index_dir)
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
    theme: str = "auto",
    wrap: bool | None = None,
    width: int | None = None,
    custom_index_dir: str | None = None,
):
    display_cfg = get_display_config(
        folder=folder,
        cli_overrides={
            "width": width,
            "theme": theme,
            "links": links,
            "wrap": wrap,
        },
    )
    resolved_width = display_cfg.get("width")
    resolved_theme = display_cfg.get("theme", "auto")
    resolved_links = display_cfg.get("links", "auto")
    resolved_wrap = display_cfg.get("wrap")

    from .ipc import discover_daemon_target, query_target
    target = discover_daemon_target(folder, custom_index_dir)
    results = query_target(target, query, limit=limit, doc_type=doc_type) if target else None
    if results is not None:
        raw_out = format_search_results(results, folder)
        print(
            format_output_links(
                raw_out,
                mode=resolved_links,
                wrap=resolved_wrap,
                width=resolved_width,
                theme=resolved_theme,
            )
        )
        return

    # Fallback to direct in-process search
    print(
        "💡 Tip: 'cn watch' is not running. Loading LanceDB embeddings in-process.\n"
        "   Run 'cn watch' in a separate terminal for instant sub-second searches!\n",
        file=sys.stderr,
    )
    from .index import VectorIndex
    idx = VectorIndex(folder, custom_index_dir)
    results = idx.search(query, limit=limit, doc_type=doc_type)
    raw_out = format_search_results(results, folder)
    print(
        format_output_links(
            raw_out,
            mode=resolved_links,
            wrap=resolved_wrap,
            width=resolved_width,
            theme=resolved_theme,
        )
    )


def _run_tags(folder: Path, symbol: str, exact: bool = False, limit: int = 20):
    mgr = TagsManager(folder)
    results = mgr.lookup_symbol(symbol, exact=exact, limit=limit)
    print(format_tag_results(results))


import itertools
import threading
import time


class StatusSpinner:
    """Lightweight animated terminal spinner for long-running steps on TTY."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str, stream=sys.stderr):
        self.message = message
        self.stream = stream
        self.is_tty = hasattr(stream, "isatty") and stream.isatty()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            # Calculate available width to prevent terminal line wrapping
            prefix = f"\r\033[36m{frame}\033[0m "
            try:
                term_width = shutil.get_terminal_size((80, 20)).columns
            except Exception:
                term_width = 80

            # 2 chars for frame icon + space
            avail = max(10, term_width - 3)
            msg = self.message
            if len(msg) > avail:
                msg = msg[: avail - 3].rstrip() + "..."

            self.stream.write(f"\r\033[2K{prefix}{msg}")
            self.stream.flush()
            time.sleep(0.08)

    def start(self):
        if self.is_tty:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            self.stream.write(f"{self.message}\n")
            self.stream.flush()

    def update_message(self, new_message: str):
        self.message = new_message
        if not self.is_tty:
            self.stream.write(f"{new_message}\n")
            self.stream.flush()

    def stop(self, final_line: str | None = None):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
        if self.is_tty:
            # Clear spinner line
            self.stream.write("\r\033[2K")
            if final_line:
                self.stream.write(f"{final_line}\n")
            self.stream.flush()


def _run_ask(
    folder: Path,
    question: str,
    model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    limit: int | None = None,
    max_searches: int | None = None,
    links: str = "auto",
    theme: str = "auto",
    wrap: bool | None = None,
    width: int | None = None,
    custom_index_dir: str | None = None,
    quiet: bool = False,
    new_session: bool = False,
):
    from .ask import ask_codebase, load_llm_config

    display_cfg = get_display_config(
        folder=folder,
        cli_overrides={
            "width": width,
            "theme": theme,
            "links": links,
            "wrap": wrap,
        },
    )
    resolved_width = display_cfg.get("width")
    resolved_theme = display_cfg.get("theme", "auto")
    resolved_links = display_cfg.get("links", "auto")
    resolved_wrap = display_cfg.get("wrap")

    config = load_llm_config(
        folder=folder,
        cli_overrides={
            "model": model,
            "endpoint": endpoint,
            "api_key": api_key,
            "system_prompt": system_prompt,
            "limit": limit,
            "max_searches": max_searches,
        },
    )

    try:
        spinner: StatusSpinner | None = None
        is_tty = sys.stderr.isatty() if hasattr(sys.stderr, "isatty") else False

        def handle_progress_cli(line: str):
            nonlocal spinner
            if quiet:
                return
            if not is_tty:
                print(line, file=sys.stderr, flush=True)
                return

            # Keep spinner active and updating dynamically in-place
            if spinner:
                spinner.update_message(line)
            else:
                spinner = StatusSpinner(line, stream=sys.stderr)
                spinner.start()

        try:
            # When TTY, we let handle_progress_cli manage the animated spinner output
            answer, stats = ask_codebase(
                folder=folder,
                question=question,
                config=config,
                custom_index_dir=custom_index_dir,
                verbose=False if is_tty else (not quiet),
                output_stream=sys.stderr,
                new_session=new_session,
                progress_callback=handle_progress_cli if not quiet else None,
            )
        finally:
            if spinner:
                spinner.stop()

        if not quiet and is_tty:
            ans_status = stats.get("status", "answered") if stats else "answered"
            if ans_status == "refusal":
                print("⚠️ Answer not found in codebase / Off-topic", file=sys.stderr)
            else:
                print("✅ Answer found by agent", file=sys.stderr)
        if not quiet:
            term_cols = shutil.get_terminal_size((80, 24)).columns
            if resolved_width:
                divider_width = min(resolved_width, term_cols) if term_cols > 0 else resolved_width
            else:
                divider_width = min(term_cols, 100)
            divider_width = max(20, divider_width)
            det_theme = detect_terminal_theme(resolved_theme)
            divider_color = "\033[34m" if det_theme == "light" else "\033[36m"
            print(f"\n{divider_color}{'=' * divider_width}\033[0m\n")

        print(
            format_output_links(
                answer,
                mode=resolved_links,
                wrap=resolved_wrap,
                width=resolved_width,
                theme=resolved_theme,
            )
        )

        if not quiet and stats and (stats.get("turn_total_tokens") or 0) > 0:
            turn_in = stats.get("turn_prompt_tokens", 0)
            turn_out = stats.get("turn_completion_tokens", 0)
            turn_total = stats.get("turn_total_tokens", 0)
            tool_count = stats.get("tool_calls_count", 0)

            tool_suffix = f" | {tool_count} tool call{'s' if tool_count != 1 else ''}" if tool_count > 0 else ""
            print(
                f"\n\033[2mTokens: {turn_total:,} (prompt: {turn_in:,}, completion: {turn_out:,}){tool_suffix}\033[0m",
                file=sys.stderr,
            )
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_mcp(folder: Path | None = None, transport: str = "stdio"):
    from .mcp_server import run_mcp_server, resolve_repository_root
    if folder:
        resolve_repository_root(str(folder))
    run_mcp_server(transport=transport)
