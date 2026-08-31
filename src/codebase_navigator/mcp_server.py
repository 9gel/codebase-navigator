"""Model Context Protocol (MCP) Server for codebase-navigator."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import unquote, urlparse

from mcp.server.mcpserver import MCPServer, Context

from . import __version__
from .config import get_socket_path
from .index import VectorIndex
from .ipc import ping_socket, query_socket
from .tags import TagsManager
from .tools import (
    find_references as tool_find_references,
    get_call_tree as tool_get_call_tree,
    grep_search as tool_grep_search,
    read_code as tool_read_code,
)

mcp = MCPServer("codebase-navigator", version=__version__)

# In-memory LRU repository cache
_REPO_LOCK = threading.Lock()
_INDEX_CACHE: dict[str, VectorIndex] = {}
_LAST_DISCOVERED_ROOT: Path | None = None


def _clean_path_str(p_str: str) -> str:
    """Normalize file:// URI or filesystem path string."""
    if p_str.startswith("file://"):
        parsed = urlparse(p_str)
        return unquote(parsed.path)
    return p_str


def resolve_repository_root(
    repo_root: str | None = None,
    ctx: Context | None = None,
) -> Path:
    """Resolve target repository directory using hierarchy:
    
    1. Explicit `repo_root` parameter passed in tool call.
    2. Client declared workspace root from MCP handshake (`roots/list`).
    3. Last successfully used / auto-discovered repository root.
    4. Current working directory (`Path.cwd()`).
    """
    global _LAST_DISCOVERED_ROOT

    # 1. Explicit tool parameter
    if repo_root and repo_root.strip():
        cleaned = _clean_path_str(repo_root.strip())
        target = Path(cleaned).expanduser().resolve()
        if target.is_dir():
            _LAST_DISCOVERED_ROOT = target
            return target

    # 2. Check client workspace roots via MCP Context if available
    # Note: FastMCP provides access to client session roots
    if ctx and hasattr(ctx, "session") and ctx.session:
        try:
            # Check roots from client session
            client_roots = getattr(ctx.session, "roots", None)
            if client_roots:
                for r in client_roots:
                    r_uri = str(getattr(r, "uri", ""))
                    cleaned_uri = _clean_path_str(r_uri)
                    candidate = Path(cleaned_uri).resolve()
                    if candidate.is_dir():
                        _LAST_DISCOVERED_ROOT = candidate
                        return candidate
        except Exception:
            pass

    # 3. Last discovered / cached root
    if _LAST_DISCOVERED_ROOT and _LAST_DISCOVERED_ROOT.is_dir():
        return _LAST_DISCOVERED_ROOT

    # 4. Fallback to current working directory
    cwd = Path.cwd().resolve()
    _LAST_DISCOVERED_ROOT = cwd
    return cwd


def get_or_create_repo_index(folder: Path, auto_sync: bool = True) -> VectorIndex:
    """Get hot VectorIndex for repository, auto-indexing if not already present."""
    with _REPO_LOCK:
        f_str = str(folder)
        if f_str in _INDEX_CACHE:
            return _INDEX_CACHE[f_str]

        idx = VectorIndex(folder)
        # Check if index exists or needs initial sync
        lancedb_dir = idx.cache_dir / "lancedb"
        tags_file = folder / ".tags"

        if auto_sync and (not lancedb_dir.exists() or not tags_file.exists()):
            try:
                tags_mgr = TagsManager(folder)
                if not tags_file.exists():
                    tags_mgr.generate()
                idx.sync()
            except Exception:
                pass

        _INDEX_CACHE[f_str] = idx
        return idx


@mcp.tool()
def codebase_search(
    query: str,
    doc_type: str = "all",
    limit: int = 5,
    repo_root: str | None = None,
    ctx: Context = None,
) -> str:
    """Perform hybrid semantic vector & keyword search across codebase documentation, comments, and docstrings.

    Args:
        query: Natural language or keyword search query.
        doc_type: Document type filter ('all', 'markdown', 'code_doc').
        limit: Maximum number of search results (default: 5).
        repo_root: Optional absolute or relative path to target repository root.
    """
    folder = resolve_repository_root(repo_root, ctx)

    # 1. Check if live cn watch daemon is active for this folder (via socket or TCP port)
    from .ipc import discover_daemon_target, query_target
    target = discover_daemon_target(folder)
    if target is not None:
        results = query_target(target, query, limit=limit, doc_type=doc_type)
        if results is not None:
            return _format_search_chunks(results, folder)

    # 2. In-process vector index search
    idx = get_or_create_repo_index(folder, auto_sync=True)
    results = idx.search(query, limit=limit, doc_type=doc_type)
    return _format_search_chunks(results, folder)


@mcp.tool()
def codebase_read(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    repo_root: str | None = None,
    ctx: Context = None,
) -> str:
    """Inspect exact source code lines in a file with line numbers and clickable file URIs.

    Args:
        path: File path (relative to repo root or absolute).
        start_line: 1-indexed starting line number (optional, default: 1).
        end_line: 1-indexed ending line number (optional, max 500 lines).
        repo_root: Optional path to target repository root.
    """
    folder = resolve_repository_root(repo_root, ctx)
    res = tool_read_code(folder, path, start_line=start_line, end_line=end_line)
    if "error" in res:
        return f"Error: {res['error']}"
    return res.get("content", "")


@mcp.tool()
def codebase_tags(
    symbol: str,
    exact: bool = False,
    limit: int = 15,
    repo_root: str | None = None,
    ctx: Context = None,
) -> str:
    """Quickly look up exact or regex symbol definitions (classes, functions, methods) in .tags.

    Args:
        symbol: Symbol name or regex pattern.
        exact: Match exact symbol name (default: false).
        limit: Maximum results (default: 15).
        repo_root: Optional path to target repository root.
    """
    folder = resolve_repository_root(repo_root, ctx)
    tags_mgr = TagsManager(folder)
    matches = tags_mgr.lookup_symbol(symbol, exact=exact, limit=limit)
    if not matches:
        return f"No symbol tags found matching '{symbol}' in {folder.name}."

    lines = []
    for m in matches:
        lines.append(
            f"- Symbol: `{m['symbol']}` ({m.get('kind', 'symbol')}) -> [{m['path']}:{m['line']}](file://{m['abs_path']}#L{m['line']})\n"
            f"  Preview: `{m.get('preview', '')}`"
        )
    return "\n".join(lines)


@mcp.tool()
def codebase_references(
    symbol: str,
    path_filter: str | None = None,
    limit: int = 15,
    repo_root: str | None = None,
    ctx: Context = None,
) -> str:
    """1-shot hybrid tool: locate symbol definitions and all caller/usage sites across the repository.

    Args:
        symbol: Function, class, or method name to find usages for.
        path_filter: Optional file glob filter (e.g. '*.py' or 'src/*').
        limit: Max call/usage sites to return (default: 15).
        repo_root: Optional path to target repository root.
    """
    folder = resolve_repository_root(repo_root, ctx)
    refs = tool_find_references(folder, symbol, path_filter=path_filter, limit=limit)
    if not refs:
        return f"No definitions or references found for '{symbol}' in {folder.name}."

    lines = []
    for r in refs:
        t = r.get("type", "reference")
        if t == "definition":
            lines.append(f"📌 Definition: [{r['path']}:{r['line']}](file://{r['abs_path']}#L{r['line']}) ({r.get('kind', 'symbol')}) - `{r.get('preview', '')}`")
        else:
            lines.append(f"🔍 Usage/Caller: [{r['path']}:{r['line']}](file://{r['abs_path']}#L{r['line']}) - `{r.get('context', '')}`")
    return "\n".join(lines)


@mcp.tool()
def codebase_call_tree(
    symbol: str,
    path: str | None = None,
    repo_root: str | None = None,
    ctx: Context = None,
) -> str:
    """Trace incoming callers and outgoing callees for a function or class using AST and cross-file references.

    Args:
        symbol: Function or class name to trace call hierarchy for.
        path: Optional file path where the symbol is defined.
        repo_root: Optional path to target repository root.
    """
    folder = resolve_repository_root(repo_root, ctx)
    tree = tool_get_call_tree(folder, symbol, path=path)
    out = [f"Call Tree for `{symbol}` in {folder.name}:"]

    if tree.get("definitions"):
        out.append("Definitions:")
        for d in tree["definitions"]:
            out.append(f"  - [{d['path']}:{d['line']}](file://{d['abs_path']}#L{d['line']})")
    if tree.get("callers"):
        out.append("Callers (Functions/Files that invoke this symbol):")
        for c in tree["callers"]:
            fn_ctx = f" (in `{c.get('caller_function')}`)" if c.get("caller_function") else ""
            out.append(f"  - [{c['path']}:{c.get('call_line', 1)}](file://{c['abs_path']}#L{c.get('call_line', 1)}){fn_ctx}: `{c.get('preview', '')}`")
    if tree.get("callees"):
        out.append("Callees (Functions invoked by this symbol):")
        for c in tree["callees"]:
            out.append(f"  - Calls `{c.get('symbol')}` at [{c['path']}:{c['line']}](file://{c['abs_path']}#L{c['line']})")

    if not tree.get("callers") and not tree.get("callees") and not tree.get("definitions"):
        return f"No call tree data found for '{symbol}' in {folder.name}."
    return "\n".join(out)


@mcp.tool()
def codebase_grep(
    pattern: str,
    path_glob: str | None = None,
    case_sensitive: bool = False,
    limit: int = 25,
    repo_root: str | None = None,
    ctx: Context = None,
) -> str:
    """Search for literal text or regex patterns across files using ripgrep with pure-Python fallback.

    Args:
        pattern: Regex or keyword string to search for.
        path_glob: Optional glob filter (e.g. '*.rs', '*.go', 'src/**').
        case_sensitive: Whether match is case sensitive (default: false).
        limit: Max matches to return (default: 25).
        repo_root: Optional path to target repository root.
    """
    folder = resolve_repository_root(repo_root, ctx)
    matches = tool_grep_search(folder, pattern, path_glob=path_glob, case_sensitive=case_sensitive, limit=limit)
    if not matches:
        return f"No pattern matches found for '{pattern}' in {folder.name}."

    lines = []
    for m in matches:
        lines.append(f"- [{m['path']}:{m['line']}](file://{m['abs_path']}#L{m['line']}): `{m['content']}`")
    return "\n".join(lines)


def _format_search_chunks(results: list[dict[str, Any]], folder: Path) -> str:
    """Format search results cleanly for MCP agent consumption."""
    if not results:
        return f"No relevant code or documentation chunks found in {folder.name}."

    chunks_text = []
    for idx, r in enumerate(results, start=1):
        rel_p = r.get("path", "")
        abs_p = r.get("abs_path", "")
        s_line = r.get("start_line", 1)
        e_line = r.get("end_line", 1)
        title = r.get("title", "")
        score_pct = int(r.get("score", 0.0) * 100)
        content = r.get("content", "")

        header = f"[{idx}] {rel_p}:{s_line}-{e_line} — {title} (Score: {score_pct}%)\nURI: file://{abs_p}#L{s_line}-L{e_line}"
        body = f"```\n{content}\n```"
        chunks_text.append(f"{header}\n{body}")

    return "\n\n".join(chunks_text)


def run_mcp_server(transport: str = "stdio"):
    """Run the codebase-navigator MCP server."""
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Unsupported transport: {transport}")


if __name__ == "__main__":
    run_mcp_server()
