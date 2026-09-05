"""Agent tools for codebase inspection, navigation, and reference tracing."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import CODE_EXTENSIONS, DOC_EXTENSIONS, IGNORE_DIR_NAMES
from .tags import TagsManager

_WARNED_MISSING_RIPGREP = False


# A single matched line can be enormous in generated/minified sources (one line in
# vikunja is 486k chars). Cap per-match content so one grep cannot flood the context.
MAX_MATCH_CHARS = 240


def _truncate_match(text: str, limit: int = MAX_MATCH_CHARS) -> str:
    """Clamp a single grep match line so pathological long lines cannot blow up context."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [+{len(text) - limit} chars truncated]"


def check_ripgrep_installed(verbose: bool = True, output_stream=sys.stderr) -> bool:
    """Check if ripgrep (rg) is installed on PATH. Warn loudly if missing."""
    global _WARNED_MISSING_RIPGREP
    rg_path = shutil.which("rg")
    if rg_path is not None:
        return True

    if verbose and not _WARNED_MISSING_RIPGREP:
        _WARNED_MISSING_RIPGREP = True
        print(
            "\n"
            "⚠️  [WARNING] 'rg' (ripgrep) binary was not found on PATH!\n"
            "   Falling back to Python file traversal. Code searches and reference tracing will be significantly slower on large repositories.\n"
            "   👉 Install ripgrep (e.g. 'nix-shell -p ripgrep', 'apt install ripgrep', or 'brew install ripgrep') for optimal performance.\n",
            file=output_stream,
        )
    return False


def read_code(
    folder: Path,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_lines: int = 2000,
) -> dict[str, Any]:
    """Read contents of a file with optional line bounds.

    Returns formatted code with line numbers and absolute file URI.
    """
    fpath = (folder / path).resolve()
    try:
        # Check boundary
        fpath.relative_to(folder.resolve())
    except ValueError:
        return {"error": f"Path '{path}' is outside the repository root."}

    if not fpath.is_file():
        return {"error": f"File not found: {path}"}

    try:
        text = fpath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": f"Error reading file '{path}': {e}"}

    lines = text.splitlines()
    total_lines = len(lines)

    s = max(1, start_line or 1)
    e = min(total_lines, end_line or (s + max_lines - 1))

    if s > total_lines:
        return {
            "path": path,
            "abs_path": str(fpath),
            "total_lines": total_lines,
            "content": f"[File has {total_lines} lines; requested start line {s} is beyond EOF]",
        }

    if (e - s + 1) > max_lines:
        e = s + max_lines - 1

    selected_lines = lines[s - 1 : e]
    numbered = [f"{i:5d} | {line}" for i, line in enumerate(selected_lines, start=s)]
    formatted_code = "\n".join(numbered)

    rel_p = str(fpath.relative_to(folder.resolve()))
    abs_uri = f"file://{fpath}#L{s}-L{e}"  # returned for callers, not inlined below

    return {
        "path": rel_p,
        "abs_path": str(fpath),
        "start_line": s,
        "end_line": e,
        "total_lines": total_lines,
        "uri": abs_uri,
        "content": f"File: {rel_p}:{s}-{e}\n```\n{formatted_code}\n```",
    }


def read_code_ranges(
    folder: Path,
    ranges: list[dict[str, Any]],
    max_lines: int = 2000,
) -> str:
    """Read several line ranges in one call and concatenate the formatted results.

    Half of all read_code calls in the benchmark re-opened a file the agent had
    already read, each costing a full round trip (system prompt + tool spec +
    the whole growing history) to return a few dozen lines. Batching lets one
    turn cover every range the agent needs.
    """
    if not ranges:
        return "Error: no ranges supplied."

    budget = max_lines
    parts: list[str] = []
    for spec in ranges:
        if budget <= 0:
            parts.append("[Line budget exhausted; request the remaining ranges in a new call.]")
            break
        path = spec.get("path", "")
        if not path:
            parts.append("Error: a range is missing 'path'.")
            continue
        start_line = spec.get("start_line")
        end_line = spec.get("end_line")
        res = read_code(
            folder,
            path,
            start_line=start_line,
            end_line=end_line,
            max_lines=budget,
        )
        if "error" in res:
            parts.append(f"Error reading {path}: {res['error']}")
            continue
        consumed = (res.get("end_line", 1) - res.get("start_line", 1)) + 1
        budget -= max(0, consumed)
        parts.append(res.get("content", ""))

    return "\n\n".join(p for p in parts if p)


def grep_search(
    folder: Path,
    pattern: str,
    path_glob: str | None = None,
    case_sensitive: bool = False,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Search for regex or keyword pattern across codebase using ripgrep or pure-Python fallback."""
    has_rg = check_ripgrep_installed()

    if has_rg:
        cmd = ["rg", "--json", "-m", str(limit)]
        if not case_sensitive:
            cmd.append("-i")
        if path_glob:
            cmd.extend(["-g", path_glob])
        cmd.extend(["-e", pattern, "."])

        try:
            res = subprocess.run(
                cmd,
                cwd=folder,
                capture_output=True,
                text=True,
                check=False,
            )
            matches: list[dict[str, Any]] = []
            import json

            for line in res.stdout.splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        match_data = data.get("data", {})
                        rel_path = match_data.get("path", {}).get("text", "")
                        line_no = match_data.get("line_number", 1)
                        line_text = match_data.get("lines", {}).get("text", "").rstrip("\r\n")
                        line_text = _truncate_match(line_text)
                        abs_p = (folder / rel_path).resolve()

                        matches.append(
                            {
                                "path": rel_path,
                                "abs_path": str(abs_p),
                                "line": line_no,
                                "content": line_text,
                                "uri": f"file://{abs_p}#L{line_no}",
                            }
                        )
                        if len(matches) >= limit:
                            break
                except Exception:
                    continue
            return matches
        except Exception:
            pass  # Fall through to python search

    # Python Fallback
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error:
        regex = re.compile(re.escape(pattern), flags)

    matches = []
    glob_regex = None
    if path_glob:
        import fnmatch

        glob_regex = re.compile(fnmatch.translate(path_glob))

    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES and not d.startswith(".")]
        for file in files:
            if file.startswith("."):
                continue
            fpath = Path(root) / file
            rel_p = str(fpath.relative_to(folder))

            if glob_regex and not glob_regex.match(rel_p) and not glob_regex.match(file):
                continue

            ext = fpath.suffix.lower()
            if ext not in CODE_EXTENSIONS and ext not in DOC_EXTENSIONS:
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for idx, line in enumerate(f, start=1):
                        if regex.search(line):
                            abs_p = fpath.resolve()
                            matches.append(
                                {
                                    "path": rel_p,
                                    "abs_path": str(abs_p),
                                    "line": idx,
                                    "content": _truncate_match(line),
                                    "uri": f"file://{abs_p}#L{idx}",
                                }
                            )
                            if len(matches) >= limit:
                                return matches
            except Exception:
                continue

    return matches


def find_references(
    folder: Path,
    symbol: str,
    path_filter: str | None = None,
    limit: int = 20,
    tag_file: Path | None = None,
) -> list[dict[str, Any]]:
    """1-shot hybrid tool to find definitions and all call/usage sites of a symbol.

    Returns structured definitions from .tags / AST along with caller lines across the repo.
    """
    sym = symbol.strip()
    if not sym:
        return []

    # 1. Look up definition(s)
    tags_mgr = TagsManager(folder, tag_file=tag_file)
    defs = tags_mgr.lookup_symbol(sym, exact=True, limit=5)
    def_locations = {(d["path"], d["line"]) for d in defs}

    # 2. Grep for symbol call / usage sites (using word boundary pattern)
    pattern = rf"\b{re.escape(sym)}\b"
    raw_matches = grep_search(
        folder,
        pattern,
        path_glob=path_filter,
        case_sensitive=True,
        limit=limit + len(def_locations),
    )

    results: list[dict[str, Any]] = []

    # Add definitions first
    for d in defs:
        results.append(
            {
                "type": "definition",
                "symbol": sym,
                "path": d["path"],
                "abs_path": d["abs_path"],
                "line": d["line"],
                "kind": d.get("kind", "symbol"),
                "preview": d.get("preview", ""),
                "uri": f"file://{d['abs_path']}#L{d['line']}",
            }
        )

    # Add usage / call sites (excluding the definition lines)
    for m in raw_matches:
        if (m["path"], m["line"]) in def_locations:
            continue
        results.append(
            {
                "type": "reference",
                "symbol": sym,
                "path": m["path"],
                "abs_path": m["abs_path"],
                "line": m["line"],
                "context": m["content"],
                "uri": m["uri"],
            }
        )
        if len(results) >= limit:
            break

    return results


def analyze_python_calls(fpath: Path, target_symbol: str) -> dict[str, Any]:
    """Inspect a Python file using AST to find functions calling or called by target_symbol."""
    try:
        source = fpath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(fpath))
    except Exception:
        return {}

    lines = source.splitlines()
    callees: list[dict[str, Any]] = []
    callers: list[dict[str, Any]] = []

    # Check if target_symbol is defined in this file and extract what it calls
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == target_symbol:
                # Find all calls inside this function (callees)
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        fn_name = None
                        if isinstance(sub.func, ast.Name):
                            fn_name = sub.func.id
                        elif isinstance(sub.func, ast.Attribute):
                            fn_name = sub.func.attr
                        if fn_name and fn_name != target_symbol:
                            line_no = getattr(sub, "lineno", node.lineno)
                            line_preview = (
                                _truncate_match(lines[line_no - 1]) if line_no <= len(lines) else ""
                            )
                            callees.append(
                                {
                                    "symbol": fn_name,
                                    "line": line_no,
                                    "preview": line_preview,
                                }
                            )
            else:
                # Check if this function calls target_symbol (callers)
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        fn_name = None
                        if isinstance(sub.func, ast.Name):
                            fn_name = sub.func.id
                        elif isinstance(sub.func, ast.Attribute):
                            fn_name = sub.func.attr
                        if fn_name == target_symbol:
                            line_no = getattr(sub, "lineno", node.lineno)
                            line_preview = (
                                _truncate_match(lines[line_no - 1]) if line_no <= len(lines) else ""
                            )
                            callers.append(
                                {
                                    "caller_function": node.name,
                                    "caller_line": node.lineno,
                                    "call_line": line_no,
                                    "preview": line_preview,
                                }
                            )

    return {"callees": callees, "callers": callers}


def get_call_tree(
    folder: Path,
    symbol: str,
    path: str | None = None,
    limit: int = 15,
    tag_file: Path | None = None,
) -> dict[str, Any]:
    """Trace caller and callee relationships for a given function or class."""
    sym = symbol.strip()
    tags_mgr = TagsManager(folder, tag_file=tag_file)
    defs = tags_mgr.lookup_symbol(sym, exact=True, limit=5)

    result: dict[str, Any] = {
        "symbol": sym,
        "definitions": defs,
        "callees": [],
        "callers": [],
    }

    # If symbol is in a Python file, run AST call analysis
    py_files_to_check: set[Path] = set()
    if path and path.endswith(".py"):
        py_files_to_check.add((folder / path).resolve())
    for d in defs:
        if d["path"].endswith(".py"):
            py_files_to_check.add(Path(d["abs_path"]))

    for py_file in py_files_to_check:
        if py_file.is_file():
            ast_data = analyze_python_calls(py_file, sym)
            for c in ast_data.get("callees", []):
                c["path"] = str(py_file.relative_to(folder))
                c["abs_path"] = str(py_file)
                c["uri"] = f"file://{py_file}#L{c['line']}"
                result["callees"].append(c)
            for c in ast_data.get("callers", []):
                c["path"] = str(py_file.relative_to(folder))
                c["abs_path"] = str(py_file)
                c["uri"] = f"file://{py_file}#L{c['call_line']}"
                result["callers"].append(c)

    # General reference search for external callers across the repo
    external_refs = find_references(folder, sym, limit=limit, tag_file=tag_file)
    for ref in external_refs:
        if ref.get("type") == "reference":
            # Add to callers if not already present
            loc = (ref["path"], ref["line"])
            if not any((c["path"], c.get("call_line")) == loc for c in result["callers"]):
                result["callers"].append(
                    {
                        "path": ref["path"],
                        "abs_path": ref["abs_path"],
                        "call_line": ref["line"],
                        "preview": ref.get("context", ""),
                        "uri": ref["uri"],
                    }
                )

    result["callees"] = result["callees"][:limit]
    result["callers"] = result["callers"][:limit]
    return result
