"""Semantic content extraction from Markdown files and source code comments/docstrings."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
from typing import Any


class DocExtractor:
    """Extracts semantic chunks from Markdown documentation and code docstrings/comments."""

    def __init__(self, base_folder: Path):
        self.base_folder = base_folder

    def extract_markdown(self, path: Path) -> list[dict[str, Any]]:
        """Extract markdown sections and term definitions preserving headers and line ranges."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        try:
            rel_path = str(path.relative_to(self.base_folder))
        except ValueError:
            rel_path = str(path)

        lines = text.splitlines()
        chunks: list[dict[str, Any]] = []

        header_re = re.compile(r"^(#{1,6})\s+(.+)$")
        term_re = re.compile(r"^(?:-\s+)?(?:\*\*([^*]+)\*\*|__([^_]+)__)\s*[-—:]\s*(.+)$")

        current_headers: list[str] = []
        chunk_start = 1
        chunk_lines: list[str] = []

        def flush_chunk(end_line: int):
            nonlocal chunk_start, chunk_lines
            content = "\n".join(chunk_lines).strip()
            if len(content) >= 20:
                title = " > ".join(current_headers) if current_headers else f"{path.name} (top)"
                chunk_id = hashlib.sha256(f"{rel_path}:{chunk_start}:{title}".encode("utf-8")).hexdigest()[:16]
                chunks.append({
                    "id": chunk_id,
                    "path": rel_path,
                    "abs_path": str(path.resolve()),
                    "doc_type": "markdown",
                    "title": title,
                    "start_line": chunk_start,
                    "end_line": end_line,
                    "content": f"# {title}\n\n{content}",
                })
            chunk_lines = []

        # 1. Section-level chunking
        for idx, line in enumerate(lines, start=1):
            match = header_re.match(line)
            if match:
                level = len(match.group(1))
                heading_text = match.group(2).strip()

                if chunk_lines:
                    flush_chunk(idx - 1)

                while len(current_headers) >= level:
                    current_headers.pop()
                current_headers.append(heading_text)
                chunk_start = idx
                chunk_lines.append(line)
            else:
                chunk_lines.append(line)

        if chunk_lines:
            flush_chunk(len(lines))

        # 2. Granular term/definition chunking (for glossaries, bullet rules, specifications)
        t_start = 0
        t_name = ""
        t_lines: list[str] = []
        for idx, line in enumerate(lines, start=1):
            s = line.strip()
            match = term_re.match(s)
            if match:
                if t_lines and t_name and len("\n".join(t_lines)) >= 30:
                    c_id = hashlib.sha256(f"{rel_path}:{t_start}:{t_name}".encode("utf-8")).hexdigest()[:16]
                    chunks.append({
                        "id": c_id,
                        "path": rel_path,
                        "abs_path": str(path.resolve()),
                        "doc_type": "markdown",
                        "title": f"{path.name} > {t_name}",
                        "start_line": t_start,
                        "end_line": idx - 1,
                        "content": f"# {path.name} > {t_name}\n\n" + "\n".join(t_lines),
                    })
                t_name = match.group(1) or match.group(2)
                t_start = idx
                t_lines = [line]
            elif t_lines:
                if s == "" and len(t_lines) >= 2 and not any(
                    lines[min(len(lines) - 1, idx)].strip().startswith(p) for p in ["-", "*", ">"]
                ):
                    if len("\n".join(t_lines)) >= 30:
                        c_id = hashlib.sha256(f"{rel_path}:{t_start}:{t_name}".encode("utf-8")).hexdigest()[:16]
                        chunks.append({
                            "id": c_id,
                            "path": rel_path,
                            "abs_path": str(path.resolve()),
                            "doc_type": "markdown",
                            "title": f"{path.name} > {t_name}",
                            "start_line": t_start,
                            "end_line": idx,
                            "content": f"# {path.name} > {t_name}\n\n" + "\n".join(t_lines),
                        })
                    t_name = ""
                    t_lines = []
                else:
                    t_lines.append(line)

        if t_lines and t_name and len("\n".join(t_lines)) >= 30:
            c_id = hashlib.sha256(f"{rel_path}:{t_start}:{t_name}".encode("utf-8")).hexdigest()[:16]
            chunks.append({
                "id": c_id,
                "path": rel_path,
                "abs_path": str(path.resolve()),
                "doc_type": "markdown",
                "title": f"{path.name} > {t_name}",
                "start_line": t_start,
                "end_line": len(lines),
                "content": f"# {path.name} > {t_name}\n\n" + "\n".join(t_lines),
            })

        return chunks

    def extract_code_doc(self, path: Path) -> list[dict[str, Any]]:
        """Extract docstrings and comment blocks from source files."""
        if path.suffix == ".py":
            return self._extract_python(path)
        return self._extract_generic_comments(path)

    def _extract_python(self, path: Path) -> list[dict[str, Any]]:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        try:
            rel_path = str(path.relative_to(self.base_folder))
        except ValueError:
            rel_path = str(path)

        chunks: list[dict[str, Any]] = []
        try:
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, ValueError):
            return self._extract_generic_comments(path)

        lines = source.splitlines()

        # Module docstring
        module_doc = ast.get_docstring(tree)
        if module_doc and len(module_doc.strip()) > 10:
            doc_lines = len(module_doc.splitlines())
            chunk_id = hashlib.sha256(f"{rel_path}:1:module".encode("utf-8")).hexdigest()[:16]
            chunks.append({
                "id": chunk_id,
                "path": rel_path,
                "abs_path": str(path.resolve()),
                "doc_type": "code_doc",
                "title": f"{path.name} (module docstring)",
                "start_line": 1,
                "end_line": min(len(lines), doc_lines + 5),
                "content": f"Module {rel_path}:\n{module_doc.strip()}",
            })

        # Functions & Classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                start_l = getattr(node, "lineno", 1)
                end_l = getattr(node, "end_lineno", start_l + 10)
                def_line = lines[start_l - 1].strip() if start_l <= len(lines) else ""

                decorators = []
                for dec in getattr(node, "decorator_list", []):
                    d_line = getattr(dec, "lineno", None)
                    if d_line and d_line <= len(lines):
                        decorators.append(lines[d_line - 1].strip())

                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                node_name = node.name

                text_parts = []
                if decorators:
                    text_parts.append("\n".join(decorators))
                text_parts.append(def_line)
                if doc:
                    text_parts.append(doc.strip())

                text = "\n".join(text_parts).strip()
                if len(text) > 30:
                    chunk_id = hashlib.sha256(f"{rel_path}:{start_l}:{node_name}".encode("utf-8")).hexdigest()[:16]
                    chunks.append({
                        "id": chunk_id,
                        "path": rel_path,
                        "abs_path": str(path.resolve()),
                        "doc_type": "code_doc",
                        "title": f"{path.name} > {node_name} ({kind})",
                        "start_line": start_l,
                        "end_line": end_l,
                        "content": f"{rel_path} ({kind} {node_name}):\n{text}",
                    })

        return chunks

    def _extract_generic_comments(self, path: Path) -> list[dict[str, Any]]:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        try:
            rel_path = str(path.relative_to(self.base_folder))
        except ValueError:
            rel_path = str(path)

        lines = source.splitlines()
        chunks: list[dict[str, Any]] = []

        comment_block: list[str] = []
        block_start = 1

        for idx, line in enumerate(lines, start=1):
            s = line.strip()
            is_comment = False
            comment_content = ""

            if s.startswith(("//", "#", "--", ";", "/*", "*")):
                is_comment = True
                comment_content = re.sub(r"^(\/\/|#|--|;|\/\*|\*)\s*", "", s)

            if is_comment and comment_content:
                if not comment_block:
                    block_start = idx
                comment_block.append(comment_content)
            else:
                if len(comment_block) >= 3:
                    text = "\n".join(comment_block).strip()
                    if len(text) > 40:
                        chunk_id = hashlib.sha256(f"{rel_path}:{block_start}:comment".encode("utf-8")).hexdigest()[:16]
                        chunks.append({
                            "id": chunk_id,
                            "path": rel_path,
                            "abs_path": str(path.resolve()),
                            "doc_type": "code_doc",
                            "title": f"{path.name}: comment (L{block_start}-{idx-1})",
                            "start_line": block_start,
                            "end_line": idx - 1,
                            "content": f"{rel_path} (L{block_start}-{idx-1}):\n{text}",
                        })
                comment_block = []

        if len(comment_block) >= 3:
            text = "\n".join(comment_block).strip()
            if len(text) > 40:
                chunk_id = hashlib.sha256(f"{rel_path}:{block_start}:comment".encode("utf-8")).hexdigest()[:16]
                chunks.append({
                    "id": chunk_id,
                    "path": rel_path,
                    "abs_path": str(path.resolve()),
                    "doc_type": "code_doc",
                    "title": f"{path.name}: comment (L{block_start}-{len(lines)})",
                    "start_line": block_start,
                    "end_line": len(lines),
                    "content": f"{rel_path} (L{block_start}-{len(lines)}):\n{text}",
                })

        return chunks
