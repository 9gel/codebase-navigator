"""Git-aware ctags indexing and symbol lookup."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any

from .config import CODE_EXTENSIONS, DOC_EXTENSIONS, IGNORE_DIR_NAMES


def get_available_files(folder: Path) -> tuple[list[Path], list[Path]]:
    """Discover all Git-tracked and unignored source and documentation files."""
    code_files: list[Path] = []
    doc_files: list[Path] = []

    # 1. Try Git-based discovery
    try:
        res = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=folder,
            capture_output=True,
            text=True,
            check=True,
        )
        for line in res.stdout.splitlines():
            fpath = folder / line.strip()
            if fpath.is_file():
                ext = fpath.suffix.lower()
                if ext in CODE_EXTENSIONS:
                    code_files.append(fpath)
                elif ext in DOC_EXTENSIONS:
                    doc_files.append(fpath)
        return sorted(code_files), sorted(doc_files)
    except Exception:
        pass

    # 2. Fallback to filesystem walk
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIR_NAMES and not d.startswith(".")]
        for file in files:
            if file.startswith("."):
                continue
            fpath = Path(root) / file
            ext = fpath.suffix.lower()
            if ext in CODE_EXTENSIONS:
                code_files.append(fpath)
            elif ext in DOC_EXTENSIONS:
                doc_files.append(fpath)

    return sorted(code_files), sorted(doc_files)


class TagsManager:
    """Manages generation, updates, and symbol lookups from .tags files."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.tag_file = folder / ".tags"

    def generate(self) -> tuple[bool, str]:
        """Generate or regenerate .tags for all source files in the folder."""
        code_files, _ = get_available_files(self.folder)
        if not code_files:
            return False, "No source files found to index"

        rel_paths = []
        for p in code_files:
            try:
                rel_paths.append(str(p.relative_to(self.folder)))
            except ValueError:
                rel_paths.append(str(p))

        input_data = "\n".join(rel_paths) + "\n"
        cmd = [
            "ctags",
            "-L", "-",
            "-f", str(self.tag_file),
            "--fields=+n+K",
            "--sort=yes",
        ]

        try:
            res = subprocess.run(
                cmd,
                input=input_data,
                cwd=self.folder,
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                return False, f"ctags error ({res.returncode}): {res.stderr.strip()}"
            if self.tag_file.exists():
                size_mb = self.tag_file.stat().st_size / (1024 * 1024)
                return True, f"Indexed {len(code_files)} source files ({size_mb:.2f} MB)"
            return False, "Tag file was not created"
        except FileNotFoundError:
            return False, "ctags binary not found on PATH"
        except Exception as e:
            return False, f"Error generating tags: {e}"

    def find_tag_file(self) -> Path | None:
        """Find .tags in folder or climb parent directories."""
        curr = self.folder
        for parent in [curr, *curr.parents]:
            tf = parent / ".tags"
            if tf.exists():
                return tf
        return self.tag_file if self.tag_file.exists() else None

    def lookup_symbol(
        self,
        pattern: str,
        exact: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Look up symbols matching a regex or exact string."""
        tag_files: list[Path] = []
        tf = self.find_tag_file()
        if tf:
            tag_files.append(tf)

        # Also search child folders' .tags
        for p in self.folder.rglob(".tags"):
            if p not in tag_files:
                tag_files.append(p)

        if not tag_files:
            return []

        results: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, int]] = set()

        for tag_file in tag_files:
            try:
                regex = re.compile(pattern if not exact else f"^{re.escape(pattern)}$", re.IGNORECASE)
                with open(tag_file, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if line.startswith("!_"):
                            continue
                        parts = line.split("\t")
                        if not parts:
                            continue
                        sym = parts[0]
                        if regex.search(sym):
                            parsed = self._parse_tag_line(line, tag_file.parent)
                            if parsed:
                                key = (parsed["symbol"], parsed["path"], parsed["line"])
                                if key not in seen_keys:
                                    seen_keys.add(key)
                                    results.append(parsed)
                                    if len(results) >= limit:
                                        return results
            except Exception:
                pass

        return results

    def _parse_tag_line(self, line: str, base_dir: Path) -> dict[str, Any] | None:
        if not line or line.startswith("!_"):
            return None
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 3:
            return None
        sym = parts[0]
        fpath = parts[1]
        pattern_or_line = parts[2]

        kind = "symbol"
        line_no = 1
        for field in parts[3:]:
            if field.startswith("line:"):
                try:
                    line_no = int(field[5:])
                except ValueError:
                    pass
            elif field.startswith("kind:"):
                kind = field[5:]
            elif len(field) == 1:
                kind = field

        abs_p = (base_dir / fpath).resolve()
        try:
            rel_p = str(abs_p.relative_to(self.folder))
        except ValueError:
            rel_p = fpath

        return {
            "symbol": sym,
            "path": rel_p,
            "abs_path": str(abs_p),
            "line": line_no,
            "kind": kind,
            "preview": pattern_or_line.strip("/^$"),
        }
