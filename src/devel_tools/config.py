"""Configuration, schema definitions, and shared utilities for devel-tools."""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from typing import Iterator

# Set offline environment variables before importing ML libraries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pyarrow as pa

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384

DOC_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("path", pa.string()),
    pa.field("abs_path", pa.string()),
    pa.field("doc_type", pa.string()),  # "markdown" or "code_doc"
    pa.field("title", pa.string()),
    pa.field("start_line", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("content", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
])

CODE_EXTENSIONS = {
    ".py", ".rs", ".go", ".ts", ".tsx", ".js", ".jsx",
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh",
    ".nix", ".sh", ".bash", ".zsh", ".sql", ".java",
    ".kt", ".scala", ".rb", ".php", ".cs", ".swift",
    ".lua", ".zig", ".nim", ".elm", ".ex", ".exs",
    ".erl", ".hrl", ".hs", ".ml", ".mli", ".pl", ".pm",
    ".r", ".jl", ".clj", ".cljs", ".lisp", ".scm",
}

DOC_EXTENSIONS = {
    ".md", ".markdown", ".rst", ".adoc", ".org",
}

IGNORE_DIR_NAMES = {
    ".git", ".hg", ".svn",
    "node_modules", "target", "build", "dist",
    ".venv", "venv", "env", ".direnv",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".cache", ".devel-index", ".devel-tools", ".dagster_home", "pipeline-cache",
}


@contextlib.contextmanager
def silence_stdio() -> Iterator[None]:
    """Silence low-level stdout/stderr (e.g. from C-level libraries and model loaders)."""
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_stdout = os.dup(1)
        save_stderr = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        try:
            yield
        finally:
            os.dup2(save_stdout, 1)
            os.dup2(save_stderr, 2)
            os.close(null_fd)
            os.close(save_stdout)
            os.close(save_stderr)
    except Exception:
        yield


def get_cache_dir(folder: Path, custom_index_dir: str | None = None) -> Path:
    """Determine the persistence directory for vector indexes and tools metadata."""
    if custom_index_dir:
        cdir = Path(custom_index_dir).resolve()
        cdir.mkdir(parents=True, exist_ok=True)
        return cdir

    target = folder / ".devel-tools"
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_socket_path(folder: Path, custom_index_dir: str | None = None) -> Path:
    """Return the Unix Domain Socket path used for IPC with devel-watch."""
    return get_cache_dir(folder, custom_index_dir) / "watch.sock"
