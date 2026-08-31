"""Configuration, schema definitions, and shared utilities for codebase-navigator."""

from __future__ import annotations

import contextlib
import os
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Ensure portable user caching & offline environment variables before importing ML libraries
_cache_base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
if "FASTEMBED_CACHE_DIR" not in os.environ:
    os.environ["FASTEMBED_CACHE_DIR"] = str(_cache_base / "fastembed")
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_cache_base / "huggingface")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pyarrow as pa

EMBEDDING_MODEL_NAME = os.environ.get(
    "CN_EMBEDDING_MODEL",
    os.environ.get("CODEBASE_NAVIGATOR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)

# Common dimensions for known FastEmbed ONNX models, default 384 for MiniLM
_MODEL_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}
VECTOR_DIM = _MODEL_DIMS.get(EMBEDDING_MODEL_NAME, 384)

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
    ".cache", ".devel-index", ".devel-tools", ".codebase-navigator", ".dagster_home", "pipeline-cache",
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
    except Exception:  # noqa: BLE001
        yield


def parse_toml_file(p: Path) -> dict[str, Any]:
    """Safely parse a TOML file if it exists."""
    if p.is_file():
        try:
            with open(p, "rb") as f:
                return tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            return {}
    return {}


def get_display_config(
    folder: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hierarchical resolution for display options (width, max_width, theme, links, wrap):
    CLI > ENV > project config.toml > global ~/.config/codebase-navigator/config.toml > defaults.
    """
    cli = {k: v for k, v in (cli_overrides or {}).items() if v is not None}

    # 1. Global user config
    home = Path.home()
    user_candidates = [
        home / ".config" / "codebase-navigator" / "config.toml",
        home / ".config" / "codebase-navigator.toml",
    ]
    user_data: dict[str, Any] = {}
    for uc in user_candidates:
        if uc.is_file():
            user_data = parse_toml_file(uc)
            break

    # 2. Project local config
    project_data: dict[str, Any] = {}
    if folder:
        project_candidates = [
            folder / ".codebase-navigator" / "config.toml",
            folder / "codebase-navigator.toml",
            folder / ".codebase-navigator.toml",
        ]
        for pc in project_candidates:
            if pc.is_file():
                project_data = parse_toml_file(pc)
                break

    # Merge config layers
    merged: dict[str, Any] = {}
    for src in [user_data, project_data]:
        display_sec = src.get("display", {}) if isinstance(src.get("display"), dict) else {}
        for k in ("width", "max_width", "theme", "links", "wrap"):
            val = display_sec.get(k) if k in display_sec else src.get(k)
            if val is not None:
                if k in ("width", "max_width"):
                    try:
                        merged["width"] = int(val)
                    except ValueError:
                        pass
                else:
                    merged[k] = val

    # Environment variable overrides
    env_width = os.environ.get("CN_WIDTH") or os.environ.get("CN_MAX_WIDTH")
    if env_width:
        try:
            merged["width"] = int(env_width)
        except ValueError:
            pass

    if "CN_THEME" in os.environ:
        merged["theme"] = os.environ["CN_THEME"]
    if "CN_LINKS" in os.environ:
        merged["links"] = os.environ["CN_LINKS"]
    if "CN_WRAP" in os.environ:
        v = os.environ["CN_WRAP"].lower()
        merged["wrap"] = v in ("1", "true", "yes")

    # CLI overrides
    if "width" in cli:
        merged["width"] = cli["width"]
    if "theme" in cli and cli["theme"] != "auto":
        merged["theme"] = cli["theme"]
    elif "theme" not in merged:
        merged["theme"] = cli.get("theme", "auto")

    if "links" in cli and cli["links"] != "auto":
        merged["links"] = cli["links"]
    elif "links" not in merged:
        merged["links"] = cli.get("links", "auto")

    if "wrap" in cli and cli["wrap"] is not None:
        merged["wrap"] = cli["wrap"]

    return merged


def get_cache_dir(folder: Path, custom_index_dir: str | None = None) -> Path:
    """Determine the persistence directory for vector indexes and tools metadata."""
    if custom_index_dir:
        cdir = Path(custom_index_dir).resolve()
        cdir.mkdir(parents=True, exist_ok=True)
        return cdir

    target = folder / ".codebase-navigator"
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_socket_path(folder: Path, custom_index_dir: str | None = None) -> Path:
    """Return the Unix Domain Socket path used for IPC with cn watch."""
    return get_cache_dir(folder, custom_index_dir) / "watch.sock"


def get_port_path(folder: Path, custom_index_dir: str | None = None) -> Path:
    """Return the TCP port metadata file path used for IPC with cn watch."""
    return get_cache_dir(folder, custom_index_dir) / "watch.port"


def get_default_tcp_port(folder: Path) -> int:
    """Compute a deterministic loopback TCP port in range 10000..59999 based on directory path hash."""
    import zlib
    canonical_path = str(folder.resolve()).encode("utf-8")
    hash_val = zlib.crc32(canonical_path)
    return 10000 + (hash_val % 50000)
