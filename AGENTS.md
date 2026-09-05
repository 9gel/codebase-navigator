# AGENTS.md — Agent Operating Guidelines for codebase-navigator

This document provides mandatory project instructions, standards, and operational rules for AI agents and coding harnesses working on `codebase-navigator`.

---

## 1. Mandatory Version Bump Rule

> [!IMPORTANT]
> **Every push or set of changes MUST bump the package build version.**
> Automated CI checks (`.github/workflows/check-version-bump.yml`) enforce that `src/codebase_navigator/__init__.py` (`__version__`) is incremented on every commit/PR. `pyproject.toml` uses dynamic versioning via Hatchling (`[tool.hatch.version] path = "src/codebase_navigator/__init__.py"`).

When making changes:
1. **Single Source of Truth**: Increment `__version__` in [`src/codebase_navigator/__init__.py`](file:///home/nigel/code/codebase-navigator/src/codebase_navigator/__init__.py).
2. **Bump the patch segment by default.** It is an integer, not a digit: `0.8.9` → `0.8.10` → `0.8.11` is correct and expected. There is no reason to roll the minor over just because the patch reached 9.
3. **Reserve the minor for changes worth announcing** — a new default, a changed tool contract, anything requiring a re-index or invalidating existing indexes. Ordinary fixes, refactors, test additions and documentation are patch bumps however many there are in a row.

> [!NOTE]
> This project moves in many small measured steps, so patch numbers climb quickly. That is the intended shape: a long patch series within one minor tells the reader those changes were incremental and compatible, which is information a rolled-over minor would destroy.

---

## 2. Architecture & Subsystems

Before modifying code, review [`DESIGN.md`](file:///home/nigel/code/codebase-navigator/DESIGN.md) for full architectural context.

- **`src/codebase_navigator/ask.py`**:
  - LLM agent reasoning loop and function-calling specifications.
  - Multi-turn `AgentSession` tracking turn tokens and session lifetime tokens.
  - Initial 10-result seed search injection in initial user turn.
  - Configurable system prompts (`LLMConfig.system_prompt`).
- **`src/codebase_navigator/tools.py`**:
  - Code intelligence tools: `search`, `read_code`, `tags_lookup`, `grep_search`, `find_references`, `call_tree`.
  - Always implement pure-Python fallbacks with loud warnings when external binaries (e.g. `rg`) are absent.
- **`src/codebase_navigator/watcher.py` & `src/codebase_navigator/ipc.py`**:
  - Filesystem watcher (`watchfiles`) maintaining live `.tags` and LanceDB vector embeddings.
  - Unix domain socket server (`.codebase-navigator/watch.sock`) running NDJSON protocol.
  - In-memory `AgentSession` hosting enabling provider KV prompt caching across `cn ask` invocations.
  - Client-server version handshake rejecting mismatched protocol clients.
- **`src/codebase_navigator/extractor.py` & `src/codebase_navigator/index.py`**:
  - Hybrid LanceDB vector indexing with phrase/title boosting.
  - AST docstring parsing for Python and regex markdown/comment chunking.
- **`src/codebase_navigator/cli.py`**:
  - Unified CLI argument parser (`cn status`, `cn sync`, `cn watch`, `cn search`, `cn tags`, `cn ask`).
  - Terminal hyperlink formatting (`OSC 8`, clean terminal path, markdown links) and theme detection.

---

## 3. Development & Testing Environment

- **Nix & Direnv**:
  - Package dependencies and virtual environment are managed via Nix flakes and `direnv`.
  - Run all commands inside the dev environment using `direnv exec . <command>` or `uv run <command>`.
- **Running Tests**:
  ```bash
  uv run pytest -v
  ```
  - Always run the test suite to verify changes before committing.
  - When adding new features or fixing bugs, add corresponding regression tests in `tests/`.

---

## 4. Code Style & Commit Conventions

- **Python Requirements**: Python >= 3.11 with `from __future__ import annotations`.
- **Formatting & Linting**: Line length 100, checked with `ruff`.
- **Git Commits**:
  - Commit logically grouped changes with clear, descriptive commit messages.
  - Do not push to remote without explicit user consent.
