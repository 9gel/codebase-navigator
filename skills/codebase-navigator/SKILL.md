
name: codebase-navigator
description: Efficiently navigate codebases and project documentation using cn tools (cn search, cn tags, cn status, cn sync, cn watch). Use when locating concepts, architecture docs, symbol definitions, or maintaining live semantic indexes across development sessions.
---

# Codebase Navigatory Guide

`codebase-navigator` (`cn`) provides high-speed code and documentation discovery for agents and developers. It complements `ripgrep` (`rg`) by providing semantic concept search and indexed ctags symbol navigation.

## Tool Suite Overview

| Command | Purpose | When to Use |
|---|---|---|
| `cn search` | Semantic vector search | Finding concepts, architectural docs, domain rules, and code docstrings |
| `cn tags` | Universal Ctags symbol lookup | Finding exact class, function, struct, or variable definitions |
| `cn status` | Index and daemon health | Checking indexed file counts and verifying if `ac watch` is active |
| `cn sync` | Full index synchronization | Forcing an immediate refresh of `.tags` and LanceDB vector embeddings |
| `cn watch` | Live background watcher & IPC daemon | Keeping indexes live and serving in-memory queries over Unix domain socket |

---

## 1. Finding Concepts & Documentation (`cn search`)

Use `cn search` when you know *what* you want to achieve conceptually, but do not know the exact variable or file names:

```bash
# General conceptual search
cn search "content-addressed publisher bytes" .

# Filter for documentation only (.md, .rst, etc.)
cn search "data migration ledger" . --type md

# Filter for code comments and docstrings only
cn search "lot geometry transformation" . --type code

# Limit result count (default: 5)
cn search "authentication tokens" . --limit 3
```

Results provide direct Markdown links with line ranges:
```markdown
### 1. [src/sitingdata_pipeline/sources/README.md:L1-L40](file:///.../sources/README.md#L1-L40) — Publisher source mechanics (Match: 88%)
```

---

## 2. Locating Symbols & Definitions (`cn tags`)

Use `cn tags` for fast symbol lookups across large codebases without scanning file trees:

```bash
# Substring or regex symbol lookup
cn tags calculate_metrics .

# Exact symbol lookup
cn tags Defs . --excn

# Limit symbol results
cn tags parse_ . --limit 10
```

Results output the symbol type, path, and definition snippet:
```text
1. `defs` (variable) -> [src/sitingdata_pipeline/definitions.py:L13](file:///.../definitions.py#L13)
   `defs = dg.Definitions(`
```

---

## 3. Tool Selection: When to Use What

- **Use `cn search`** for domain logic questions, API specifications, workflow descriptions, and module purposes (e.g., *"where is the cache expiration logic?"*).
- **Use `cn tags`** when you know the identifier name (e.g., `BarrierReferenceStore`, `transform_records`) and need its declaration location.
- **Use `ripgrep` (`rg`)** for exact string occurrences, import statements, or regular expressions across code lines.

---

## 4. Index Lifecycle & Daemon Management

### Project Directory Layout
Indexes and runtime sockets are stored in the project-local `.cn/` directory:
- `.cn/lancedb` — Vector database
- `.cn/files_meta.json` — Incremental mtime/size cache
- `.cn/watch.sock` — Unix Domain Socket for fast IPC

### Checking Daemon & Index Status (`cn status`)
Always check the health and daemon status before starting work in a repository:

```bash
cn status .
```

Example output:
```text
📊 Navigation Status for: /path/to/project
  Available files: 924 source code files, 48 doc files
  🏷️  Tags file: /path/to/project/.tags (1.77 MB)
  🟢 cn watch daemon: ACTIVE (socket: /path/to/project/.cn/watch.sock)
  🧠 Vector index: /path/to/project/.cn
     Indexed files: 952, Total chunks: 7771
```

### Running the Live Daemon (`cn watch`)
A coordinating agent or background task can launch `cn watch` in the repository root:

```bash
cn watch .
```

- **Socket Acceleration**: When `cn watch` is running, `cn search` queries the in-memory index via `.cn/watch.sock` in **< 30ms**, skipping Python ML library loading.
- **Concurrency Safety**: If multiple agents run `cn watch` under the same project directory, subsequent runs will detect the active socket and safely exit without conflicting.
- **Crash Recovery**: If `cn watch` was killed unexpectedly, stale socket files are automatically detected, unlinked, and recovered on subsequent commands.

### Manual Synchronization (`cn sync`)
If `cn watch` is not running and you pulled major changes or switched branches:

```bash
# Incremental sync (only changed files)
cn sync .

# Complete re-indexing from scratch
cn sync . --force
```
