---
name: dev-devel-tools
description: Efficiently navigate codebases and project documentation using dt tools (dt search, dt tags, dt status, dt sync, dt watch). Use when locating concepts, architecture docs, symbol definitions, or maintaining live semantic indexes across development sessions.
---

# devel-tools Navigation Guide

`devel-tools` (`dt`) provides high-speed code and documentation discovery for agents and developers. It complements `ripgrep` (`rg`) by providing semantic concept search and indexed ctags symbol navigation.

## Tool Suite Overview

| Command | Purpose | When to Use |
|---|---|---|
| `dt search` | Semantic vector search | Finding concepts, architectural docs, domain rules, and code docstrings |
| `dt tags` | Universal Ctags symbol lookup | Finding exact class, function, struct, or variable definitions |
| `dt status` | Index and daemon health | Checking indexed file counts and verifying if `dt watch` is active |
| `dt sync` | Full index synchronization | Forcing an immediate refresh of `.tags` and LanceDB vector embeddings |
| `dt watch` | Live background watcher & IPC daemon | Keeping indexes live and serving in-memory queries over Unix domain socket |

---

## 1. Finding Concepts & Documentation (`dt search`)

Use `dt search` when you know *what* you want to achieve conceptually, but do not know the exact variable or file names:

```bash
# General conceptual search
dt search "content-addressed publisher bytes" .

# Filter for documentation only (.md, .rst, etc.)
dt search "data migration ledger" . --type md

# Filter for code comments and docstrings only
dt search "lot geometry transformation" . --type code

# Limit result count (default: 5)
dt search "authentication tokens" . --limit 3
```

Results provide direct Markdown links with line ranges:
```markdown
### 1. [src/sitingdata_pipeline/sources/README.md:L1-L40](file:///.../sources/README.md#L1-L40) — Publisher source mechanics (Match: 88%)
```

---

## 2. Locating Symbols & Definitions (`dt tags`)

Use `dt tags` for fast symbol lookups across large codebases without scanning file trees:

```bash
# Substring or regex symbol lookup
dt tags calculate_metrics .

# Exact symbol lookup
dt tags Defs . --exact

# Limit symbol results
dt tags parse_ . --limit 10
```

Results output the symbol type, path, and definition snippet:
```text
1. `defs` (variable) -> [src/sitingdata_pipeline/definitions.py:L13](file:///.../definitions.py#L13)
   `defs = dg.Definitions(`
```

---

## 3. Tool Selection: When to Use What

- **Use `dt search`** for domain logic questions, API specifications, workflow descriptions, and module purposes (e.g., *"where is the cache expiration logic?"*).
- **Use `dt tags`** when you know the identifier name (e.g., `BarrierReferenceStore`, `transform_records`) and need its declaration location.
- **Use `ripgrep` (`rg`)** for exact string occurrences, import statements, or regular expressions across code lines.

---

## 4. Index Lifecycle & Daemon Management

### Project Directory Layout
Indexes and runtime sockets are stored in the project-local `.devel-tools/` directory:
- `.devel-tools/lancedb` — Vector database
- `.devel-tools/files_meta.json` — Incremental mtime/size cache
- `.devel-tools/watch.sock` — Unix Domain Socket for fast IPC

### Checking Daemon & Index Status (`dt status`)
Always check the health and daemon status before starting work in a repository:

```bash
dt status .
```

Example output:
```text
📊 Navigation Status for: /path/to/project
  Available files: 924 source code files, 48 doc files
  🏷️  Tags file: /path/to/project/.tags (1.77 MB)
  🟢 dt watch daemon: ACTIVE (socket: /path/to/project/.devel-tools/watch.sock)
  🧠 Vector index: /path/to/project/.devel-tools
     Indexed files: 952, Total chunks: 7771
```

### Running the Live Daemon (`dt watch`)
A coordinating agent or background task can launch `dt watch` in the repository root:

```bash
dt watch .
```

- **Socket Acceleration**: When `dt watch` is running, `dt search` queries the in-memory index via `.devel-tools/watch.sock` in **< 30ms**, skipping Python ML library loading.
- **Concurrency Safety**: If multiple agents run `dt watch` under the same project directory, subsequent runs will detect the active socket and safely exit without conflicting.
- **Crash Recovery**: If `dt watch` was killed unexpectedly, stale socket files are automatically detected, unlinked, and recovered on subsequent commands.

### Manual Synchronization (`dt sync`)
If `dt watch` is not running and you pulled major changes or switched branches:

```bash
# Incremental sync (only changed files)
dt sync .

# Complete re-indexing from scratch
dt sync . --force
```
