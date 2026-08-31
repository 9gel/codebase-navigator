---
name: codebase-navigator
description: Efficiently navigate codebases and project documentation using codebase-navigator CLI (cn) or MCP server tools (codebase_search, codebase_read, codebase_tags, codebase_references, codebase_call_tree, codebase_grep). Use when locating concepts, architecture docs, symbol definitions, tracing callers/callees, or reading code ranges without bloating context tokens.
---

# Codebase Navigator Guide & MCP Reference

`codebase-navigator` provides high-speed code intelligence, semantic documentation retrieval, and indexed symbol navigation for AI agents and developers. It saves 40%–80% in token overhead compared to naive full-file ingestion or blind ripgreps.

---

## 1. Using via MCP Server (For AI Agents & Coding Harnesses)

When connected to the `codebase-navigator` MCP server, agents have access to 6 specialized code intelligence tools:

| MCP Tool | Purpose | When to Call |
|---|---|---|
| `codebase_search` | Hybrid vector & keyword search | Find domain concepts, architecture docs, design rules, or docstrings. |
| `codebase_tags` | Symbol definition lookup | Find declaration locations for functions, classes, structs, or methods via `.tags`. |
| `codebase_references` | 1-shot definitions & callers | Find where a symbol is defined and all places it is called or referenced. |
| `codebase_call_tree` | Caller & Callee hierarchy | Trace what calls a function and what functions that function calls. |
| `codebase_read` | Bounded line range reader | Read specific lines of a file (e.g. lines 40–90) with clickable `file://` links. |
| `codebase_grep` | Exact regex / literal match | Search exact string patterns with ripgrep speed and Python fallback. |

### Recommended Agent Investigation Strategy

1. **Locate Concepts & Docs First (`codebase_search`)**:
   - Don't guess file names. Query `codebase_search("auth token validation")` to find relevant markdown files and code modules.
2. **Find Identifiers & Usages (`codebase_tags` / `codebase_references`)**:
   - If you have an identifier (e.g. `process_chunk`), call `codebase_references("process_chunk")` to get definition and caller sites in a single turn.
3. **Trace Execution Flow (`codebase_call_tree`)**:
   - Trace caller and callee hierarchies when understanding multi-step pipelines.
4. **Read Exact Slices (`codebase_read`)**:
   - Read only the relevant line ranges (e.g. `codebase_read("src/auth.py", start_line=50, end_line=110)`) rather than entire files.

---

## 2. Using via CLI (`cn`)

For humans or command-line workflows:

| CLI Command | Purpose |
|---|---|
| `cn ask "<question>"` | Autonomous LLM agent harness with iterative search and multi-turn KV session caching. |
| `cn search "<query>"` | Semantic & keyword search across markdown documentation and code comments. |
| `cn tags <symbol>` | Fast symbol definition lookup in `.tags`. |
| `cn status` | Check index status and active `cn watch` daemon socket. |
| `cn sync [--force]` | Synchronize `.tags` and LanceDB vector embeddings. |
| `cn watch` | Live file watcher and background socket daemon. |
| `cn mcp` | Launch the Model Context Protocol (MCP) server over stdio. |

---

## 3. Index Lifecycle & Daemon

- **Live Daemon (`cn watch`)**: Keeps `.tags` and LanceDB embeddings up to date on file saves. When active, `cn` queries run over Unix domain socket in **< 10ms**.
- **Self-Healing Indexing**: If a repository has not been indexed, `codebase-navigator` auto-generates `.tags` (~0.1s) and LanceDB FastEmbed ONNX vectors (~1–2s) on the first query.
