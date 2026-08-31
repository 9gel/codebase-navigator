# codebase-navigator

Tools for ultra-fast semantic codebase navigation for humans and AI agents,
Git-aware ctags indexing, live watchers, and
[LanceDB](https://github.com/lancedb/lancedb) semantic search. Self-contained
runtime, no additional servers (e.g. Ollama) to run for embeddings.

Cut 40%-80% token use by having your LLM search precisely, instead of ingesting
entire code bases or searching incorrectly using blind ripgreps.

Get oriented to a codebase quickly by asking natural language questions, instead
of wasting time switching between files and losing your train of thought.

## How it works

Launch the `cn watch` daemon in one terminal pane, and use it in another.

| Pane 1 | Pane 2 |
| -------- | -------- |
| <pre><code>❯ cn watch<br/> 🚀 Starting cn watch for: /home/user/code/project<br/>   Performing initial sync...<br/>   .tags: Indexed 924 source files (1.78 MB)<br/>   LanceDB: 954 files updated (7814 chunks), 0 pruned.<br/>   Index location: /home/user/code/project/.codebase-navigator<br/>   🔌 IPC Socket: /home/user/code/project/.codebase-navigator/watch.sock<br/>   🧠 Agent Session Daemon: Ready (KV prompt caching enabled)<br/> 👀 Watching for file changes (Ctrl+C to stop)...<br/> <br/> [18:15:08] ⚡ Synced 1 file(s) (4 chunks) in 784ms<br/> [18:54:52] ⚡ Synced 1 file(s) (4 chunks) in 710ms<br/> [18:56:19] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [18:56:19] ⚡ Synced 134 file(s) (1505 chunks) in 111142ms</code></pre> | <pre><code>❯ cn ask "How does request dispatching work?"<br/> 🔍 Searching codebase for: "How does request dispatching work?"...<br/> ✓ Found 10 relevant code/doc chunks.<br/> 🔎 [Tool 1/15: find_references] symbol='dispatch_request'...<br/> 🔎 [Tool 2/15: read_code] path='src/app.py', start_line=45, end_line=90...<br/> <br/> ================================================================================<br/> Request dispatching in this codebase is orchestrated in [src/app.py:45-90](file:///home/user/code/project/src/app.py#L45-L90)...</code></pre> |

## Features

- 💬 **Autonomous Agent Harness (`cn ask`)**: Ask architectural and
  implementation questions in natural language. Powered by an iterative LLM
  reasoning loop with 1-shot hybrid code intelligence tools:
  - `search`: Semantic and hybrid search over code comments and markdown docs.
  - `tags_lookup`: Instant symbol definition resolution via `.tags`.
  - `read_code`: Range-bounded source inspection with line numbers and clickable file links.
  - `find_references`: 1-shot hybrid symbol definitions + all call and usage sites.
  - `call_tree`: AST & cross-file caller and callee tracer.
  - `grep_search`: Fast pattern search via `rg` (with pure-Python fallback).
- ⚡ **Multi-Turn Session Memory & KV Prompt Caching**: When running `cn watch`,
  conversational context is preserved in-memory across successive `cn ask` commands.
  Follow-up questions hit provider-side prefix KV caches for instant responses and lower token cost.
- 🧠 **LanceDB Semantic & Hybrid Search**: Vector search powered by
  `FastEmbed ONNX `all-MiniLM-L6-v2`` with hybrid phrase/title match
  boosting for markdown documentation, glossary terms, and code comments.
- 🏷️ **Git-Aware `.tags` Generation**: Uses `universal-ctags` to index genuine
  source code while ignoring huge data dumps, JSON caches, `.git`, `node_modules`,
  and build artifacts.
- 👀 **Live File Watcher**: Automatically re-indexes `.tags` and incrementally
  updates LanceDB embeddings on every save with sub-second debounce.
- 🔗 **Clickable GitHub Markdown Links**: Returns results formatted as
  `[file:Lstart-Lend](file:///abs_path#Lstart-Lend)`.
- ⚙️ **Configurable System Prompts**: Customize agent persona, auditing constraints,
  or architectural instructions via CLI flags, environment variables, or TOML config.
- ⚡ **Strict Offline Mode**: Can run 100% locally from disk, with zero
  HuggingFace network requests.

## Quick Start

### Embedding Model & First-Time Download

`codebase-navigator` uses **FastEmbed** with ONNX runtime for ultra-fast vector embeddings:
- **Default Model**: `FastEmbed ONNX `all-MiniLM-L6-v2`` (384-dimensional vector embeddings).
- **First Run**: On the very first invocation of `cn sync`, `cn watch`, or `cn search`, the ~80MB ONNX model weights are automatically downloaded once to your local machine.
- **Cache Location**: Stored in `~/.cache/fastembed/` (or `$FASTEMBED_CACHE_DIR`).
- **Customizing Cache Directory**:
  ```bash
  export FASTEMBED_CACHE_DIR="/path/to/custom/cache/fastembed"
  ```
- **100% Offline After Download**: Once downloaded, `cn` operates strictly from disk with zero external network calls for embeddings.

### Prerequisites

Ensure these command-line tools are installed on your system:

- **[Git](https://git-scm.com/downloads)**: Used for repository discovery and
  ignoring non-tracked files.
- **[universal-ctags](https://github.com/universal-ctags/ctags#installation)**:
  Required for generating `.tags` code symbol indexes.
- **[ripgrep](https://github.com/BurntSushi/ripgrep)** (optional but recommended):
  Provides blazing-fast pattern searching and reference tracing. If missing, `cn`
  falls back to pure-Python traversal.

### Try it out using uvx

You can run `cn` using `uvx` (the tool runner from [uv](https://docs.astral.sh/uv/)). At the top level of a code tree under a git repository, run:

```bash
# Start the live watcher in one terminal:
uvx codebase-navigator watch

# Ask about the code in another terminal:
uvx codebase-navigator ask "What functions call process_data?"

# Ask a follow-up question (KV cache enabled):
uvx codebase-navigator ask "Show me the unit test for that function"

# Start a fresh conversation session:
uvx codebase-navigator ask "How is authentication handled?" --new-session

# Tag search for exact symbols:
uvx codebase-navigator tags flush_chunk

# Search documentation and comments:
uvx codebase-navigator search "Flush Chunk"
```

## CLI Commands

The unified `cn` command provides all indexing and search tools:

| Command | Purpose |
|---|---|
| `cn ask <question> [folder]` | LLM-powered codebase Q&A with iterative multi-tool reasoning and session memory |
| `cn search <query> [folder]` | Semantic & hybrid search in markdown docs and code comments |
| `cn tags <symbol> [folder]` | Fast symbol definition lookup in `.tags` |
| `cn sync [folder] [--force]` | Synchronize `.tags` and LanceDB vector embeddings |
| `cn watch [folder]` | Live filesystem watcher, socket server, and agent session host |
| `cn status [folder]` | Inspect index and `.tags` status |

### `cn ask` Options

- `-n, --new-session`: Start a fresh conversation session with the daemon.
- `--system-prompt "<text>"`: Append custom system instructions or persona (e.g. security auditing).
- `--model "<model>"`: LLM model name (default: `google/gemini-2.5-flash`).
- `--endpoint "<url>"`: OpenAI-compatible LLM endpoint (default: `https://openrouter.ai/api/v1`).
- `--api-key "<key>"`: LLM API key.
- `--limit <N>`: Initial pre-flight search results count (default: 10).
- `--max-searches <N>`: Max additional tool calls allowed (default: 15).
- `-q, --quiet`: Suppress progress indicators.

## Configuration

`codebase-navigator` supports hierarchical configuration for LLM queries (`cn ask`) and display preferences.

### Configuration File Locations

Configuration files are parsed in TOML format from:
- **Project-level**: `.codebase-navigator/config.toml` (or `codebase-navigator.toml` in repository root)
- **User-level**: `~/.config/codebase-navigator/config.toml` (or `~/.config/codebase-navigator.toml`)

### Example `config.toml`

```toml
[llm]
# OpenAI-compatible API endpoint (defaults to OpenRouter)
endpoint = "https://openrouter.ai/api/v1"

# LLM model to query
model = "google/gemini-2.5-flash"

# API authentication token (or pass via environment variable)
api_key = "sk-or-v1-..."

# Maximum follow-up tool calls the LLM can execute (default: 15)
max_searches = 15

# Initial number of semantic search chunks provided to the LLM (default: 10)
limit = 10

# Optional custom system prompt / persona
system_prompt = "You are an expert security and performance auditor."

[display]
# Maximum width for terminal text wrapping and dividers (e.g. 80, 100, or terminal width)
width = 80

# Terminal color theme: "auto" (queries terminal background / OSC 11), "dark", "light"
theme = "auto"

# Link formatting: "auto" (detects OSC 8 & TTY), "osc8", "terminal" (clean path:line), "markdown"
links = "auto"

# Enable/disable line wrapping on TTY output
wrap = true
```

### Environment Variables

Environment variables take precedence over config files:

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` / `CN_API_KEY` / `OPENAI_API_KEY` | API Key for LLM completions | `None` |
| `CN_ENDPOINT` / `CN_BASE_URL` / `OPENROUTER_BASE_URL` | OpenAI-compatible endpoint | `https://openrouter.ai/api/v1` |
| `CN_MODEL` / `OPENROUTER_MODEL` | Default LLM model | `google/gemini-2.5-flash` |
| `CN_SYSTEM_PROMPT` | Additional custom system prompt | `None` |
| `CN_MAX_SEARCHES` | Max tool calls allowed by the LLM | `15` |
| `CN_ASK_LIMIT` | Initial search result count | `10` |
| `CN_WIDTH` / `CN_MAX_WIDTH` | Maximum terminal wrap width | `terminal width or 100` |
| `CN_THEME` | Terminal theme (`auto`, `dark`, `light`) | `auto` |
| `CN_LINKS` | Link mode (`auto`, `osc8`, `terminal`, `markdown`) | `auto` |
| `CN_WRAP` | Line wrapping (`true`, `false`) | `true (on TTY)` |

### Precedence Order

When resolving settings, `cn` applies the following order of precedence:
1. **CLI flags** (e.g. `--api-key`, `--model`, `--system-prompt`, `--width`, `--theme`, `--links`, `--wrap`)
2. **Environment variables** (`OPENROUTER_API_KEY`, `CN_SYSTEM_PROMPT`, `CN_WIDTH`, etc.)
3. **Project config** (`.codebase-navigator/config.toml`)
4. **User config** (`~/.config/codebase-navigator/config.toml`)
5. **Built-in defaults**

## Development

The project uses Nix and `direnv`. Ensure you have both installed, then:

```bash
cd codebase-navigator/
direnv allow
uv run pytest
```

## References

* [How RAG Can Cut Your AI Coding Costs by 80%](https://blog.mornati.net/how-rag-can-cut-your-ai-coding-costs-by-80)
