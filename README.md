# codebase-navigator

Ultra-fast semantic codebase navigation for humans and AI agents, Git-aware
ctags indexing, live watchers, and [LanceDB](https://github.com/lancedb/lancedb)
semantic search. Self-contained runtime, no additional servers (e.g. Ollama) to
run for embeddings.

Cut 40%-80% token use by having your LLM search precisely, instead of ingesting
entire code bases or searching incorrectly using blind ripgreps.

For the human: get oriented to a codebase quickly by asking natural language
questions, instead of wasting time switching between files and losing your train
of thought.

## How it works

### As a standalone coding aid

Launch the `cn watch` daemon in one terminal pane:

```
❯ cd yourcode
❯ cn watch
🚀 Starting cn watch for: /home/user/yourcode
  Performing initial sync...
  .tags: Indexed 931 source files (1.80 MB)
  LanceDB: 21 files updated (407 chunks), 0 pruned.
  Index location: /home/user/yourcode/.codebase-navigator
  🔌 IPC Socket: /home/user/yourcode/.codebase-navigator/watch.sock
  🧠 Agent Session Daemon: Ready (KV prompt caching enabled)
👀 Watching for file changes (Ctrl+C to stop)...

[00:29:03] 🏷️  Tags updated: Indexed 931 source files (1.80 MB)
[00:29:03] ⚡ Synced 1 file(s) (31 chunks) in 927ms
[00:29:04] 🏷️  Tags updated: Indexed 931 source files (1.80 MB)
[00:29:04] ⚡ Synced 1 file(s) (16 chunks) in 649ms
[00:30:56] 🏷️  Tags updated: Indexed 931 source files (1.80 MB)
[00:30:56] ⚡ Synced 1 file(s) (8 chunks) in 448ms
[00:30:56] 🏷️  Tags updated: Indexed 931 source files (1.80 MB)
[00:30:56] ⚡ Synced 1 file(s) (6 chunks) in 416ms
...
```

Run `cn ask` in a separate terminal:

```
❯ cn ask "Where is the separator ============= produced? Give me a one-line answer"
✅ Answer found by agent

================================================================================

The `=============` separator is produced in src/codebase_navigator/cli.py:770
via `print(f"\n{divider_color}{'=' * divider_width}\033[0m\n")` in the `cn ask`
output path.

Tokens: 12,456 (prompt: 12,005, completion: 451)

❯ cn ask "What's the weather like today? Give a one-line answer"
⚠️ Answer not found in codebase / Off-topic

================================================================================

I can't answer that — I'm a codebase navigation assistant scoped to this
repository, and weather data isn't part of this codebase.

Tokens: 8,420 (prompt: 8,358, completion: 62)
```

### As an MCP or cli tool for your agent

Use codebase-navigator as a lightweight embeddings server for your favorite
coding harness, either via the MCP, or give the cli tools to your agent so they
find code quickly and token-efficiently.


## Features

- 💬 **Autonomous Agent Harness (`cn ask`)**: Ask architectural and
  implementation questions in natural language. Powered by an iterative LLM
  reasoning loop with 1-shot hybrid code intelligence tools.
- 🧠 **LanceDB Semantic & Hybrid Search (`cn search`)**: Vector search powered
  by `FastEmbed ONNX `all-MiniLM-L6-v2`` with hybrid phrase/title match boosting
  for markdown documentation, glossary terms, and code comments.
- 🏷️ **Git-Aware `.tags` Generation** (`cn tags`): Uses `universal-ctags` to
  index genuine source code while ignoring huge data dumps, JSON caches, `.git`,
  `node_modules`, and build artifacts.
- 👀 **Live File Watcher** (`cn watch`): Automatically re-indexes `.tags` and
  incrementally updates LanceDB embeddings on every save with sub-second
  debounce.
- 🔌 **Model Context Protocol (MCP) Server (`cn mcp`)**: Exposes code
  intelligence tools for AI assistants and coding harnesses and IDEs to find
  code quickly and efficiently.
- ⚙️ **Configurable System Prompts**: For `cn ask`, customize agent persona,
  auditing constraints, or architectural instructions via CLI flags, environment
  variables, or TOML config.

## Quick Start

### Prerequisites

Ensure these command-line tools are installed on your system:

- **[Git](https://git-scm.com/downloads)**: Used for repository discovery and
  ignoring non-tracked files.
- **[universal-ctags](https://github.com/universal-ctags/ctags#installation)**:
  Required for generating `.tags` code symbol indexes.
- **[ripgrep](https://github.com/BurntSushi/ripgrep)** (optional but recommended):
  Provides blazing-fast pattern searching and reference tracing. If missing, `cn`
  falls back to pure-Python traversal.

### Model Context Protocol (MCP) Server

Add to your MCP configuration file (e.g. `claude_desktop_config.json`, `~/.config/antigravity/mcp_config.json`, or `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "codebase-navigator": {
      "command": "uvx",
      "args": ["codebase-navigator", "mcp"]
    }
  }
}
```

### Install Using `uv`

```bash
uv tool install codebase-navigator
```

### Install Using `pip`
```bash
pip install codebase-navigator
```

### Try the CLI tools using uvx

At the top level of a code tree under a git repository, run:

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

## Skills

Use the skill at
[`skills/codebase-navigator/SKILL.md`](skills/codebase-navigator/SKILL.md) for
your agent, so it knows how to use `cn`'s CLI and MCP tools
effectively.

You can copy or symlink it to your agent harness's skill directory. For example:

- **Claude / Cursor / Cline**: `.claude/skills/codebase-navigator/SKILL.md`
- **Antigravity / Gemini**: `~/.gemini/config/skills/codebase-navigator/SKILL.md`

### Why Install the Skill?

- **40%–80% Token Savings**: Teaches agents to read targeted line ranges
  (`codebase_read`) rather than ingesting entire files into context.
- **1-Shot Reference Discovery**: Directs agents to resolve symbol declarations
  and all caller/usage sites in a single turn (`codebase_references`).
- **Semantic Concept Retrieval**: Guides agents to discover relevant
  documentation / documented code and modules via vector search
  (`codebase_search`) before guessing file paths.
- **Call-Tree Tracing**: Helps agents trace multi-step execution flows and
  caller hierarchies (`codebase_call_tree`).

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
1. **CLI flags** (e.g. `--api-key`, `--model`, `--system-prompt`, `--width`,
   `--theme`, `--links`, `--wrap`)
2. **Environment variables** (`OPENROUTER_API_KEY`, `CN_SYSTEM_PROMPT`,
   `CN_WIDTH`, etc.)
3. **Project config** (`.codebase-navigator/config.toml`)
4. **User config** (`~/.config/codebase-navigator/config.toml`)
5. **Built-in defaults**

### Embedding Model & First-Time Download

`cn` uses **FastEmbed** with ONNX runtime for ultra-fast vector embeddings:
- **Default Model**: `FastEmbed ONNX `all-MiniLM-L6-v2`` (384-dimensional vector
  embeddings).
- **First Run**: On the very first invocation of `cn sync`, `cn watch`, or `cn
  search`, the ~80MB ONNX model weights are automatically downloaded once to
  your local machine.
- **Cache Location**: Stored in `~/.cache/fastembed/` (or
  `$FASTEMBED_CACHE_DIR`).
- **Customizing Cache Directory**: ```bash export
  FASTEMBED_CACHE_DIR="/path/to/custom/cache/fastembed" ```
- **100% Offline After Download**: Once downloaded, `cn` operates strictly from
  disk with zero external network calls for embeddings.

## Under the hood

- `cn ask` functions as a lightweight agent harness, where the agent's context
  is stored in the `cn watch` server.
- When you call `cn ask`, the script sends the question to `cn watch` via a unix
  socket.
- In `cn watch`, your question is then used to perform a nearest neighbor
  search using LanceDB.
- The embeddings search results and your question is then sent to the LLM
  as context for the LLM to answer your question.
- The LLM then homes in on the answer by using these tools to efficiently find
  the answer in the codebase:
  - `search`: Semantic and hybrid search over code comments and markdown docs.
  - `tags_lookup`: Instant symbol definition resolution via `.tags`.
  - `read_code`: Range-bounded source inspection with line numbers and clickable file links.
  - `find_references`: 1-shot hybrid symbol definitions + all call and usage sites.
  - `call_tree`: AST & cross-file caller and callee tracer.
  - `grep_search`: Fast pattern search via `rg` (with pure-Python fallback).
- Multi-turn session memory & KV prompt caching: as long as you keep `cn watch`
  running, conversational context is preserved in-memory in `cn watch` across
  successive `cn ask` commands. Follow-up questions hit provider-side prefix KV
  caches for instant responses and lower token cost.

### MCP

`cn`'s **MCP server** provides AI agents (in Antigravity, Claude Desktop, Cursor, Cline, etc.) with 6 code intelligence tools:

- `codebase_search`: Hybrid vector & keyword search in docs and code.
- `codebase_tags`: Ctags symbol definition lookups.
- `codebase_references`: 1-shot definitions and all caller/usage sites.
- `codebase_call_tree`: AST & cross-file caller and callee hierarchy.
- `codebase_read`: Precise line range reader with line numbers and clickable links.
- `codebase_grep`: High-speed regex / literal pattern matcher.

The MCP server operates without requiring any LLM API keys or external
endpoints — your agent performs local FastEmbed ONNX vector queries and ctags
symbol lookups directly for their work.

## Development

The project uses nix and `direnv`. Ensure you have both installed, then:

```bash
git clone https://github.com/9gel/codebase-navigator.git
cd codebase-navigator/
direnv allow
uv run pytest
```

### Evaluation Harness

`cn` includes a multi-language evaluation and benchmarking suite
in [`eval/`](eval/) to measure retrieval accuracy, agent reasoning, latency, and
token efficiency against diverse open-source codebases:

| Repository | Language | Architectural Focus |
|---|---|---|
| **Flask** | Python | Request dispatch lifecycle, Click CLI integration |
| **FastAPI** | Python | Dependency override resolution in `solve_dependencies`, middleware stack assembly |
| **HTTPX** | Python | Mount-based transport routing in `_transport_for_url`, event hooks |
| **Express** | JavaScript / Node.js | Router package delegation, `app.listen()` HTTP server wrap |
| **Vikunja** | Go | Cron worker routines, background periodic job scheduling |
| **uv** | Rust | Core workspace dependency resolver in `uv-resolver` |

#### Running the Benchmark Suite

```bash
# Run all benchmark questions across all exercise repositories:
uv run eval/runner.py

# Benchmark a specific repository:
uv run eval/runner.py --repo fastapi

# Run with keyword validation only (without LLM-as-a-judge):
uv run eval/runner.py --no-judge

# Run A/B token-savings comparison against a generic baseline agent (cat/rg/find/ls):
uv run eval/runner.py --compare-baseline

# Export structured JSON evaluation report:
uv run eval/runner.py --report eval/report.json
```

## References

* [How RAG Can Cut Your AI Coding Costs by 80%](https://blog.mornati.net/how-rag-can-cut-your-ai-coding-costs-by-80)
