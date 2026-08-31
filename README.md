# codebase-navigator

Tools for ultra-fast semantic codebase navigation for humans and AI agents,
Git-aware ctags indexing, live watchers, and
[LanceDB](https://github.com/lancedb/lancedb) semantic search. Self-contained
runtime, no additional servers (e.g. Ollama) to run for embeddings.

Cut 40%-80% token use by having your LLM search precisely, instead of ingesting
entire code bases or search incorrectly using ripgreps.

Get oriented to a codebase quickly by asking natural language questions, instead
of wasting time switcing between files and losing your train of thought.

## How it works

Launch the `cn watch` daemon in one terminal pane, and use it in another.

| Pane 1 | Pane 2 |
| -------- | -------- |
| <pre><code>❯ cn watch<br/> 🚀 Starting cn watch for: /home/user/code/project<br/>   Performing initial sync...<br/>   .tags: Indexed 924 source files (1.78 MB)<br/>   LanceDB: 954 files updated (7814 chunks), 0 pruned.<br/>   Index location: /home/user/code/project/.codebase-navigator<br/>   🔌 IPC Socket: /home/user/code/project/.codebase-navigator/watch.sock<br/> 👀 Watching for file changes (Ctrl+C to stop)...<br/> <br/> [18:15:08] ⚡ Synced 1 file(s) (4 chunks) in 784ms<br/> [18:54:52] ⚡ Synced 1 file(s) (4 chunks) in 710ms<br/> [18:56:19] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [18:56:19] ⚡ Synced 134 file(s) (1505 chunks) in 111142ms<br/> [19:04:26] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [19:04:26] ⚡ Synced 1 file(s) (7 chunks) in 1226ms<br/> [19:04:27] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [19:04:27] ⚡ Synced 1 file(s) (6 chunks) in 1064ms<br/> [19:19:08] ⚡ Synced 1 file(s) (4 chunks) in 753ms<br/> [20:18:13] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [20:18:13] ⚡ Synced 1 file(s) (21 chunks) in 3220ms<br/> [20:18:39] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [20:18:39] ⚡ Synced 1 file(s) (13 chunks) in 1939ms<br/> [20:19:03] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [20:19:03] ⚡ Synced 1 file(s) (16 chunks) in 1703ms<br/> [20:19:29] ⚡ Synced 1 file(s) (4 chunks) in 1285ms<br/> [20:20:50] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [20:20:50] ⚡ Synced 1 file(s) (60 chunks) in 3779ms<br/> [20:21:01] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [20:21:01] ⚡ Synced 1 file(s) (21 chunks) in 1681ms<br/> [20:22:50] 🏷️  Tags updated: Indexed 924 source files (1.78 MB)<br/> [20:22:50] ⚡ Synced 2 file(s) (73 chunks) in 4747ms</code></pre>||

## Features

- 💬 **Agentic Codebase Q&A (`cn ask`)**: (the full RAG) Ask architectural and
  implementation questions in natural language. Powered by an iterative LLM
  reasoning loop with autonomous tool-calling that actively investigates your
  codebase across multiple search rounds, synthesizing clear answers backed by
  clickable file and line links (when supported). Perfect for humans: no need
  to spin up a harness (e.g. opencode), have the LLM ripgrep, consume lots of
  code and wasting a bunch of tokens, and wait forever for an answer.
- 🧠 **LanceDB Semantic & Hybrid Search**: (the retrival) Vector search powered
  by `sentence-transformers/all-MiniLM-L6-v2` with hybrid phrase/title match
  boosting for markdown documentation, glossary terms, and code comments.
  Perfect for LLM in coding harnesses to home in on the right code, without
  wildly ripgrep'ping a bunch of code and waste time and tokens.
- 🏷️ **Git-Aware `.tags` Generation**: (exact symbol match) Uses
  `universal-ctags` to index genuine source code while completely ignoring huge
  data dumps, JSON caches, `.git`, `node_modules`, and build artifacts. Another
  efficient tool for agents to find code.
- 👀 **Live File Watcher**: Automatically re-indexes `.tags` and incrementally
  updates LanceDB embeddings on every save with sub-second debounce.
- 🔗 **Clickable GitHub Markdown Links**: If your terminal supports it, returns
  results formatted as `[file:Lstart-Lend](file:///abs_path#Lstart-Lend)`.
- ⚡ **Strict Offline Mode**: Can run 100% locally from disk, with zero
  HuggingFace network requests.

## Quick Start

### Prerequisites

Ensure these command-line tools are installed on your system:

- **[Git](https://git-scm.com/downloads)**: Used for repository discovery and
  ignoring non-tracked files.
- **[universal-ctags](https://github.com/universal-ctags/ctags#installation)**:
  Required for generating `.tags` code symbol indexes. Installation instructions
  for various platforms:
  - macOS (Homebrew): `brew install universal-ctags`
  - Ubuntu / Debian: `sudo apt install universal-ctags`
  - Arch Linux: `sudo pacman -S universal-ctags`
  - Windows (Chocolatey / Scoop): `choco install universal-ctags` or `scoop
    install universal-ctags`

*(Note: When running via Nix Flakes or `nix run`, these dependencies are
automatically bundled and handled for you).*

### Try it out using uvx

You can run `cn` using `uvx` (the tool runner from [uv](https://docs.astral.sh/uv/)). At the top level of a code tree under a git repository, run:

```bash
# Index the code:
uvx codebase-navigator sync

# Ask about the code:
uvx codebase-navigator ask "What functions call process_data?"

# Search documentation and symbols:
uvx codebase-navigator search "Flush Chunk"

# Tag search for exact symbols:
uvx codebase-navigator tags flush_chunk
```

To make `ask` and `search` faster and update the index as you change code, run
the watcher:

```bash
# Run this in a separate terminal / tmux pane:
uvx codebase-navigator sync

# Ask returns quickly, communicating via a unix socket:
uvx codebase-navigator ask "What functions call process_data?"

# Search returns almost instantly:
uvx codebase-navigator search "Flush Chunk"
```

### Try it out using nix

Run `cn` instantly without installing. At the top level of a code tree under a
git repository, run:

```bash
# Index the code:
nix run github:9gel/codebase-navigator -- sync

# Run help or any command
nix run github:9gel/codebase-navigator -- --help
nix run github:9gel/codebase-navigator -- search "authentication flow"
```

## Installation

### Nix Flakes

Add `codebase-navigator` to your `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    codebase-navigator = {
      url = "github:9gel/codebase-navigator";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, codebase-navigator, ... }:
    let
      system = "x86_64-linux"; # or "aarch64-darwin", etc.
      pkgs = nixpkgs.legacyPackages.''${system};
    in
    {
      # Add to environment packages or devShells:
      devShells.''${system}.default = pkgs.mkShell {
        packages = [
          codebase-navigator.packages.''${system}.default
        ];
      };
    };
}
```

Or install it to your user profile:

```bash
nix profile install github:9gel/codebase-navigator
```

## CLI Commands

The unified `cn` command provides all indexing and search tools:

| Command | Purpose |
|---|---|
| `cn ask <question> [folder]` | LLM-powered codebase Q&A with iterative semantic search |
| `cn search <query> [folder]` | Semantic & hybrid search in markdown docs and code comments |
| `cn tags <symbol> [folder]` | Fast symbol definition lookup in `.tags` |
| `cn sync [folder] [--force]` | Synchronize `.tags` and LanceDB vector embeddings |
| `cn watch [folder]` | Live filesystem watcher for automatic re-indexing |
| `cn status [folder]` | Inspect index and `.tags` status |

## Configuration

`codebase-navigator` supports hierarchical configuration for LLM queries (`cn ask`) and indexing preferences.

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

# Maximum follow-up searches the LLM can execute (default: 5)
max_searches = 5

# Initial number of semantic search chunks provided to the LLM (default: 10)
limit = 10

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

*Note: Keys can be specified either under their respective sections (`[llm]`, `[display]`) or as top-level keys.*

### Environment Variables

Environment variables take precedence over config files:

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` / `CN_API_KEY` / `OPENAI_API_KEY` | API Key for LLM completions | `None` |
| `CN_ENDPOINT` / `CN_BASE_URL` / `OPENROUTER_BASE_URL` | OpenAI-compatible endpoint | `https://openrouter.ai/api/v1` |
| `CN_MODEL` / `OPENROUTER_MODEL` | Default LLM model | `google/gemini-2.5-flash` |
| `CN_MAX_SEARCHES` | Max follow-up searches allowed by the LLM | `5` |
| `CN_ASK_LIMIT` | Initial search result count | `10` |
| `CN_WIDTH` / `CN_MAX_WIDTH` | Maximum terminal wrap width | `terminal width or 100` |
| `CN_THEME` | Terminal theme (`auto`, `dark`, `light`) | `auto` |
| `CN_LINKS` | Link mode (`auto`, `osc8`, `terminal`, `markdown`) | `auto` |
| `CN_WRAP` | Line wrapping (`true`, `false`) | `true (on TTY)` |

### Precedence Order

When resolving settings, `cn` applies the following order of precedence:
1. **CLI flags** (e.g. `--api-key`, `--model`, `--width`, `--theme`, `--links`, `--wrap`)
2. **Environment variables** (`OPENROUTER_API_KEY`, `CN_WIDTH`, `CN_THEME`, etc.)
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
