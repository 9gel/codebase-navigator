# codebase-navigator

Tools for ultra-fast semantic codebase navigation, Git-aware ctags indexing, live watchers, and [LanceDB](https://github.com/lancedb/lancedb) semantic search. Self-contained runtime, no additional servers (e.g. Ollama) to run for embeddings.

## How it works


## Quick Start

### Prerequisites

Ensure these command-line tools are installed on your system:

- **[Git](https://git-scm.com/downloads)**: Used for repository discovery and ignoring non-tracked files.
- **[universal-ctags](https://github.com/universal-ctags/ctags#installation)**: Required for generating `.tags` code symbol indexes. Installation instructions for various platforms:
  - macOS (Homebrew): `brew install universal-ctags`
  - Ubuntu / Debian: `sudo apt install universal-ctags`
  - Arch Linux: `sudo pacman -S universal-ctags`
  - Windows (Chocolatey / Scoop): `choco install universal-ctags` or `scoop install universal-ctags`

*(Note: When running via Nix Flakes or `nix run`, these dependencies are automatically bundled and handled for you).*

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

## Features

- 🏷️ **Git-Aware `.tags` Generation**: Uses `universal-ctags` to index genuine source code while completely ignoring huge data dumps, JSON caches, `.git`, `node_modules`, and build artifacts.
- 🧠 **LanceDB Semantic & Hybrid Search**: Vector search powered by `sentence-transformers/all-MiniLM-L6-v2` with hybrid phrase/title match boosting for markdown documentation, glossary terms, and code comments.
- ⚡ **Strict Offline Mode**: Runs 100% locally from disk cache with zero HuggingFace network requests or unauthenticated token warnings.
- 👀 **Live File Watcher**: Automatically re-indexes `.tags` and incrementally updates LanceDB embeddings on every save with sub-second debounce.
- 🔗 **Clickable GitHub Markdown Links**: Returns results formatted as `[file:Lstart-Lend](file:///abs_path#Lstart-Lend)`.

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
```

*Note: Keys can be specified either under the `[llm]` section or as top-level keys.*

### Environment Variables

Environment variables take precedence over config files:

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` / `CN_API_KEY` / `OPENAI_API_KEY` | API Key for LLM completions | `None` |
| `CN_ENDPOINT` / `CN_BASE_URL` / `OPENROUTER_BASE_URL` | OpenAI-compatible endpoint | `https://openrouter.ai/api/v1` |
| `CN_MODEL` / `OPENROUTER_MODEL` | Default LLM model | `google/gemini-2.5-flash` |
| `CN_MAX_SEARCHES` | Max follow-up searches allowed by the LLM | `5` |
| `CN_ASK_LIMIT` | Initial search result count | `10` |

### Precedence Order

When resolving settings, `cn` applies the following order of precedence:
1. **CLI flags** (e.g. `--api-key`, `--model`, `--endpoint`, `--limit`, `--max-searches`)
2. **Environment variables** (`OPENROUTER_API_KEY`, `CN_ENDPOINT`, etc.)
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

