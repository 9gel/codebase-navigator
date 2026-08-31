# codebase-navigator

Developer tools for ultra-fast codebase navigation, Git-aware ctags indexing, live watchers, and LanceDB semantic search.

## Quick Start

### Using nix

Run `cn` instantly without installing:

```bash
# Run directly from GitHub
nix run github:9gel/codebase-navigator -- sync

# Run help or any command
nix run github:9gel/codebase-navigator -- --help
nix run github:9gel/codebase-navigator -- search "authentication flow"
```

### Using uvx

You can run `cn` using `uvx` (the tool runner from [uv](https://docs.astral.sh/uv/)):

```bash
# Directly from the Git repository:
uvx --from git+https://github.com/9gel/codebase-navigator.git cn --help

# Once published to PyPI:
uvx codebase-navigator --help
```

> **Note:** `cn tags` requires `universal-ctags` and `git` to be installed on your system.

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
| `cn search <query> [folder]` | Semantic & hybrid search in markdown docs and code comments |
| `cn tags <symbol> [folder]` | Fast symbol definition lookup in `.tags` |
| `cn sync [folder] [--force]` | Synchronize `.tags` and LanceDB vector embeddings |
| `cn watch [folder]` | Live filesystem watcher for automatic re-indexing |
| `cn status [folder]` | Inspect index and `.tags` status |

## Development

Requires Nix and `direnv`:

```bash
direnv allow
uv run pytest
```
