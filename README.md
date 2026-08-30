# devel-tools

Developer tools for ultra-fast codebase navigation, Git-aware ctags indexing, live watchers, and LanceDB semantic search.

## Features

- 🏷️ **Git-Aware `.tags` Generation**: Uses `universal-ctags` to index genuine source code while completely ignoring huge data dumps, JSON caches, `.git`, `node_modules`, and build artifacts.
- 🧠 **LanceDB Semantic & Hybrid Search**: Vector search powered by `sentence-transformers/all-MiniLM-L6-v2` with hybrid phrase/title match boosting for markdown documentation, glossary terms, and code comments.
- ⚡ **Strict Offline Mode**: Runs 100% locally from disk cache with zero HuggingFace network requests or unauthenticated token warnings.
- 👀 **Live File Watcher**: Automatically re-indexes `.tags` and incrementally updates LanceDB embeddings on every save with sub-second debounce.
- 🔗 **Clickable GitHub Markdown Links**: Returns results formatted as `[file:Lstart-Lend](file:///abs_path#Lstart-Lend)`.

## CLI Commands

| Command | Purpose |
|---|---|
| `devel-search <query> [folder]` | Semantic & hybrid search in markdown docs and code comments |
| `devel-tags <symbol> [folder]` | Fast symbol definition lookup in `.tags` |
| `devel-sync [folder] [--force]` | Synchronize `.tags` and LanceDB vector embeddings |
| `devel-watch [folder]` | Live filesystem watcher for automatic re-indexing |
| `devel-status [folder]` | Inspect index and `.tags` status |
| `devel-nav <cmd> [folder]` | Unified CLI for all operations |

## Development

Requires Nix and `direnv`:

```bash
direnv allow
uv run pytest
```
