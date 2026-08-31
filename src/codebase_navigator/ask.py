"""LLM-assisted codebase questioning with iterative semantic search."""

from __future__ import annotations

import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_socket_path
from .index import VectorIndex
from .ipc import query_socket

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_MAX_SEARCHES = 5
DEFAULT_INITIAL_LIMIT = 10


@dataclass
class LLMConfig:
    """Configuration settings for LLM queries."""

    endpoint: str = DEFAULT_ENDPOINT
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_searches: int = DEFAULT_MAX_SEARCHES
    initial_limit: int = DEFAULT_INITIAL_LIMIT


def _parse_toml_file(p: Path) -> dict[str, Any]:
    """Safely parse a TOML file if it exists."""
    if p.is_file():
        try:
            with open(p, "rb") as f:
                return tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            return {}
    return {}


def load_llm_config(
    folder: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> LLMConfig:
    """Load LLM configuration with hierarchical resolution:

    1. CLI argument overrides (highest priority)
    2. Environment variables (CN_API_KEY, OPENROUTER_API_KEY, etc.)
    3. Project configuration (.codebase-navigator/config.toml, etc.)
    4. User global configuration (~/.config/codebase-navigator/config.toml, etc.)
    5. Built-in defaults
    """
    cli = {k: v for k, v in (cli_overrides or {}).items() if v is not None}

    # 1. Global user configs
    home = Path.home()
    user_candidates = [
        home / ".config" / "codebase-navigator" / "config.toml",
        home / ".config" / "codebase-navigator.toml",
        home / ".config" / "codebase-navigator" / "config",
    ]
    user_data: dict[str, Any] = {}
    for uc in user_candidates:
        if uc.is_file():
            user_data = _parse_toml_file(uc)
            break

    # 2. Project local configs
    project_data: dict[str, Any] = {}
    if folder:
        project_candidates = [
            folder / ".codebase-navigator" / "config.toml",
            folder / "codebase-navigator.toml",
            folder / ".codebase-navigator.toml",
        ]
        for pc in project_candidates:
            if pc.is_file():
                project_data = _parse_toml_file(pc)
                break

    # Merge TOML layers (user < project)
    merged_toml: dict[str, Any] = {}
    for src in [user_data, project_data]:
        # Handle top-level keys
        for k in [
            "endpoint",
            "base_url",
            "api_key",
            "model",
            "max_searches",
            "initial_limit",
            "limit",
        ]:
            if k in src:
                merged_toml[k] = src[k]
        # Handle [llm] section
        llm_sec = src.get("llm", {})
        if isinstance(llm_sec, dict):
            for k, v in llm_sec.items():
                merged_toml[k] = v

    # 3. Environment variables
    env_api_key = (
        os.environ.get("CN_API_KEY")
        or os.environ.get("CODEBASE_NAVIGATOR_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    env_endpoint = (
        os.environ.get("CN_ENDPOINT")
        or os.environ.get("CN_BASE_URL")
        or os.environ.get("CODEBASE_NAVIGATOR_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )
    env_model = (
        os.environ.get("CN_MODEL")
        or os.environ.get("CODEBASE_NAVIGATOR_MODEL")
        or os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("OPENAI_MODEL")
    )
    env_max_searches = os.environ.get("CN_MAX_SEARCHES")
    env_initial_limit = os.environ.get("CN_ASK_LIMIT") or os.environ.get("CN_INITIAL_LIMIT")

    # Resolve endpoint
    endpoint = (
        cli.get("endpoint")
        or env_endpoint
        or merged_toml.get("endpoint")
        or merged_toml.get("base_url")
        or DEFAULT_ENDPOINT
    )

    # Resolve api_key
    api_key = cli.get("api_key") or env_api_key or merged_toml.get("api_key")

    # Resolve model
    model = cli.get("model") or env_model or merged_toml.get("model") or DEFAULT_MODEL

    # Resolve max_searches
    max_searches_raw = (
        cli.get("max_searches")
        or env_max_searches
        or merged_toml.get("max_searches")
        or DEFAULT_MAX_SEARCHES
    )
    try:
        max_searches = int(max_searches_raw)
    except (ValueError, TypeError):
        max_searches = DEFAULT_MAX_SEARCHES

    # Resolve initial_limit
    initial_limit_raw = (
        cli.get("limit")
        or env_initial_limit
        or merged_toml.get("limit")
        or merged_toml.get("initial_limit")
        or DEFAULT_INITIAL_LIMIT
    )
    try:
        initial_limit = int(initial_limit_raw)
    except (ValueError, TypeError):
        initial_limit = DEFAULT_INITIAL_LIMIT

    return LLMConfig(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        max_searches=max_searches,
        initial_limit=initial_limit,
    )


def execute_search(
    folder: Path,
    query: str,
    limit: int = 5,
    doc_type: str = "all",
    custom_index_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Perform semantic vector search using socket daemon if available, else in-process."""
    socket_path = get_socket_path(folder, custom_index_dir)
    results = query_socket(socket_path, query, limit=limit, doc_type=doc_type)
    if results is not None:
        return results

    idx = VectorIndex(folder, custom_index_dir)
    return idx.search(query, limit=limit, doc_type=doc_type)


def format_chunks_for_llm(results: list[dict[str, Any]]) -> str:
    """Format search results cleanly for LLM consumption."""
    if not results:
        return "No relevant code or documentation chunks found."

    chunks_text = []
    for idx, r in enumerate(results, start=1):
        rel_p = r.get("path", "")
        abs_p = r.get("abs_path", "")
        s_line = r.get("start_line", 1)
        e_line = r.get("end_line", 1)
        title = r.get("title", "")
        doc_type = r.get("doc_type", "")
        score_pct = int(r.get("score", 0.0) * 100)
        content = r.get("content", "")

        header = f"[{idx}] File: {rel_p}:{s_line}-{e_line} ({doc_type}) — {title} (Relevance: {score_pct}%)\nAbsURI: file://{abs_p}#L{s_line}-L{e_line}"
        body = f"```\n{content}\n```"
        chunks_text.append(f"{header}\n{body}")

    return "\n\n".join(chunks_text)


SEARCH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Perform semantic and keyword search across codebase files and documentation chunks. Use this to find functions, classes, comments, architecture, and module logic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or keyword search query. Vary your terms if initial queries do not yield enough detail.",
                },
                "type": {
                    "type": "string",
                    "enum": ["all", "md", "code_doc", "markdown", "code"],
                    "description": "Filter by document type (optional, default: all).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (optional, default: 5).",
                },
            },
            "required": ["query"],
        },
    },
}


def call_chat_completions(
    endpoint: str,
    api_key: str | None,
    payload: dict[str, Any],
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Send a request to an OpenAI-compatible /chat/completions endpoint."""
    url = endpoint.strip()
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "codebase-navigator/0.1.0",
        "HTTP-Referer": "https://github.com/9gel/devel-tools",
        "X-Title": "codebase-navigator",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read().decode("utf-8")
            return json.loads(resp_body)
    except urllib.error.HTTPError as e:
        try:
            error_content = e.read().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            error_content = ""
        raise RuntimeError(
            f"LLM API request failed with HTTP {e.code} ({e.reason}): {error_content}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to LLM endpoint ({url}): {e.reason}") from e


SYSTEM_PROMPT = """You are an expert codebase intelligence assistant.
Your goal is to answer the developer's question accurately and thoroughly using evidence from the codebase.

Instructions:
1. You are provided with initial semantic search results from the codebase.
2. If you need more details, specific function definitions, or related docs, you can invoke the `search` function with varied descriptive queries. Do not repeat the exact same search query.
3. When referencing files and line ranges, use markdown links format: [path:Lstart-Lend](file:///abs_path#Lstart-Lend).
4. Provide concrete explanations, citing relevant files and line numbers.
"""


def ask_codebase(
    folder: Path,
    question: str,
    config: LLMConfig,
    custom_index_dir: str | None = None,
    verbose: bool = True,
    output_stream=sys.stderr,
) -> str:
    """Execute LLM codebase Q&A with bounded iterative semantic search."""
    if not config.api_key:
        raise RuntimeError(
            "No LLM API key found.\n"
            "Please provide an API key via:\n"
            "  1. Environment variable: export OPENROUTER_API_KEY=your_key (or CN_API_KEY)\n"
            '  2. Project config: .codebase-navigator/config.toml (api_key = "...")\n'
            "  3. Global config: ~/.config/codebase-navigator/config.toml\n"
            '  4. CLI argument: cn ask --api-key your_key "question"'
        )

    # Step 1: Initial semantic search
    if verbose:
        print(
            f'🔍 Searching codebase for: "{question}" (limit: {config.initial_limit})...',
            file=output_stream,
        )

    initial_chunks = execute_search(
        folder,
        question,
        limit=config.initial_limit,
        custom_index_dir=custom_index_dir,
    )

    if verbose:
        print(f"✓ Found {len(initial_chunks)} relevant code/doc chunks.", file=output_stream)

    initial_context_text = format_chunks_for_llm(initial_chunks)

    user_prompt = (
        f"User Question:\n{question}\n\nInitial Codebase Search Results:\n{initial_context_text}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    searches_remaining = config.max_searches
    seen_queries: set[str] = set()

    while True:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if searches_remaining > 0:
            payload["tools"] = [SEARCH_TOOL_SPEC]
            payload["tool_choice"] = "auto"

        response_data = call_chat_completions(config.endpoint, config.api_key, payload)
        choices = response_data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Unexpected empty response from LLM: {response_data}")

        choice = choices[0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls")

        # If model responded with tool calls and we still have budget
        if tool_calls and searches_remaining > 0:
            messages.append(msg)
            for tool_call in tool_calls:
                fn = tool_call.get("function", {})
                fn_name = fn.get("name")
                fn_args_raw = fn.get("arguments", "{}")
                try:
                    fn_args = (
                        json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    fn_args = {}

                if fn_name == "search":
                    query_term = fn_args.get("query", "").strip()
                    doc_type = fn_args.get("type", "all")
                    limit = int(fn_args.get("limit", 5))

                    search_num = (config.max_searches - searches_remaining) + 1
                    if verbose:
                        print(
                            f'🔎 [Search {search_num}/{config.max_searches}] Query: "{query_term}" (type: {doc_type}, limit: {limit})...',
                            file=output_stream,
                        )

                    search_results = execute_search(
                        folder,
                        query_term,
                        limit=limit,
                        doc_type=doc_type,
                        custom_index_dir=custom_index_dir,
                    )
                    tool_content = format_chunks_for_llm(search_results)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "name": "search",
                            "content": tool_content,
                        }
                    )
                    seen_queries.add(query_term.lower())

            searches_remaining -= 1
            if searches_remaining <= 0:
                if verbose:
                    print(
                        "ℹ️ Search budget limit reached. Generating final answer...",
                        file=output_stream,
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": "You have reached the maximum number of searches. Please synthesize your complete final answer using all the search results and context gathered.",
                    }
                )
            continue

        # Final answer received
        content = msg.get("content") or ""
        return content
