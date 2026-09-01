"""LLM-assisted codebase questioning with iterative multi-tool navigation and session memory."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any
import urllib.error
import urllib.request

from .config import get_socket_path
from .index import VectorIndex
from .ipc import ping_socket, query_socket, send_socket_command
from .tags import TagsManager
from .tools import (
    check_ripgrep_installed,
    find_references,
    get_call_tree,
    grep_search,
    read_code,
)

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_MAX_SEARCHES = 15
DEFAULT_INITIAL_LIMIT = 5


class LLMConfig:
    """Configuration settings for LLM queries."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_searches: int = DEFAULT_MAX_SEARCHES,
        initial_limit: int = DEFAULT_INITIAL_LIMIT,
        system_prompt: str | None = None,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.max_searches = max_searches
        self.initial_limit = initial_limit
        self.system_prompt = system_prompt

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "api_key": self.api_key,
            "model": self.model,
            "max_searches": self.max_searches,
            "initial_limit": self.initial_limit,
            "system_prompt": self.system_prompt,
        }


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
    """Load LLM configuration with hierarchical resolution."""
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
        for k in [
            "endpoint",
            "base_url",
            "api_key",
            "model",
            "max_searches",
            "initial_limit",
            "limit",
            "system_prompt",
        ]:
            if k in src:
                merged_toml[k] = src[k]
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
    env_system_prompt = os.environ.get("CN_SYSTEM_PROMPT") or os.environ.get("CODEBASE_NAVIGATOR_SYSTEM_PROMPT")

    endpoint = (
        cli.get("endpoint")
        or env_endpoint
        or merged_toml.get("endpoint")
        or merged_toml.get("base_url")
        or DEFAULT_ENDPOINT
    )
    api_key = cli.get("api_key") or env_api_key or merged_toml.get("api_key")
    model = cli.get("model") or env_model or merged_toml.get("model") or DEFAULT_MODEL

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

    system_prompt = (
        cli.get("system_prompt")
        or env_system_prompt
        or merged_toml.get("system_prompt")
    )

    return LLMConfig(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        max_searches=max_searches,
        initial_limit=initial_limit,
        system_prompt=system_prompt,
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


AGENT_TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Perform semantic and hybrid search across codebase documentation, docstrings, and comments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing the concept, feature, or function to find.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "read_code",
            "description": "Inspect exact source code lines in a file to verify implementations, class structures, or function bodies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path within the repository.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed starting line number to read (optional, default: 1).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed ending line number to read (optional).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tags_lookup",
            "description": "Quickly locate exact or regex symbol definitions (classes, methods, functions) across the codebase using the .tags index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The exact symbol name or regex pattern to look up.",
                    },
                    "exact": {
                        "type": "boolean",
                        "description": "Whether to match the exact symbol name (default: false).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tag matches to return (default: 10).",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": "1-shot hybrid tool: finds symbol definitions and all caller/usage sites across the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The exact function, method, or class name to find definitions and usages for.",
                    },
                    "path_filter": {
                        "type": "string",
                        "description": "Optional glob filter for file paths (e.g. '*.py' or 'src/*').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of call/usage sites to return (default: 15).",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_tree",
            "description": "Trace incoming callers and outgoing callees for a function or class using AST and cross-file references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The function or class name to trace call hierarchy for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional file path where the symbol is defined.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Run pattern or literal keyword search across codebase files using ripgrep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression or keyword string to search for.",
                    },
                    "path_glob": {
                        "type": "string",
                        "description": "Optional file glob filter (e.g. '*.go', '*.rs', 'pkg/**').",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether search is case-sensitive (default: false).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max matches to return (default: 25).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


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
        "User-Agent": "codebase-navigator/0.2.0",
        "HTTP-Referer": "https://github.com/9gel/codebase-navigator",
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


SYSTEM_PROMPT = """You are an expert codebase intelligence assistant and code navigation agent.
Your primary role is to answer questions about this repository accurately, thoroughly, and concisely using concrete evidence from the code and documentation.

Core Operating Principles:
1. Grounding & Code Verification: Never speculate or guess implementation details. Inspect source files using `read_code`, `find_references`, `call_tree`, or `tags_lookup` before making factual assertions.
2. Token Economy & Early Exit: Conserve tokens by avoiding repetitive searches or multiple tool calls with identical/similar terms.
   - Once you have located the primary function, file, and core mechanism that answers the user's question, stop calling tools and synthesize your answer immediately.
   - Do NOT recursively trace downstream library internals, helpers, or external packages unless specifically requested.
   - If a feature is delegated to an external dependency (e.g. in `package.json`, `Cargo.toml`, `go.mod`, or imports) or not in this repo, state this clearly without exhausting tool budgets on exhaustive scans.
3. Scope & Topic Boundary: Strictly focus on the codebase, its architecture, functions, data models, APIs, and workflows. If the user asks general, off-topic, or non-code questions, politely clarify that your focus is navigating and explaining this repository.
4. Citing Evidence: Whenever referencing files or functions, cite them using markdown links: `[path:Lstart-Lend](file:///abs_path#Lstart-Lend)`.
"""


def execute_tool_call(
    folder: Path,
    fn_name: str,
    fn_args: dict[str, Any],
    custom_index_dir: str | None = None,
) -> str:
    """Dispatch tool call to appropriate backend."""
    if fn_name == "search":
        query_term = fn_args.get("query", "").strip()
        doc_type = fn_args.get("type", "all")
        limit = int(fn_args.get("limit", 5))
        res = execute_search(folder, query_term, limit=limit, doc_type=doc_type, custom_index_dir=custom_index_dir)
        return format_chunks_for_llm(res)

    elif fn_name == "read_code":
        path = fn_args.get("path", "")
        start_line = fn_args.get("start_line")
        end_line = fn_args.get("end_line")
        res = read_code(folder, path, start_line=start_line, end_line=end_line)
        if "error" in res:
            return f"Error: {res['error']}"
        return res.get("content", "")

    elif fn_name == "tags_lookup":
        symbol = fn_args.get("symbol", "")
        exact = bool(fn_args.get("exact", False))
        limit = int(fn_args.get("limit", 10))
        tags_mgr = TagsManager(folder)
        matches = tags_mgr.lookup_symbol(symbol, exact=exact, limit=limit)
        if not matches:
            return f"No symbol tags found matching '{symbol}'."
        out = []
        for m in matches:
            out.append(
                f"- Symbol: `{m['symbol']}` ({m.get('kind', 'symbol')}) at [{m['path']}:{m['line']}](file://{m['abs_path']}#L{m['line']})\n  Preview: `{m.get('preview', '')}`"
            )
        return "\n".join(out)

    elif fn_name == "find_references":
        symbol = fn_args.get("symbol", "")
        path_filter = fn_args.get("path_filter")
        limit = int(fn_args.get("limit", 15))
        refs = find_references(folder, symbol, path_filter=path_filter, limit=limit)
        if not refs:
            return f"No definitions or references found for '{symbol}'."
        out = []
        for r in refs:
            t = r.get("type", "reference")
            if t == "definition":
                out.append(f"📌 Definition: [{r['path']}:{r['line']}](file://{r['abs_path']}#L{r['line']}) ({r.get('kind', 'symbol')}) - `{r.get('preview', '')}`")
            else:
                out.append(f"🔍 Usage/Caller: [{r['path']}:{r['line']}](file://{r['abs_path']}#L{r['line']}) - `{r.get('context', '')}`")
        return "\n".join(out)

    elif fn_name == "call_tree":
        symbol = fn_args.get("symbol", "")
        path = fn_args.get("path")
        tree = get_call_tree(folder, symbol, path=path)
        out = [f"Call Tree for `{symbol}`:"]
        if tree.get("definitions"):
            out.append("Definitions:")
            for d in tree["definitions"]:
                out.append(f"  - [{d['path']}:{d['line']}](file://{d['abs_path']}#L{d['line']})")
        if tree.get("callers"):
            out.append("Callers (Functions/Files that invoke this symbol):")
            for c in tree["callers"]:
                fn_ctx = f" (in `{c.get('caller_function')}`)" if c.get("caller_function") else ""
                out.append(f"  - [{c['path']}:{c.get('call_line', 1)}](file://{c['abs_path']}#L{c.get('call_line', 1)}){fn_ctx}: `{c.get('preview', '')}`")
        if tree.get("callees"):
            out.append("Callees (Functions invoked by this symbol):")
            for c in tree["callees"]:
                out.append(f"  - Calls `{c.get('symbol')}` at [{c['path']}:{c['line']}](file://{c['abs_path']}#L{c['line']})")
        if not tree.get("callers") and not tree.get("callees") and not tree.get("definitions"):
            return f"No call tree data found for '{symbol}'."
        return "\n".join(out)

    elif fn_name == "grep_search":
        pattern = fn_args.get("pattern", "")
        path_glob = fn_args.get("path_glob")
        case_sensitive = bool(fn_args.get("case_sensitive", False))
        limit = int(fn_args.get("limit", 25))
        matches = grep_search(folder, pattern, path_glob=path_glob, case_sensitive=case_sensitive, limit=limit)
        if not matches:
            return f"No pattern matches found for '{pattern}'."
        out = []
        for m in matches:
            out.append(f"- [{m['path']}:{m['line']}](file://{m['abs_path']}#L{m['line']}): `{m['content']}`")
        return "\n".join(out)

    return f"Unknown tool: {fn_name}"


def build_effective_system_prompt(custom_prompt: str | None = None) -> str:
    """Combine built-in system prompt guardrails with user-provided system instructions."""
    base = SYSTEM_PROMPT.strip()
    if not custom_prompt or not custom_prompt.strip():
        return base
    return f"{base}\n\nAdditional User Instructions:\n{custom_prompt.strip()}"


class AgentSession:
    """Manages multi-turn conversation state to preserve context and leverage KV caching."""

    def __init__(self, folder: Path, config: LLMConfig, custom_index_dir: str | None = None):
        self.folder = folder
        self.config = config
        self.custom_index_dir = custom_index_dir
        self.effective_system_prompt = build_effective_system_prompt(config.system_prompt)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.effective_system_prompt}]
        self.turn_count = 0
        self.lifetime_prompt_tokens = 0
        self.lifetime_completion_tokens = 0

    def reset(self):
        """Reset conversation messages back to system prompt."""
        self.effective_system_prompt = build_effective_system_prompt(self.config.system_prompt)
        self.messages = [{"role": "system", "content": self.effective_system_prompt}]
        self.turn_count = 0

    def ask(
        self,
        question: str,
        verbose: bool = True,
        output_stream=sys.stderr,
        progress_callback=None,
    ) -> str:
        """Run multi-tool reasoning turn on top of ongoing conversation session."""
        if not self.config.api_key:
            raise RuntimeError(
                "No LLM API key found.\n"
                "Please provide an API key via:\n"
                "  1. Environment variable: export OPENROUTER_API_KEY=your_key (or CN_API_KEY)\n"
                '  2. Project config: .codebase-navigator/config.toml (api_key = "...")\n'
                "  3. Global config: ~/.config/codebase-navigator/config.toml\n"
                '  4. CLI argument: cn ask --api-key your_key "question"'
            )

        self.turn_count += 1

        def emit(line: str):
            if verbose:
                print(line, file=output_stream, flush=True)
            if progress_callback:
                progress_callback(line)

        # Check ripgrep status for best performance
        check_ripgrep_installed(verbose=verbose, output_stream=output_stream)

        # Pre-flight seed search
        emit("🔍 Searching codebase...")

        initial_chunks = execute_search(
            self.folder,
            question,
            limit=self.config.initial_limit,
            custom_index_dir=self.custom_index_dir,
        )

        emit(f"🤖 Retrieved {len(initial_chunks)} code/doc chunks. Reasoning with agent...")

        initial_context_text = format_chunks_for_llm(initial_chunks)

        user_content = (
            f"Question:\n{question}\n\n"
            f"Pre-flight Codebase Retrieval:\n{initial_context_text}"
        )
        self.messages.append({"role": "user", "content": user_content})

        searches_remaining = self.config.max_searches
        seen_tool_calls: set[str] = set()
        turn_completion_tokens = 0
        last_prompt_tokens = 0
        tool_calls_count = 0

        while True:
            payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": self.messages,
                "temperature": 0.2,
            }
            if searches_remaining > 0:
                payload["tools"] = AGENT_TOOLS_SPEC
                payload["tool_choice"] = "auto"

            response_data = call_chat_completions(self.config.endpoint, self.config.api_key, payload)
            usage = response_data.get("usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            last_prompt_tokens = p_tok
            turn_completion_tokens += c_tok
            self.lifetime_prompt_tokens = max(self.lifetime_prompt_tokens, p_tok)
            self.lifetime_completion_tokens += c_tok

            choices = response_data.get("choices", [])
            if not choices:
                raise RuntimeError(f"Unexpected empty response from LLM: {response_data}")

            choice = choices[0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls")

            # Handle tool calls
            if tool_calls and searches_remaining > 0:
                self.messages.append(msg)
                for tool_call in tool_calls:
                    fn = tool_call.get("function", {})
                    fn_name = fn.get("name")
                    fn_args_raw = fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except (json.JSONDecodeError, TypeError, ValueError):
                        fn_args = {}

                    search_num = (self.config.max_searches - searches_remaining) + 1
                    tool_calls_count += 1
                    call_sig = f"{fn_name}:{json.dumps(fn_args, sort_keys=True)}"

                    arg_summary = ", ".join(f"{k}={v!r}" for k, v in list(fn_args.items())[:3])
                    emit(f"🔎 [Tool {tool_calls_count}: {fn_name}] {arg_summary}...")

                    tool_output = execute_tool_call(
                        self.folder,
                        fn_name,
                        fn_args,
                        custom_index_dir=self.custom_index_dir,
                    )
                    seen_tool_calls.add(call_sig)

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "name": fn_name,
                            "content": tool_output,
                        }
                    )

                searches_remaining -= 1
                if searches_remaining <= 0:
                    emit("ℹ️ Search budget limit reached. Generating final answer...")
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "You have completed your tool budget. Please synthesize your complete final answer using all the evidence gathered.",
                        }
                    )
                continue

            # Final answer
            content = msg.get("content") or ""
            self.messages.append({"role": "assistant", "content": content})
            # Check if model refused or could not find answer
            content_lower = content.lower()
            refusal_patterns = [
                "cannot answer", "unable to answer", "i cannot find", "could not find",
                "not related to this codebase", "outside the scope of this repository",
                "weather", "i am a codebase intelligence", "no relevant code",
            ]
            is_refusal = any(p in content_lower for p in refusal_patterns)

            stats = {
                "turn_prompt_tokens": last_prompt_tokens,
                "turn_completion_tokens": turn_completion_tokens,
                "turn_total_tokens": last_prompt_tokens + turn_completion_tokens,
                "tool_calls_count": tool_calls_count,
                "lifetime_prompt_tokens": self.lifetime_prompt_tokens,
                "lifetime_completion_tokens": self.lifetime_completion_tokens,
                "lifetime_total_tokens": self.lifetime_prompt_tokens + self.lifetime_completion_tokens,
                "status": "refusal" if is_refusal else "answered",
            }
            return content, stats


def ask_codebase(
    folder: Path,
    question: str,
    config: LLMConfig,
    custom_index_dir: str | None = None,
    verbose: bool = True,
    output_stream=sys.stderr,
    new_session: bool = False,
    progress_callback = None,
) -> tuple[str, dict[str, Any]]:
    """Query codebase using daemon session over socket if running, or standalone session."""
    from .ipc import discover_daemon_target, send_target_command

    # 1. Try sending ask request to active cn watch daemon (via socket or TCP port)
    target = discover_daemon_target(folder, custom_index_dir)
    if target is not None:
        def handle_remote_progress(line: str):
            if verbose:
                print(line, file=output_stream, flush=True)
            if progress_callback:
                progress_callback(line)

        res = send_target_command(
            target,
            action="ask",
            payload={
                "question": question,
                "config": config.to_dict(),
                "new_session": new_session,
                "verbose": True,
            },
            timeout=180.0,
            progress_callback=handle_remote_progress,
        )
        if res:
            if res.get("status") == "version_mismatch":
                raise RuntimeError(res.get("error", "Version mismatch between cn client and cn watch daemon."))
            if res.get("status") == "ok":
                return res.get("answer", ""), res.get("stats", {})
            if res.get("status") == "error":
                raise RuntimeError(f"Daemon error: {res.get('error', 'Unknown error')}")

    # 2. Standalone fallback (warn user)
    if verbose:
        print(
            "💡 Tip: 'cn watch' is not running. LanceDB index is loaded in-process and session context is not preserved.\n"
            "   Run 'cn watch' in a separate terminal for instant vector searches and multi-turn KV prompt caching!\n",
            file=output_stream,
        )

    session = AgentSession(folder, config, custom_index_dir=custom_index_dir)
    return session.ask(question, verbose=verbose, output_stream=output_stream, progress_callback=progress_callback)
