from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codebase_navigator.ask import (
    DEFAULT_ENDPOINT,
    DEFAULT_INITIAL_LIMIT,
    DEFAULT_MAX_SEARCHES,
    DEFAULT_MODEL,
    LLMConfig,
    ask_codebase,
    execute_search,
    extract_symbol_candidates,
    find_preflight_symbols,
    format_chunks_for_llm,
    load_llm_config,
)


def test_execute_search_reports_local_embedding_phases(tmp_path: Path):
    phases: list[str] = []

    def fake_search(_query, **kwargs):
        kwargs["progress_callback"]("Waiting for shared embedding model...")
        kwargs["progress_callback"]("Embedding search query...")
        kwargs["progress_callback"]("Querying LanceDB index...")
        return []

    with (
        patch("codebase_navigator.ask.query_socket", return_value=None),
        patch("codebase_navigator.ask.VectorIndex") as index,
    ):
        index.return_value.search.side_effect = fake_search
        results = execute_search(tmp_path, "query", progress_callback=phases.append)

    assert results == []
    assert phases == [
        "Checking for a live index daemon...",
        "Opening local index...",
        "Waiting for shared embedding model...",
        "Embedding search query...",
        "Querying LanceDB index...",
    ]


def test_load_llm_config_defaults(monkeypatch, tmp_path):
    # Clear any ambient env vars
    for env in [
        "CN_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "CN_ENDPOINT",
        "CN_MODEL",
        "CODEBASE_NAVIGATOR_MODEL",
    ]:
        monkeypatch.delenv(env, raising=False)

    # Point home to clean tmp_path so user ~/.config is isolated
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cfg = load_llm_config(folder=Path("/nonexistent"))
    assert cfg.endpoint == DEFAULT_ENDPOINT
    assert cfg.model == DEFAULT_MODEL
    assert cfg.max_searches == DEFAULT_MAX_SEARCHES
    assert cfg.initial_limit == DEFAULT_INITIAL_LIMIT
    assert cfg.api_key is None


def test_load_llm_config_from_toml(tmp_path: Path, monkeypatch):
    for env in ["CN_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(env, raising=False)

    dot_cn = tmp_path / ".codebase-navigator"
    dot_cn.mkdir()
    cfg_file = dot_cn / "config.toml"
    cfg_file.write_text("""
[llm]
endpoint = "https://custom.api.com/v1"
api_key = "toml-secret-key"
model = "anthropic/claude-3.5-sonnet"
max_searches = 3
limit = 8
""")

    cfg = load_llm_config(folder=tmp_path)
    assert cfg.endpoint == "https://custom.api.com/v1"
    assert cfg.api_key == "toml-secret-key"
    assert cfg.model == "anthropic/claude-3.5-sonnet"
    assert cfg.max_searches == 3
    assert cfg.initial_limit == 8


def test_load_llm_config_precedence(tmp_path: Path, monkeypatch):
    # Project TOML
    dot_cn = tmp_path / ".codebase-navigator"
    dot_cn.mkdir()
    cfg_file = dot_cn / "config.toml"
    cfg_file.write_text("""
api_key = "toml-key"
model = "toml-model"
""")

    # Env var overrides TOML
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("CN_MODEL", "env-model")

    cfg = load_llm_config(folder=tmp_path)
    assert cfg.api_key == "env-key"
    assert cfg.model == "env-model"

    # CLI overrides Env var & TOML
    cfg_cli = load_llm_config(
        folder=tmp_path,
        cli_overrides={
            "api_key": "cli-key",
            "model": "cli-model",
            "max_searches": 10,
        },
    )
    assert cfg_cli.api_key == "cli-key"
    assert cfg_cli.model == "cli-model"
    assert cfg_cli.max_searches == 10


def test_format_chunks_for_llm_empty():
    assert format_chunks_for_llm([]) == "No relevant code or documentation chunks found."


def test_format_chunks_for_llm_populated():
    results = [
        {
            "path": "src/ipc.py",
            "abs_path": "/repo/src/ipc.py",
            "start_line": 10,
            "end_line": 25,
            "title": "IPC Server Implementation",
            "doc_type": "code_doc",
            "score": 0.92,
            "content": "class IPCServer:\n    def start(self): pass",
        }
    ]
    formatted = format_chunks_for_llm(results)
    assert "[1] File: src/ipc.py:10-25" in formatted
    assert "file:///repo/src/ipc.py#L10-L25" in formatted
    # Relevance is expressed relative to the top hit, not as an absolute score:
    # RRF values are unbounded and were previously all clamped to 99%.
    assert "Relevance: 100%" in formatted
    assert "class IPCServer:" in formatted


def test_format_chunks_for_llm_tiered():
    results = [
        {
            "path": "src/app.py",
            "abs_path": "/repo/src/app.py",
            "start_line": 1,
            "end_line": 20,
            "title": "App Main",
            "doc_type": "code_doc",
            "score": 0.95,
            "content": "def create_app(): return None",
        },
        {
            "path": "src/sessions.py",
            "abs_path": "/repo/src/sessions.py",
            "start_line": 50,
            "end_line": 70,
            "title": "Session Handler",
            "doc_type": "code_doc",
            "score": 0.82,
            "content": "class SecureSession:\n    # session internals\n    pass",
        },
    ]
    # full_limit=1 means index 1 is full, index 2 is compact candidate
    formatted = format_chunks_for_llm(results, full_limit=1)
    assert "[1] File: src/app.py:1-20" in formatted
    assert "def create_app(): return None" in formatted
    assert "Additional Candidate Locations" in formatted
    assert "[2] [src/sessions.py:50-70]" in formatted
    assert "SecureSession" in formatted


def test_extract_symbol_candidates():
    q = "how does FlaskGroup work and where is preprocess_request called?"
    cands = extract_symbol_candidates(q)
    assert "FlaskGroup" in cands
    assert "preprocess_request" in cands
    assert "where" not in [c.lower() for c in cands]


def test_find_preflight_symbols(tmp_path: Path):
    # Create a mock .tags file
    tag_content = (
        "!_TAG_FILE_FORMAT\t2\t/extended format/\n"
        'FlaskGroup\tsrc/cli.py\t/^class FlaskGroup:$/;"\tc\tline:50\n'
        'SecureCookie\tsrc/sessions.py\t/^class SecureCookie:$/;"\tc\tline:100\n'
    )
    (tmp_path / ".tags").write_text(tag_content, encoding="utf-8")

    matches = find_preflight_symbols(tmp_path, "where is flaskgroup and what is SecureCookie?")
    assert len(matches) == 2
    symbols = [m["symbol"] for m in matches]
    assert "FlaskGroup" in symbols
    assert "SecureCookie" in symbols


def test_ask_codebase_missing_api_key(tmp_path: Path):
    cfg = LLMConfig(api_key=None)
    with pytest.raises(RuntimeError, match="No LLM API key found"):
        ask_codebase(tmp_path, "how does this work?", cfg)


def test_ask_codebase_direct_answer(tmp_path: Path):
    cfg = LLMConfig(api_key="test-key")

    mock_search = MagicMock(
        return_value=[
            {
                "path": "README.md",
                "abs_path": "/repo/README.md",
                "start_line": 1,
                "end_line": 10,
                "title": "Project README",
                "doc_type": "markdown",
                "score": 0.95,
                "content": "# Test Project Overview",
            }
        ]
    )

    mock_chat = MagicMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This project is a code navigation tool.",
                    }
                }
            ]
        }
    )

    with (
        patch("codebase_navigator.ask.execute_search", mock_search),
        patch("codebase_navigator.ask.call_chat_completions", mock_chat),
    ):
        answer, _stats = ask_codebase(tmp_path, "What is this project?", cfg, verbose=False)
        assert answer == "This project is a code navigation tool."
        assert mock_search.call_count == 1
        assert mock_chat.call_count == 1


def test_ask_codebase_with_tool_calling_loop(tmp_path: Path):
    cfg = LLMConfig(api_key="test-key", max_searches=2)

    # Initial search
    mock_search = MagicMock(
        side_effect=[
            # Initial search results
            [
                {
                    "path": "cli.py",
                    "abs_path": "/repo/cli.py",
                    "start_line": 1,
                    "end_line": 5,
                    "title": "CLI",
                    "doc_type": "code",
                    "score": 0.8,
                    "content": "def main(): pass",
                }
            ],
            # Tool search 1 results
            [
                {
                    "path": "ipc.py",
                    "abs_path": "/repo/ipc.py",
                    "start_line": 20,
                    "end_line": 40,
                    "title": "Socket",
                    "doc_type": "code",
                    "score": 0.9,
                    "content": "class IPCServer: pass",
                }
            ],
        ]
    )

    # 1st call: returns tool call to search ipc
    resp_1 = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps(
                                    {"query": "IPCServer definition", "limit": 3}
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }

    # 2nd call: returns final answer after reviewing search results
    resp_2 = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "IPCServer is implemented in `src/ipc.py`.",
                }
            }
        ]
    }

    mock_chat = MagicMock(side_effect=[resp_1, resp_2])

    with (
        patch("codebase_navigator.ask.execute_search", mock_search),
        patch("codebase_navigator.ask.call_chat_completions", mock_chat),
    ):
        # A conceptual question still gets pre-flight retrieval, so execute_search
        # runs twice: once for the seed and once for the agent's own `search` call.
        answer, _stats = ask_codebase(
            tmp_path, "How does the IPCServer handle socket lifecycle?", cfg, verbose=False
        )
        assert answer == "IPCServer is implemented in `src/ipc.py`."
        assert mock_search.call_count == 2
        assert mock_chat.call_count == 2


def test_ask_token_metrics_accumulate_every_model_call(tmp_path: Path):
    cfg = LLMConfig(api_key="test-key", max_searches=2)
    (tmp_path / "demo.py").write_text("value = 1\n", encoding="utf-8")
    first = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "one",
                            "function": {
                                "name": "read_code",
                                "arguments": json.dumps({"path": "demo.py"}),
                            },
                        }
                    ],
                }
            }
        ],
    }
    second = {
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 120},
        },
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
    }

    with (
        patch("codebase_navigator.ask.execute_search", return_value=[]),
        patch("codebase_navigator.ask.call_chat_completions", side_effect=[first, second]),
    ):
        answer, stats = ask_codebase(tmp_path, "Where is demo?", cfg, verbose=False)

    assert answer == "done"
    assert stats["prompt_tokens"] == 300
    assert stats["completion_tokens"] == 30
    assert stats["cached_tokens"] == 160
    assert stats["total_tokens"] == 330
    assert stats["net_tokens"] == 170
    assert stats["api_calls"] == 2
    assert stats["lifetime_total_tokens"] == 330


def test_custom_system_prompt_configuration(tmp_path: Path, monkeypatch):
    for env in ["CN_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(env, raising=False)

    dot_cn = tmp_path / ".codebase-navigator"
    dot_cn.mkdir()
    cfg_file = dot_cn / "config.toml"
    cfg_file.write_text("""
[llm]
system_prompt = "You are a specialized security auditor."
""")

    cfg = load_llm_config(folder=tmp_path)
    assert cfg.system_prompt == "You are a specialized security auditor."

    from codebase_navigator.ask import AgentSession, build_effective_system_prompt

    eff_prompt = build_effective_system_prompt(cfg.system_prompt)
    assert "Additional User Instructions:" in eff_prompt
    assert "You are a specialized security auditor." in eff_prompt

    session = AgentSession(tmp_path, cfg)
    assert session.messages[0]["role"] == "system"
    assert "You are a specialized security auditor." in session.messages[0]["content"]


def test_build_compact_tree(tmp_path: Path):
    from codebase_navigator.ask import build_compact_tree

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "foo.js").write_text("// ignore")
    (tmp_path / ".git").mkdir()
    (tmp_path / "README.md").write_text("# Test")

    tree = build_compact_tree(tmp_path, max_depth=2, max_entries=50)
    assert "src/" in tree
    assert "main.py" in tree
    assert "README.md" in tree
    assert "node_modules" not in tree
    assert ".git" not in tree


def test_ask_empty_seed_skips_preflight_framing(tmp_path: Path):
    from unittest.mock import MagicMock, patch

    from codebase_navigator.ask import AgentSession, LLMConfig

    cfg = LLMConfig(api_key="test-key")
    (tmp_path / "main.py").write_text("print(1)")
    session = AgentSession(tmp_path, cfg)

    mock_search = MagicMock(return_value=[])
    mock_chat = MagicMock(
        return_value={"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    )

    with (
        patch("codebase_navigator.ask.execute_search", mock_search),
        patch("codebase_navigator.ask.call_chat_completions", mock_chat),
    ):
        ans, _ = session.ask("How does the foo subsystem work end to end?", verbose=False)
        assert ans == "Done"
        # Verify user message structure
        user_msg = session.messages[1]["content"]
        assert "Repository Structure:" in user_msg
        assert "No confident pre-flight retrieval chunks found." in user_msg
        assert "Pre-flight Codebase Retrieval:" not in user_msg


def test_ask_lookup_question_skips_seed_search(tmp_path: Path):
    """Identifier lookups must not pay for pre-flight retrieval at all."""
    from unittest.mock import MagicMock, patch

    from codebase_navigator.ask import AgentSession, LLMConfig

    cfg = LLMConfig(api_key="test-key")
    (tmp_path / "main.py").write_text("print(1)")
    session = AgentSession(tmp_path, cfg)

    mock_search = MagicMock(return_value=[])
    mock_chat = MagicMock(
        return_value={"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    )

    with (
        patch("codebase_navigator.ask.execute_search", mock_search),
        patch("codebase_navigator.ask.call_chat_completions", mock_chat),
    ):
        ans, _ = session.ask("Where is SecureCookieSessionInterface defined?", verbose=False)
        assert ans == "Done"
        assert mock_search.call_count == 0, "lookup questions must skip the seed entirely"
        user_msg = session.messages[1]["content"]
        # The tree is conceptual-only: every benchmark lookup task went straight to
        # read_code on the file and line the .tags symbols supplied, so ~166 tokens
        # of directory listing per turn bought nothing.
        assert "Repository Structure:" not in user_msg
        assert "no pre-flight retrieval was run" in user_msg


def test_ask_seed_mode_always_overrides_the_router(tmp_path: Path):
    """seed_mode='always' restores pre-router behaviour for A/B comparison."""
    from unittest.mock import MagicMock, patch

    from codebase_navigator.ask import AgentSession, LLMConfig

    cfg = LLMConfig(api_key="test-key", seed_mode="always")
    (tmp_path / "main.py").write_text("print(1)")
    session = AgentSession(tmp_path, cfg)

    mock_search = MagicMock(return_value=[])
    mock_chat = MagicMock(
        return_value={"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    )

    with (
        patch("codebase_navigator.ask.execute_search", mock_search),
        patch("codebase_navigator.ask.call_chat_completions", mock_chat),
    ):
        session.ask("Where is SecureCookieSessionInterface defined?", verbose=False)
        assert mock_search.call_count == 1
