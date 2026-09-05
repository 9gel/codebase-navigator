"""Regression tests for the eval-driven retrieval and cost fixes.

Each test pins a defect found by measuring the A/B eval run
(eval/runs/run_20260904_044459_153370) so it cannot silently return.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codebase_navigator.ask import (
    AGENT_TOOLS_SPEC,
    route_question,
    execute_tool_call,
    format_chunks_for_llm,
)
from codebase_navigator.index import _EMBEDDING_INFERENCE_LOCK, VectorIndex
from codebase_navigator.tools import MAX_MATCH_CHARS, grep_search, read_code_ranges

# --- 1. grep byte cap -------------------------------------------------------


def test_grep_match_content_is_byte_capped(tmp_path: Path):
    """One minified line must not be able to flood the agent context.

    A single matched line in vikunja is 486,303 chars; uncapped it cost ~179k
    tokens in one tool result and dominated the whole benchmark's token metric.
    """
    (tmp_path / "generated.js").write_text("var x = " + ("A" * 500_000) + "; // Cron\n")
    matches = grep_search(tmp_path, "Cron")
    assert matches, "expected the long line to match"
    for m in matches:
        assert len(m["content"]) <= MAX_MATCH_CHARS + 64, len(m["content"])
    assert "truncated" in matches[0]["content"]


def test_grep_keeps_short_lines_intact(tmp_path: Path):
    (tmp_path / "a.py").write_text("def handler():\n    return 1\n")
    matches = grep_search(tmp_path, "def handler")
    assert matches[0]["content"] == "def handler():"
    assert "truncated" not in matches[0]["content"]


# --- 2. code-first seed ordering -------------------------------------------


def test_search_floats_code_above_markdown(tmp_path: Path):
    """Doc-heavy repos drowned code in prose (8.6/10 FastAPI seed hits were markdown)."""
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    rows = []
    for i in range(6):
        rows.append(
            {
                "id": f"md{i}",
                "path": f"docs/guide{i}.md",
                "abs_path": str(tmp_path / f"docs/guide{i}.md"),
                "doc_type": "markdown",
                "title": f"guide{i}",
                "start_line": 1,
                "end_line": 5,
                "content": "how the router dispatches a request",
            }
        )
    for i in range(3):
        rows.append(
            {
                "id": f"code{i}",
                "path": f"src/router{i}.py",
                "abs_path": str(tmp_path / f"src/router{i}.py"),
                "doc_type": "code_doc",
                "title": f"router{i}",
                "start_line": 1,
                "end_line": 5,
                "content": "def dispatch(request): ...",
            }
        )
    vecs = idx._embed([r["content"] for r in rows])
    for r, v in zip(rows, vecs):
        r["vector"] = v.tolist() if hasattr(v, "tolist") else list(v)
    idx.table.add(rows)
    idx._ensure_fts_index()

    res = idx.search("how does the router dispatch a request", limit=9)
    types = [r["doc_type"] for r in res]
    first_md = types.index("markdown") if "markdown" in types else len(types)
    last_code = max((i for i, t in enumerate(types) if t == "code_doc"), default=-1)
    assert last_code < first_md, f"code must precede markdown, got {types}"
    # markdown is demoted, never dropped
    assert "markdown" in types


# --- 3. RRF scoring is not saturated ---------------------------------------


def test_scores_are_not_clamped_to_a_ceiling(tmp_path: Path):
    """The old min(0.99, score) collapsed distinct candidates into ties.

    Seven of 25 benchmark questions showed all top-5 chunks at "99%".
    """
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    rows = [
        {
            "id": f"c{i}",
            "path": f"src/mod{i}.py",
            "abs_path": str(tmp_path / f"src/mod{i}.py"),
            "doc_type": "code_doc",
            "title": f"mod{i} > connection_pool (function)" if i < 2 else f"mod{i} > unrelated",
            "start_line": 1,
            "end_line": 9,
            "content": "connection pool keepalive transport" if i < 2 else "colour palette helper",
        }
        for i in range(6)
    ]
    vecs = idx._embed([r["content"] for r in rows])
    for r, v in zip(rows, vecs):
        r["vector"] = v.tolist() if hasattr(v, "tolist") else list(v)
    idx.table.add(rows)
    idx._ensure_fts_index()

    res = idx.search("connection_pool keepalive", limit=6)
    scores = [r["score"] for r in res]
    assert len(set(scores)) > 1, f"scores must discriminate, got {scores}"
    assert max(scores) != 0.99 or len(set(scores)) > 1


def test_relevance_percent_stays_in_range():
    """RRF scores can exceed 1.0 once identifier boosts apply; display must not."""
    chunks = [
        {
            "path": "a.py",
            "abs_path": "/a.py",
            "start_line": 1,
            "end_line": 2,
            "title": "a",
            "doc_type": "code_doc",
            "score": 1.19,
            "content": "x",
        },
        {
            "path": "b.py",
            "abs_path": "/b.py",
            "start_line": 1,
            "end_line": 2,
            "title": "b",
            "doc_type": "code_doc",
            "score": 0.60,
            "content": "y",
        },
    ]
    out = format_chunks_for_llm(chunks)
    assert "Relevance: 100%" in out
    assert "119%" not in out
    assert "Relevance: 50%" in out


# --- 4. question routing ---------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "where is SecureCookieSessionInterface defined?",
        "which file contains FlaskGroup",
        "find the definition of build_middleware_stack",
        "where is preprocess_request?",
    ],
)
def test_identifier_lookups_skip_retrieval(question):
    assert route_question(question) == "lookup"


@pytest.mark.parametrize(
    "question",
    [
        "how does the click cli group loading work?",
        "what happens internally when you call register blueprint on an app instance?",
        "explain the request lifecycle",
        "how does SecureCookieSessionInterface sign cookies?",
        "where does flask run before request hooks and how does the dispatch flow work?",
    ],
)
def test_conceptual_questions_still_retrieve(question):
    assert route_question(question) == "conceptual"


def test_router_handles_empty_input():
    assert route_question("") == "conceptual"
    assert route_question("   ") == "conceptual"


# --- 5. batched reads ------------------------------------------------------


def test_read_code_accepts_multiple_ranges(tmp_path: Path):
    """Half of all benchmark read_code calls re-opened an already-read file."""
    (tmp_path / "a.py").write_text("\n".join(f"line{i}" for i in range(1, 51)))
    (tmp_path / "b.py").write_text("\n".join(f"other{i}" for i in range(1, 51)))
    out = read_code_ranges(
        tmp_path,
        [
            {"path": "a.py", "start_line": 1, "end_line": 3},
            {"path": "a.py", "start_line": 40, "end_line": 42},
            {"path": "b.py", "start_line": 5, "end_line": 6},
        ],
    )
    assert "line1" in out and "line40" in out and "other5" in out
    assert out.count("File: ") == 3


def test_read_code_dispatch_supports_both_forms(tmp_path: Path):
    (tmp_path / "a.py").write_text("alpha\nbeta\ngamma\n")
    single = execute_tool_call(
        tmp_path, "read_code", {"path": "a.py", "start_line": 2, "end_line": 2}
    )
    assert "beta" in single
    batched = execute_tool_call(
        tmp_path,
        "read_code",
        {
            "ranges": [
                {"path": "a.py", "start_line": 1, "end_line": 1},
                {"path": "a.py", "start_line": 3, "end_line": 3},
            ]
        },
    )
    assert "alpha" in batched and "gamma" in batched


def test_read_code_batch_reports_bad_paths(tmp_path: Path):
    (tmp_path / "a.py").write_text("alpha\n")
    out = read_code_ranges(tmp_path, [{"path": "a.py"}, {"path": "missing.py"}])
    assert "alpha" in out
    assert "missing.py" in out


def test_read_code_spec_advertises_ranges():
    spec = next(t for t in AGENT_TOOLS_SPEC if t["function"]["name"] == "read_code")
    assert "ranges" in spec["function"]["parameters"]["properties"]


# --- 6. token-aware chunking ------------------------------------------------


def test_counting_tokenizer_is_not_padded_or_truncated(tmp_path: Path):
    """A padded/truncated tokenizer reports max_length for every input.

    That made every oversize chunk look compliant and silently disabled the cap.
    """
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    cap = idx.embed_max_tokens
    short = idx._count_tokens("File: a.py | Language: Python")
    # Size the long input relative to the window, so the test keeps its meaning
    # whichever embedding model is configured (128 for MiniLM, 8192 for jina-code).
    long_text = "def f():\n" + "    x = 1\n" * (cap * 2)
    long = idx._count_tokens(long_text)
    assert short < 40, f"short header counted as {short}"
    assert long > cap, f"long chunk counted as {long} against cap {cap}"


def test_oversize_chunks_are_split_under_the_encoder_window(tmp_path: Path):
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    cap = idx.embed_max_tokens
    big = {
        "id": "x",
        "path": "src/big.py",
        "abs_path": str(tmp_path / "src/big.py"),
        "doc_type": "code_doc",
        "title": "big.py > handler (function)",
        "start_line": 1,
        "end_line": 400,
        "content": "File: src/big.py | Language: Python\n"
        + "\n".join(f"    value_{i} = compute_something({i})" for i in range(cap * 2)),
    }
    out = idx.split_oversize_chunks([big])
    assert len(out) > 1
    for w in out:
        assert idx._count_tokens(w["content"]) <= cap
        assert w["content"].startswith("File: src/big.py")
    assert len({w["id"] for w in out}) == len(out), "window ids must be unique"


def test_single_giant_line_is_hard_split(tmp_path: Path):
    """Minified sources have no newlines to split on."""
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    chunk = {
        "id": "m",
        "path": "dist/bundle.js",
        "abs_path": str(tmp_path / "dist/bundle.js"),
        "doc_type": "code_doc",
        "title": "bundle.js",
        "start_line": 1,
        "end_line": 1,
        "content": "File: dist/bundle.js | Language: JavaScript\n"
        + ("var a=1;" * (idx.embed_max_tokens * 4)),
    }
    out = idx.split_oversize_chunks([chunk])
    assert len(out) > 1
    for w in out:
        assert idx._count_tokens(w["content"]) <= idx.embed_max_tokens


def test_small_chunks_pass_through_unchanged(tmp_path: Path):
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    small = {
        "id": "s",
        "path": "a.py",
        "abs_path": str(tmp_path / "a.py"),
        "doc_type": "code_doc",
        "title": "a",
        "start_line": 1,
        "end_line": 2,
        "content": "File: a.py | Language: Python\ndef f(): return 1",
    }
    out = idx.split_oversize_chunks([small])
    assert out == [small]


# --- 7. per-file diversity --------------------------------------------------


def test_search_caps_chunks_per_file(tmp_path: Path):
    """Several chunks of one file crowded out other candidate files.

    Unfiltered retrieval returned only 4.8 distinct files per 10 hits; capping
    at one chunk per file raised that to 9.7 and lifted MRR 0.757 -> 0.784.
    """
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    rows = []
    for f in range(4):
        for c in range(5):
            rows.append(
                {
                    "id": f"f{f}c{c}",
                    "path": f"src/mod{f}.py",
                    "abs_path": str(tmp_path / f"src/mod{f}.py"),
                    "doc_type": "code_doc",
                    "title": f"mod{f} > part{c}",
                    "start_line": c * 10 + 1,
                    "end_line": c * 10 + 9,
                    "content": f"def connection_pool_part{c}(): pass",
                }
            )
    vecs = idx._embed([r["content"] for r in rows])
    for r, v in zip(rows, vecs):
        r["vector"] = v.tolist() if hasattr(v, "tolist") else list(v)
    idx.table.add(rows)
    idx._ensure_fts_index()

    res = idx.search("connection pool", limit=4)
    paths = [r["path"] for r in res]
    assert len(set(paths)) == len(paths), f"expected one chunk per file, got {paths}"

    unbounded = idx.search("connection pool", limit=4, max_per_file=None)
    assert len({r["path"] for r in unbounded}) <= len(set(paths))


def test_split_oversize_chunks_is_off_by_default():
    """Splitting to fit the 128-token window measured as a retrieval regression."""
    from codebase_navigator.index import SPLIT_OVERSIZE_CHUNKS

    assert SPLIT_OVERSIZE_CHUNKS is False


# --- 8. doc-seeking queries keep their documentation ------------------------


def test_doc_seeking_queries_bypass_code_first():
    from codebase_navigator.index import is_doc_seeking

    assert is_doc_seeking("What is a Retail Destination")
    assert is_doc_seeking("definition of a shop")
    assert is_doc_seeking("where is the contributing guide")
    # code questions must not trip it
    assert not is_doc_seeking("what happens internally when you call register blueprint")
    assert not is_doc_seeking("what class handles client side session cookies")
    assert not is_doc_seeking("how does the click cli group loading work")


# --- 9. symbol preflight must not look up English words ---------------------


def test_identifier_shape_detection():
    from codebase_navigator.ask import _is_identifier_shaped

    for ident in (
        "create_venv",
        "SecureCookieSessionInterface",
        "app.init",
        "sha256",
        "AsyncHTTPTransport",
    ):
        assert _is_identifier_shaped(ident), ident
    for word in ("contains", "defined", "find", "definition", "where", "implemented"):
        assert not _is_identifier_shaped(word), word


def test_strong_candidates_exclude_english_words():
    """`.tags` really does contain symbols named Contains/defined in big repos.

    "which file contains create_venv?" previously spent all five symbol slots on
    matches for `contains` and never looked up `create_venv` at all.
    """
    from codebase_navigator.ask import extract_strong_symbol_candidates as strong

    assert strong("which file contains create_venv?") == ["create_venv"]
    assert strong("where is RegisterReminderCron defined?") == ["RegisterReminderCron"]
    assert strong("find the definition of AsyncHTTPTransport") == ["AsyncHTTPTransport"]
    assert strong("where is `dispatch_request` implemented") == ["dispatch_request"]
    assert strong("how does the click cli group loading work?") == []


def test_preflight_prefers_identifier_over_earlier_english_word(tmp_path: Path):
    """The identifier must win even though the English word appears first."""
    tags = (
        "!_TAG_FILE_FORMAT\t2\t/extended format/\n"
        'Contains\tsrc/a.rs\t/^fn Contains() {$/;"\tf\tline:10\n'
        'Contains\tsrc/b.rs\t/^fn Contains() {$/;"\tf\tline:20\n'
        'Contains\tsrc/c.rs\t/^fn Contains() {$/;"\tf\tline:30\n'
        'create_venv\tsrc/venv.rs\t/^pub fn create_venv() {$/;"\tf\tline:80\n'
    )
    (tmp_path / ".tags").write_text(tags, encoding="utf-8")

    from codebase_navigator.ask import find_preflight_symbols

    matches = find_preflight_symbols(tmp_path, "which file contains create_venv?")
    assert matches, "expected at least the real identifier"
    assert matches[0]["symbol"] == "create_venv", [m["symbol"] for m in matches]
    assert matches[0]["path"] == "src/venv.rs"


def test_preflight_resolves_dotted_names_via_last_segment(tmp_path: Path):
    """ctags records `app.init = function init()` under the bare name."""
    tags = (
        "!_TAG_FILE_FORMAT\t2\t/extended format/\n"
        'init\tlib/application.js\t/^app.init = function init() {$/;"\tf\tline:59\n'
    )
    (tmp_path / ".tags").write_text(tags, encoding="utf-8")

    from codebase_navigator.ask import find_preflight_symbols

    matches = find_preflight_symbols(tmp_path, "where is app.init defined?")
    assert matches and matches[0]["path"] == "lib/application.js"


# --- 10. per-turn overhead --------------------------------------------------


def test_per_turn_overhead_stays_bounded():
    """Fixed overhead is paid on every turn and was 2.6x the whole cn/baseline gap."""
    import json

    from codebase_navigator.ask import AGENT_TOOLS_SPEC, SYSTEM_PROMPT

    tools = len(json.dumps(AGENT_TOOLS_SPEC)) // 4
    system = len(SYSTEM_PROMPT) // 4
    assert tools + system < 1100, f"per-turn overhead regressed to {tools + system}"


# --- 11. long-context embedding model ---------------------------------------


def test_default_embedding_model_is_the_measured_winner():
    """MiniLM truncates at 128 tokens and discards 61.3% of indexed content --
    and still wins.

    Scored on 26 tasks across Python, JavaScript, Go and Rust it takes the best
    recall@1 (20), the only perfect recall@10 (26) and the best MRR (0.842),
    against jina-embeddings-v2-base-code's 19/25/0.822, at 82.9 chunks/sec
    versus 2.9. Six tasks differ, MiniLM better on four (sign test p = 0.69):
    indistinguishable on quality, 29x cheaper to index.

    What survives truncation is the chunk head -- signature plus docstring --
    which is where the identifying signal lives. This pins the default against
    a repeat of the long-context migration that was reverted.
    """
    from codebase_navigator.config import DEFAULT_EMBEDDING_MODEL, VECTOR_DIM

    assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert VECTOR_DIM == 384


def test_index_rejects_mismatched_embedding_dimensions(tmp_path: Path):
    """Reusing a 384-dim index with a 768-dim model surfaced as an opaque
    'no vector column' error from LanceDB at query time."""
    import lancedb
    import pyarrow as pa

    from codebase_navigator.index import IndexModelMismatch

    from codebase_navigator.config import VECTOR_DIM

    # Derive the stale width from whatever is configured, so the test keeps its
    # meaning whichever model is the default.
    stale_dim = 768 if VECTOR_DIM != 768 else 384

    idx_dir = tmp_path / ".idx"
    idx_dir.mkdir(parents=True)
    db = lancedb.connect(str(idx_dir / "lancedb"))
    stale = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("abs_path", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("title", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), stale_dim)),
        ]
    )
    db.create_table("documents", schema=stale, mode="create")

    with pytest.raises(IndexModelMismatch) as excinfo:
        VectorIndex(tmp_path, custom_index_dir=str(idx_dir))
    message = str(excinfo.value)
    assert str(stale_dim) in message and str(VECTOR_DIM) in message
    assert "cn sync --force" in message


# --- 12. seed size ----------------------------------------------------------


def test_seed_shows_fewer_full_chunks_with_capped_bodies():
    """The seed was ~16.8% of all tokens because it is re-sent on every turn."""
    from codebase_navigator.ask import SEED_CHUNK_BODY_LINES, SEED_FULL_CHUNKS

    assert SEED_FULL_CHUNKS == 2
    chunks = [
        {
            "path": f"src/m{i}.py",
            "abs_path": f"/repo/src/m{i}.py",
            "start_line": 1,
            "end_line": 90,
            "title": f"m{i}",
            "doc_type": "code_doc",
            "score": 1.0 - i * 0.1,
            "content": "\n".join(f"line {n}" for n in range(90)),
        }
        for i in range(5)
    ]
    out = format_chunks_for_llm(
        chunks, full_limit=SEED_FULL_CHUNKS, max_body_lines=SEED_CHUNK_BODY_LINES
    )
    assert "more lines — use read_code for the rest" in out
    # Only the first two render in full; the rest collapse to candidate lines.
    assert out.count("```") == 2 * SEED_FULL_CHUNKS
    assert "Additional Candidate Locations" in out

    uncapped = format_chunks_for_llm(chunks, full_limit=3)
    assert len(out) < len(uncapped)


def test_full_chunk_body_is_not_capped_when_unset():
    chunks = [
        {
            "path": "a.py",
            "abs_path": "/a.py",
            "start_line": 1,
            "end_line": 40,
            "title": "a",
            "doc_type": "code_doc",
            "score": 1.0,
            "content": "\n".join(f"line {n}" for n in range(40)),
        }
    ]
    out = format_chunks_for_llm(chunks)
    assert "line 39" in out
    assert "more lines" not in out


# --- 13. turn budget awareness ----------------------------------------------


def test_budget_warning_fires_before_the_cliff(tmp_path: Path):
    """Measured turn use was p50 5 / p90 11 against a budget of 15, so the hard
    cliff almost never fired and the long tail was the agent's own choice."""
    from unittest.mock import MagicMock, patch

    from codebase_navigator.ask import BUDGET_WARNING_TURNS, AgentSession, LLMConfig

    assert BUDGET_WARNING_TURNS > 0
    cfg = LLMConfig(api_key="k", max_searches=BUDGET_WARNING_TURNS + 1, seed_mode="never")
    (tmp_path / "a.py").write_text("x = 1\n")
    session = AgentSession(tmp_path, cfg)

    tool_turn = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "grep_search",
                                "arguments": json.dumps({"pattern": "x"}),
                            },
                        }
                    ],
                }
            }
        ]
    }
    final = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    mock_chat = MagicMock(side_effect=[tool_turn, tool_turn, final])

    with patch("codebase_navigator.ask.call_chat_completions", mock_chat):
        session.ask("where is x", verbose=False)

    warnings = [
        m
        for m in session.messages
        if m.get("role") == "user" and "Budget check" in str(m.get("content"))
    ]
    assert warnings, "expected a budget warning before the hard cutoff"
    assert "tool turns remain" in warnings[0]["content"]


# --- 14. API retry ----------------------------------------------------------


def test_transient_connection_errors_are_retried():
    """Two benchmark tasks were scored as wrong answers because a socket dropped."""
    import http.client
    from unittest.mock import MagicMock, patch

    from codebase_navigator.ask import call_chat_completions

    good = MagicMock()
    good.read.return_value = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    good.__enter__ = MagicMock(return_value=good)
    good.__exit__ = MagicMock(return_value=False)

    attempts = []

    def flaky(req, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return good

    with (
        patch("urllib.request.urlopen", side_effect=flaky),
        patch("codebase_navigator.ask.time.sleep"),
    ):
        out = call_chat_completions("https://x/v1", "k", {"model": "m", "messages": []})

    assert out["choices"][0]["message"]["content"] == "ok"
    assert len(attempts) == 3, "expected two retries before success"


def test_retry_gives_up_and_raises_after_max_retries():
    import http.client
    from unittest.mock import patch

    from codebase_navigator.ask import call_chat_completions

    def always_fail(req, timeout=None):
        raise http.client.RemoteDisconnected("Remote end closed connection without response")

    with (
        patch("urllib.request.urlopen", side_effect=always_fail),
        patch("codebase_navigator.ask.time.sleep"),
        pytest.raises(RuntimeError, match="Failed to connect"),
    ):
        call_chat_completions("https://x/v1", "k", {"model": "m", "messages": []}, max_retries=2)


def test_non_retryable_http_error_fails_immediately():
    import urllib.error
    from unittest.mock import patch

    from codebase_navigator.ask import call_chat_completions

    attempts = []

    def unauthorized(req, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    with (
        patch("urllib.request.urlopen", side_effect=unauthorized),
        patch("codebase_navigator.ask.time.sleep"),
        pytest.raises(RuntimeError, match="401"),
    ):
        call_chat_completions("https://x/v1", "k", {"model": "m", "messages": []})
    assert len(attempts) == 1, "auth failures must not be retried"


# --- 15. embedding batch efficiency -----------------------------------------


def test_length_sorted_batching_preserves_vectors_and_order(tmp_path: Path):
    """The tokenizer pads to the longest sequence in each batch.

    With a long-context encoder one 8k-token chunk drags every other chunk in its
    batch up to 8k. Sorting by length groups similar sizes together; the caller's
    order must be restored exactly and the vectors must be unchanged.
    """
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    texts = [
        "short",
        "a much longer chunk " * 40,
        "medium length chunk here",
        "x" * 5,
        "another fairly long stretch of text " * 20,
    ]
    sorted_vecs = idx._embed(texts, batch_size=2)
    with _EMBEDDING_INFERENCE_LOCK:
        raw = list(idx.model.embed(texts, batch_size=2))

    assert len(sorted_vecs) == len(texts)
    for got, want in zip(sorted_vecs, raw):
        assert len(got) == len(want)
        assert max(abs(float(a) - float(b)) for a, b in zip(got, want)) < 1e-6


def test_single_text_embed_still_works(tmp_path: Path):
    """The query path embeds one string; it must skip the sort entirely."""
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    out = idx._embed(["where is the router"])
    assert len(out) == 1
    assert len(list(out[0])) == idx_vector_dim()


def idx_vector_dim() -> int:
    from codebase_navigator.config import VECTOR_DIM

    return VECTOR_DIM


# --- 16. embedding dimension resolution -------------------------------------


def test_known_model_dimensions_resolve():
    from codebase_navigator.config import resolve_vector_dim

    assert resolve_vector_dim("jinaai/jina-embeddings-v2-base-code") == 768
    assert resolve_vector_dim("jinaai/jina-embeddings-v2-small-en") == 512
    assert resolve_vector_dim("sentence-transformers/all-MiniLM-L6-v2") == 384


def test_unlisted_model_resolves_from_fastembed_metadata():
    """Unlisted models previously defaulted silently to 384.

    Pointing CN_EMBEDDING_MODEL at a 512-dim model then built a 384-wide table
    and failed inside LanceDB with "Cannot cast to FixedSizeList(384): value at
    index 0 has length 512" -- a dimension mismatch surfaced as an Arrow error.
    """
    from codebase_navigator.config import _MODEL_DIMS, resolve_vector_dim

    assert "BAAI/bge-small-zh-v1.5" not in _MODEL_DIMS
    assert resolve_vector_dim("BAAI/bge-small-zh-v1.5") == 512


def test_unknown_model_raises_instead_of_guessing():
    from codebase_navigator.config import resolve_vector_dim

    with pytest.raises(ValueError, match="Unknown embedding dimension"):
        resolve_vector_dim("made-up/not-a-real-model")


# --- 17. read_code is ranges-only -------------------------------------------


def test_read_code_spec_advertises_only_ranges():
    """48% of reads were single-range and 46% re-opened the file just read.

    express-app-listen made three separate calls to the same file for adjacent
    spans, matching the baseline's turn count while costing 50% more tokens.
    Removing the scalar form removes the choice.
    """
    spec = next(t for t in AGENT_TOOLS_SPEC if t["function"]["name"] == "read_code")
    props = spec["function"]["parameters"]["properties"]
    assert set(props) == {"ranges"}
    assert spec["function"]["parameters"]["required"] == ["ranges"]


def test_scalar_read_code_still_served_with_a_nudge(tmp_path: Path):
    """A model may still emit the old shape; serve it rather than burn a turn."""
    (tmp_path / "a.py").write_text("alpha\nbeta\ngamma\n")
    out = execute_tool_call(tmp_path, "read_code", {"path": "a.py", "start_line": 2, "end_line": 2})
    assert "beta" in out
    assert "ranges" in out  # the nudge toward batching


def test_read_code_without_ranges_or_path_explains_itself(tmp_path: Path):
    out = execute_tool_call(tmp_path, "read_code", {})
    assert "ranges" in out and "Error" in out


# --- 18. weak symbol candidates must be specific ----------------------------


def test_weak_candidates_need_to_be_specific(tmp_path: Path):
    """A real lowercase symbol resolves to one file; an English word scatters.

    "what class handles client side session cookies and signing in flask by
    default?" previously returned five symbols costing 286 tokens per turn --
    `client` from three conftest.py fixtures, `session` from __init__.py and
    ctx.py -- none pointing at sessions.py where the answer lives.
    """
    tags = (
        "!_TAG_FILE_FORMAT\t2\t/extended format/\n"
        # scattered English word: three different files define `client`
        'client\ttests/conftest.py\t/^def client():$/;"\tf\tline:10\n'
        'client\texamples/a/conftest.py\t/^def client():$/;"\tf\tline:11\n'
        'client\texamples/b/conftest.py\t/^def client():$/;"\tf\tline:12\n'
        # specific lowercase symbol: one file
        'flaskgroup\tsrc/cli.py\t/^class flaskgroup:$/;"\tc\tline:99\n'
    )
    (tmp_path / ".tags").write_text(tags, encoding="utf-8")

    from codebase_navigator.ask import find_preflight_symbols

    scattered = find_preflight_symbols(tmp_path, "how does the client work here")
    assert scattered == [], "an English word matching many files must not anchor the agent"

    specific = find_preflight_symbols(tmp_path, "where is flaskgroup")
    assert [m["symbol"] for m in specific] == ["flaskgroup"]


# --- 19. router word-count cap removed --------------------------------------


def test_explicit_answer_shape_routes_to_lookup_regardless_of_length():
    """A <=8 word cap sent a 13-word "what class ..." question to the full seed.

    That cost 949 tokens re-sent on every turn of a 4-turn task whose answer was
    a single named class.
    """
    long_lookup = "what class handles client side session cookies and signing in flask by default?"
    assert len(long_lookup.split()) > 8
    assert route_question(long_lookup) == "lookup"
    assert (
        route_question("which file contains the retry helper for the transport layer") == "lookup"
    )


def test_conceptual_verbs_still_win_over_answer_shape():
    """Length is no longer a signal, so the conceptual check must carry the load."""
    assert route_question("what class handles sessions and how does its signing flow work") == (
        "conceptual"
    )
    assert route_question("explain what function dispatches requests") == "conceptual"


# --- 20. tests and examples must not outrank implementation -----------------


def test_support_paths_are_recognised():
    from codebase_navigator.index import is_support_path

    for p in (
        "tests/test_openapi.py",
        "test/app.router.js",
        "examples/ejs/index.js",
        "src/conftest.py",
        "pkg/foo_test.go",
        "lib/router.spec.ts",
        "benchmarks/run.py",
    ):
        assert is_support_path(p), p
    for p in ("lib/response.js", "src/flask/sessions.py", "crates/uv-resolver/src/lib.rs"):
        assert not is_support_path(p), p


def test_search_demotes_tests_below_implementation(tmp_path: Path):
    """Tests exercise the code being asked about, so they match it semantically
    while never being the answer.

    They occupied 44% of every top-10 across the benchmark (141 of 320 slots).
    On express-view-rendering four of the top five were test files and the real
    implementation sat at rank 10.
    """
    idx = VectorIndex(tmp_path, custom_index_dir=str(tmp_path / ".idx"))
    rows = []
    for i in range(4):
        rows.append(
            {
                "id": f"t{i}",
                "path": f"test/res.render{i}.js",
                "abs_path": str(tmp_path / f"test/res.render{i}.js"),
                "doc_type": "code_doc",
                "title": f"res.render{i} test",
                "start_line": 1,
                "end_line": 9,
                "content": "describe('res.render', function(){ it('renders a view'); })",
            }
        )
    rows.append(
        {
            "id": "impl",
            "path": "lib/response.js",
            "abs_path": str(tmp_path / "lib/response.js"),
            "doc_type": "code_doc",
            "title": "response.js > render (function)",
            "start_line": 1,
            "end_line": 9,
            "content": "res.render = function render(view, opts, done) { ... }",
        }
    )
    vecs = idx._embed([r["content"] for r in rows])
    for r, v in zip(rows, vecs):
        r["vector"] = v.tolist() if hasattr(v, "tolist") else list(v)
    idx.table.add(rows)
    idx._ensure_fts_index()

    res = idx.search("where is res render implemented", limit=5)
    assert res[0]["path"] == "lib/response.js", [r["path"] for r in res]
    # demoted, never dropped
    assert any("test/" in r["path"] for r in res)


def test_queries_about_tests_are_not_demoted():
    from codebase_navigator.index import is_test_seeking

    assert is_test_seeking("where are the tests for the router")
    assert is_test_seeking("show me an example of custom middleware")
    assert not is_test_seeking("where is res render implemented")


# --- 21. repository tree is conceptual-only ---------------------------------


def test_lookup_route_omits_the_repository_tree(tmp_path: Path):
    """All seven benchmark lookup tasks went straight to read_code on the exact
    file and line from the .tags symbol block; none grepped, none used the tree.

    At ~166 tokens re-sent every turn it was pure overhead on precisely the short
    tasks where cn's fixed cost is hardest to amortise.
    """
    from unittest.mock import MagicMock, patch

    from codebase_navigator.ask import AgentSession, LLMConfig

    (tmp_path / "main.py").write_text("x = 1\n")
    chat = MagicMock(return_value={"choices": [{"message": {"role": "assistant", "content": "d"}}]})

    def first_user_message(question: str) -> str:
        session = AgentSession(tmp_path, LLMConfig(api_key="k"))
        with (
            patch("codebase_navigator.ask.call_chat_completions", chat),
            patch("codebase_navigator.ask.execute_search", MagicMock(return_value=[])),
        ):
            session.ask(question, verbose=False)
        return session.messages[1]["content"]

    assert "Repository Structure" not in first_user_message(
        "where is SecureCookieSessionInterface defined?"
    )
    assert "Repository Structure" in first_user_message(
        "how does the request dispatch flow work end to end?"
    )


# --- 22. minified bundles are build artifacts, not source -------------------


def test_minified_javascript_is_detected(tmp_path: Path):
    """`.js` is both what you write and what a bundler emits — a quirk unique to
    JavaScript, so extension alone cannot separate source from build output.

    Two vendored bundles in vikunja produced 60% of its entire 43,790-tag symbol
    index (scalar.standalone.js 18,217 tags, redoc.standalone.js 8,043), because
    ctags reads minified code as thousands of one-character symbols.
    """
    from codebase_navigator.config import is_minified_file

    bundle = tmp_path / "vendor.standalone.js"
    bundle.write_text("!function(e,t){" + "a=1;" * 60_000 + "}();\n")
    assert is_minified_file(bundle)

    minified = tmp_path / "app.min.js"
    minified.write_text("var a=1;\n")
    assert is_minified_file(minified), "name marker alone is sufficient"

    source = tmp_path / "router.js"
    source.write_text("\n".join(f"function handler{i}(req, res) {{ next(); }}" for i in range(200)))
    assert not is_minified_file(source)

    # Non-JS languages do not have this problem and must not be sampled for it.
    go = tmp_path / "generated.go"
    go.write_text("package main\n" + "var x = 1 // " + "y" * 2000 + "\n")
    assert not is_minified_file(go)


def test_minified_files_are_excluded_from_discovery(tmp_path: Path):
    """They poisoned both retrieval paths: `authentication` and `handled` each
    resolved to a vendored bundle, anchoring the agent on build output."""
    from codebase_navigator.tags import get_available_files

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("function start() { return 1; }\n")
    (tmp_path / "src" / "vendor.standalone.js").write_text(
        "!function(){" + "a=1;" * 60_000 + "}()\n"
    )

    code_files, _ = get_available_files(tmp_path)
    names = {p.name for p in code_files}
    assert "app.js" in names
    assert "vendor.standalone.js" not in names


def test_find_references_is_not_in_the_default_spec():
    """Called once across 32 tasks while costing 64 tokens on all 199 turns."""
    names = [t["function"]["name"] for t in AGENT_TOOLS_SPEC]
    assert "find_references" not in names
    assert "grep_search" in names  # covers the need


# --- 23. a glob miss must not read as "not here" ----------------------------


def test_grep_reports_matches_outside_the_glob(tmp_path: Path):
    """`src/flask/*.py` does not recurse, so a search for register_blueprint
    reported zero matches while 25 sat in src/flask/sansio/.

    The agent read that as "not here", tried four more globs, then fell back to
    raw bash grep six times -- twelve turns on a true but misleading message.
    """
    (tmp_path / "src" / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "top.py").write_text("def unrelated(): pass\n")
    (tmp_path / "src" / "pkg" / "sub" / "deep.py").write_text("def register_blueprint(): pass\n")

    out = execute_tool_call(
        tmp_path, "grep_search", {"pattern": "register_blueprint", "path_glob": "src/pkg/*.py"}
    )
    assert "exist elsewhere in the repository" in out
    assert "does not recurse" in out
    assert "deep.py" in out


def test_grep_still_reports_a_genuinely_absent_pattern(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    out = execute_tool_call(tmp_path, "grep_search", {"pattern": "ZZabsentZZ", "path_glob": "*.py"})
    assert "No pattern matches found" in out
    assert "elsewhere" not in out


def test_grep_without_a_glob_is_unchanged(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    out = execute_tool_call(tmp_path, "grep_search", {"pattern": "ZZabsentZZ"})
    assert "No pattern matches found" in out


# --- 24. tool payloads carry relative paths, not absolute URIs --------------


def test_grep_results_omit_absolute_uris(tmp_path: Path):
    """Half of a grep result was file:// URIs the agent never needs.

    On one fastapi grep, 562 of 1,120 tokens were absolute URIs, and the result
    is re-sent on every later turn. The agent navigates by path and line; links
    in the final answer are built from the repository root, given once.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("dependency_overrides = {}\n")
    out = execute_tool_call(tmp_path, "grep_search", {"pattern": "dependency_overrides"})
    assert "a.py" in out
    assert "file://" not in out


def test_seed_chunks_omit_absolute_uris():
    chunks = [
        {
            "path": "src/ipc.py",
            "abs_path": "/repo/src/ipc.py",
            "start_line": 10,
            "end_line": 25,
            "title": "IPC Server",
            "doc_type": "code_doc",
            "score": 0.9,
            "content": "class IPCServer: pass",
        },
        {
            "path": "src/other.py",
            "abs_path": "/repo/src/other.py",
            "start_line": 1,
            "end_line": 5,
            "title": "Other",
            "doc_type": "code_doc",
            "score": 0.5,
            "content": "x = 1",
        },
    ]
    out = format_chunks_for_llm(chunks, full_limit=1)
    assert "file://" not in out
    assert "src/ipc.py:10-25" in out
    assert "src/other.py:1-5" in out


def test_read_code_header_has_no_absolute_uri(tmp_path: Path):
    from codebase_navigator.tools import read_code

    (tmp_path / "a.py").write_text("one\ntwo\nthree\n")
    res = read_code(tmp_path, "a.py", start_line=1, end_line=2)
    assert "file://" not in res["content"]
    assert res["uri"].startswith("file://"), "the uri field is still available to callers"


def test_first_turn_states_the_repository_root(tmp_path: Path):
    """The agent needs the root once to build clickable citations."""
    from unittest.mock import MagicMock, patch

    from codebase_navigator.ask import AgentSession, LLMConfig

    (tmp_path / "main.py").write_text("x = 1\n")
    session = AgentSession(tmp_path, LLMConfig(api_key="k"))
    chat = MagicMock(return_value={"choices": [{"message": {"role": "assistant", "content": "d"}}]})
    with (
        patch("codebase_navigator.ask.call_chat_completions", chat),
        patch("codebase_navigator.ask.execute_search", MagicMock(return_value=[])),
    ):
        session.ask("how does the thing work end to end?", verbose=False)
    assert "Repository root:" in session.messages[1]["content"]
