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


def test_default_embedding_model_has_a_long_window():
    """MiniLM truncated at 128 tokens, discarding 61.3% of all indexed content.

    Splitting chunks to fit was measured and made retrieval worse, so the window
    itself had to grow. This pins the default against silently regressing to a
    short-context model.
    """
    from codebase_navigator.config import DEFAULT_EMBEDDING_MODEL, VECTOR_DIM

    assert DEFAULT_EMBEDDING_MODEL == "jinaai/jina-embeddings-v2-base-code"
    assert VECTOR_DIM == 768


def test_index_rejects_mismatched_embedding_dimensions(tmp_path: Path):
    """Reusing a 384-dim index with a 768-dim model surfaced as an opaque
    'no vector column' error from LanceDB at query time."""
    import lancedb
    import pyarrow as pa

    from codebase_navigator.index import IndexModelMismatch

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
            pa.field("vector", pa.list_(pa.float32(), 384)),  # deliberately stale
        ]
    )
    db.create_table("documents", schema=stale, mode="create")

    with pytest.raises(IndexModelMismatch) as excinfo:
        VectorIndex(tmp_path, custom_index_dir=str(idx_dir))
    message = str(excinfo.value)
    assert "384" in message and "768" in message
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
