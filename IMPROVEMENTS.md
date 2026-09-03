# Improvements Plan

Driven by findings from the A/B eval run (`report_20260903_110303.json`), where CN
scored 21/25 vs baseline 24/25 and used **more tokens in 14 of 25 cases**.

---

## Eval Result Summary

| Metric              | CN          | Baseline    |
|---------------------|-------------|-------------|
| Pass rate           | 21/25 (84%) | 24/25 (96%) |
| Avg tokens          | 6,462       | 7,751       |
| Used more tokens    | **14/25**   | 11/25       |
| Avg wall-clock time | 104.2 s     | 116.4 s     |

CN's average token count is slightly lower, pulled by a few big wins (Q4 80%
savings, Q7 69%, Q20 63%). But the majority of individual cases are regressions.
Two of the four CN failures were judge infrastructure bugs (empty JSON response →
auto-fail), not actual answer quality problems.

---

## Defects & Corresponding Improvements

### 1. Non-Python code is nearly invisible to vector search

**Defect**: `extractor.py` `_extract_generic_comments()` only indexes comment
blocks ≥ 3 lines for non-Python files (.rs, .go, .js, .ts, .java, etc.). Zero
code is indexed — no function signatures, struct/interface declarations, or
method bodies. For Python, `_extract_python()` captures only signatures +
docstrings, ignoring function bodies entirely.

**Evidence**: 18 of 25 eval questions showed "Retrieved 0 code/doc chunks" in
the pre-flight seed search. The 7 that retrieved results were repos with rich
inline doc-comments (Express JSDoc, uv Rust `///` docs).

**Improvement**: Extend extractors to create **code structure chunks**:

- **All languages**: Extract function/method signatures with the first ~30 lines
  of the body. Use regex patterns per language family:
  - Rust: `fn`, `pub fn`, `impl`, `struct`, `enum`, `trait`, `mod`
  - Go: `func`, `type ... struct`, `type ... interface`
  - JS/TS: `function`, `class`, `export`, `const ... = ... =>`
  - Or use Tree-sitter for proper AST-based chunking across languages.
- **Python**: Include function bodies (capped at ~50 lines or 512 tokens) in
  addition to signatures and docstrings.
- **All**: Prepend contextual metadata to each chunk before embedding:
  `File: path/to/file.go | Language: Go | Package: auth`. This is a cheap
  version of Anthropic's "contextual retrieval" that doesn't require an LLM.

### 2. Pseudo-hybrid search doesn't find exact keyword matches

**Defect**: `index.py` search (lines 202–276) runs vector search first, then
applies keyword boosting as a post-filter on the vector candidate pool only. A
chunk containing the exact function name but with poor vector proximity **never
enters the candidate pool** and cannot be found.

Additionally, `content_lower` is computed on line 244 but **never used** — all
keyword and phrase matching only checks `title_lower`. Matches in the chunk body
receive zero boost.

**Improvement**:

- **Implement true BM25 + vector search with Reciprocal Rank Fusion (RRF)**.
  LanceDB supports `create_fts_index()` natively. Run both searches in parallel
  and merge with RRF (k=60, alpha tunable starting at 0.5).
- **Immediate fix**: Use `content_lower` for keyword matching (add `+0.02` per
  term match in content body, smaller than the `+0.04` title match boost).

### 3. Agent wastes tool calls on directory orientation

**Defect**: CN agent traces show 2–4 wasted tool calls per question for
directory discovery: `bash('ls -la')`, `bash('find . -maxdepth 3 -type d')`,
`bash('ls')`. The baseline agent also explores directories, but CN's per-turn
overhead is 1,239 tokens vs baseline's 639 tokens (+93.9%), so each wasted turn
costs more.

Path confusion also wastes turns — the agent sometimes tries paths like
`exercises/httpx/httpx/_auth.py` instead of `httpx/_auth.py`, triggering 3–6
recovery calls.

**Improvement**:

- Inject a **compact directory tree** (top 2–3 levels) into the initial user
  message alongside the seed search results. This gives the agent immediate
  spatial awareness without needing `ls`.
- In the eval harness, ensure `cwd` is set to the repo root so relative paths
  always work. In production, ensure `read_code` and other tools resolve paths
  relative to the indexed project root.

### 4. Pre-flight seed injection anchors on nothing

**Defect**: The system prompt tells the agent: *"Trust the pre-flight retrieval
first… answer directly from them and verify only the specific lines you cite
with read_code."* But 18/25 eval cases returned 0 pre-flight results. The agent
receives "Retrieved 0 code/doc chunks" and then has to start from scratch
anyway, having wasted the first turn processing empty context.

When pre-flight results are irrelevant docstrings, the anchoring instruction
causes the agent to hallucinate implementation details from docstrings rather
than reading actual code — this contributed to the express router failure (Q16)
and httpx streaming failure (Q18).

**Improvement**:

- **Skip seed injection entirely** when pre-flight returns 0 results or all
  results score below 0.50.
- When results are injected, soften the anchoring language: suggest the agent
  *"consider"* pre-flight results rather than *"trust"* them.
- Consider **HyDE** (Hypothetical Document Embeddings): have the LLM generate a
  short hypothetical answer, embed that, and search with it instead of the raw
  natural language question. This bridges the gap between conversational queries
  and code identifiers.

### 5. Embedding model is not code-aware

**Defect**: `all-MiniLM-L6-v2` (384 dimensions, 256–512 token context window)
is a general-purpose sentence embedding model trained on natural language. It was
not trained on code and does not understand the semantic relationship between a
natural language question and code identifiers, syntax, or structure.

**Evidence**: Even when chunks exist, vector similarity scores are mediocre — the
model struggles to match "where does Flask run before_request hooks" to a chunk
titled `app.py > preprocess_request (function)`.

**Improvement**: Upgrade to a code-aware embedding model:

- **Open-source, self-hosted**: `nomic-ai/nomic-embed-code` (best
  quality/speed), or `Alibaba-NLP/gte-base-en-v1.5` (768 dims, strong code
  performance). Both work with FastEmbed/ONNX.
- **API-based**: VoyageCode3 (32k context, Matryoshka support, purpose-built
  for code).
- This is a config change (`CN_EMBEDDING_MODEL`) plus a one-time re-index.

### 6. Unbounded chunk sizes exceed embedding model context

**Defect**: Markdown section chunks have no upper size bound — a 2,000-line
section without subheaders becomes a single chunk. `all-MiniLM-L6-v2` has a
256–512 token context window; text beyond ~300 words is silently truncated by
the tokenizer, producing poor-quality embeddings for long chunks.

**Improvement**:

- Cap chunks at **512 tokens** (matching the embedding model's context window).
- Split oversized chunks at paragraph or subheader boundaries.
- Add **10–20% overlap** at chunk boundaries so that information at the edges
  isn't lost.

### 7. Markdown heading regex matches inside code fences

**Defect**: `extractor.py` line 33: `header_re = re.compile(r"^(#{1,6})\s+(.+)$")`
matches lines inside fenced code blocks (e.g., `# python comment` inside
` ```python ` blocks), creating false section breaks and corrupted chunks.

**Improvement**: Track fenced code block state (`inside_fence = not
inside_fence` when encountering `` ``` ``). Skip header matching while inside a
fence.

### 8. Hardcoded markdown doc_type bias in ranking

**Defect**: `index.py` line 256–257 gives all markdown chunks a `+0.04` score
boost regardless of query relevance. For code-focused questions, this
artificially ranks documentation above code docstrings that may be more relevant.

**Improvement**: Remove the blanket doc_type boost. If doc-type preference is
needed, make it query-dependent (e.g., boost markdown only if the query contains
terms like "documentation", "guide", "how to").

### 9. Judge infrastructure causes false failures

**Defect**: `runner.py` `llm_judge_answer()` catches `JSONDecodeError` and marks
the task as failed with no retry. Two of CN's four failures were caused by the
judge returning empty responses, not by actual answer quality problems.

**Improvement**:

- Add retry logic (1–2 retries with backoff) for judge API calls.
- Strip markdown code fences more robustly (handle `` ```json `` prefix).
- Log the raw judge response when parsing fails, for post-hoc analysis.

### 10. No cross-encoder reranking stage

**Defect**: Search results are ranked by a simple heuristic (cosine distance +
title keyword boost). There is no second-stage precision refinement.

**Improvement**: After initial retrieval (vector + BM25), apply a lightweight
cross-encoder reranker:

- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, ~60 ms per batch of 20).
- Evaluate query–chunk pairs jointly, which dramatically improves precision over
  bi-encoder dot-product similarity alone.
- Can be made optional (off by default for latency-sensitive interactive use, on
  for `cn ask` where quality matters more).

---

## Priority Order

| Priority | Item | Effort  | Expected Impact |
|----------|------|---------|-----------------|
| 🔴 P0    | 1. Index code structure, not just comments | 1–2 days | Fixes 18/25 "0 chunks" cases |
| 🔴 P0    | 2. Real hybrid search (BM25 + vector + RRF) | 1 day | Fixes exact-keyword misses |
| 🟠 P1    | 3. Fix `content_lower` dead code | 5 min | Free precision improvement |
| 🟠 P1    | 4. Skip empty/low-confidence seed injection | 15 min | Saves 1 wasted turn per question |
| 🟠 P1    | 7. Fix markdown heading regex in code fences | 15 min | Fixes corrupted markdown chunks |
| 🟠 P1    | 9. Judge retry logic | 30 min | Fixes 2 false failures |
| 🟡 P2    | 3. Inject directory tree into agent context | 30 min | Saves 2–4 tool calls per question |
| 🟡 P2    | 5. Upgrade embedding model | 2 hours | Better vector relevance across the board |
| 🟡 P2    | 6. Chunk size caps + overlap | 1 hour | Better embedding quality for long docs |
| 🟡 P2    | 8. Remove markdown doc_type bias | 5 min | Fairer ranking for code queries |
| 🟢 P3    | 10. Cross-encoder reranking | 1 day | Precision improvement on top of hybrid search |
| 🟢 P3    | 4. HyDE for seed search | 1 day | Better query-code semantic matching |

---

## What Won't Be Fixed by Search Improvements

Some eval cases are inherently hard for any retrieval system:

- **Call-chain tracing** ("how does X flow through Y to Z"): requires reading
  multiple connected files. CN's `call_tree` and `find_references` tools are the
  right approach; no amount of embedding improvement replaces multi-hop
  reasoning.
- **Cross-module architectural questions** spanning 5+ files: will always require
  multi-step agent reasoning regardless of search quality.
- **The baseline is strong**: grep + read_file + bash is exactly what a developer
  uses. For narrow questions, ripgrep is faster and more precise than semantic
  search. CN's value is in **reducing the number of turns** for broad,
  conceptual questions — not in replacing grep for exact string lookup.

---

## Execution Plan — Agent Model Distribution

Work is distributed across 4 sequential waves. Within each wave, all tasks run
in parallel. The coordinator sequences waves and re-runs the eval between
Wave 2 and Wave 3 to decide what's still needed.

### Wave 1 — Quick Wins (all independent, parallel)

| Fix | Model | Rationale |
|-----|-------|-----------|
| Fix `content_lower` dead code | **flash** | 2-line edit, no reasoning needed |
| Fix markdown regex in code fences | **flash** | Small, self-contained logic change |
| Remove markdown doc_type bias | **flash_lite** | Delete 2 lines |
| Judge retry logic | **flash** | Straightforward error-handling pattern |
| Skip empty seed injection | **flash** | Small conditional in `ask.py` |

5 agents, all flash/flash_lite, finish in minutes. No dependencies between them.

### Wave 2 — Core Architecture (2 parallel tracks)

| Fix | Model | Rationale |
|-----|-------|-----------|
| **Index code structure** (extractor rewrite) | **pro** | Multi-language AST/regex design, biggest change in the codebase, needs to reason about chunk boundaries, content formatting, and test coverage across 6 languages |
| **Real hybrid search** (BM25 + RRF) | **pro** | Needs to understand LanceDB FTS API, design the RRF merge, retune scoring weights, and write integration tests |
| Inject directory tree into agent context | **flash** | Can run in parallel — small change to `ask.py`, just needs to read the existing code |

The two pro agents are independent (one touches `extractor.py`, the other
`index.py`) so they run simultaneously.

### Wave 3 — Depends on Wave 2 results

| Fix | Model | Rationale |
|-----|-------|-----------|
| Chunk size caps + overlap | **flash** | Mechanical once the extractor structure from Wave 2 is settled — just add a splitter |
| Upgrade embedding model | **flash** | Config change + re-index, but should come *after* the new extractor so you only re-index once |

### Wave 4 — Advanced (optional, after re-eval)

| Fix | Model | Rationale |
|-----|-------|-----------|
| Cross-encoder reranking | **pro** | New pipeline stage, new dependency, needs careful latency/quality tradeoff design |
| HyDE for seed search | **pro** | LLM-in-the-loop retrieval, prompt engineering, needs to integrate with existing `ask.py` flow |

These should only start after running the eval again post-Wave 2 to see if
they're still needed.

### Model Summary

| Model | Tasks | Why |
|-------|-------|-----|
| **flash_lite** | 1 trivial deletion | Cheapest possible |
| **flash** | 6 targeted fixes | Small, well-scoped, one-file changes |
| **pro** | 4 architectural changes | Multi-file reasoning, API design, cross-language logic |

Total: ~8 concurrent agents at peak (Wave 1), dropping to 3 in Wave 2. The
coordinator (at inherit/opus level) manages sequencing between waves and runs
the eval between Wave 2 and Wave 3 to decide what's still needed.
