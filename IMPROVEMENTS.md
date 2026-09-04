# Improvements — status and findings

Driven by measurement of the A/B eval run
`eval/runs/run_20260904_044459_153370` (cn 0.3.64, 25 tasks, 25/25 pass in both
arms). Supersedes the earlier plan written against
`report_20260903_110303.json`; several of that plan's assumptions turned out to
be wrong when measured, and those are recorded below rather than deleted.

---

## The headline number was an artifact

| | raw tokens | net tokens | wall clock |
|---|---|---|---|
| All 25 tasks | **+15.8%** | +36.2% | +6.7% |
| Excluding `vikunja-task-reminders` | **−17.5%** | −16.8% | +7.2% |

cn was cheaper on 10/25 tasks and more expensive on 15/25; the median per-task
token outcome was **−26%**. One task supplied +27.0 percentage points of the
+15.8%.

Root cause: in that task the baseline ran
`rg -i "…|func.*Cron" pkg`. One matched line in vikunja is **486,303
characters** (generated/minified). The baseline harness capped grep output at
200 *lines* but never at *bytes*, so one tool result was 717KB ≈ 179k tokens and
was re-sent on every subsequent turn. cn survived only because its cap is 30
matches rather than 200 lines — a 13× difference in exposure to the same
missing bound, not a semantic win.

**Therefore: treat "cn currently costs ~17% more tokens than a plain grep+read
agent" as the number to improve against.**

---

## Measured retrieval results

Recall over the 25 conceptual benchmark questions, gold = `required_files`:

| configuration | recall@1 | recall@3 | recall@10 | MRR |
|---|---|---|---|---|
| baseline (cn 0.3.64) | 10/25 | 15/25 | 20/25 | 0.505 |
| + code-first ordering | 17/25 | 19/25 | 22/25 | 0.736 |
| + real RRF | 18/25 | 20/25 | 21/25 | 0.757 |
| + per-file diversity cap | 18/25 | 20/25 | **23/25** | **0.784** |

Live through the production `execute_search` path the shipped stack measures
recall@1 16/25, recall@3 20/25, recall@5 23/25, recall@10 24/25.

---

## Shipped

| # | Change | Evidence |
|---|---|---|
| 1 | Byte-cap grep matches (240 chars) in **both** arms | pathological vikunja query: baseline 179k → 0.7k tokens; cn 13.5k → 1.2k |
| 2 | Code-first seed ordering, skipped for doc-seeking queries | FastAPI indexes 15,839 markdown vs 5,689 code chunks; 8.6/10 unfiltered hits were prose. MRR 0.505 → 0.736 |
| 3 | Real RRF (drop `max(cosine, rrf)`, drop `min(0.99)` clamp) | the `max()` meant BM25 was silently discarded; 7/25 questions showed all top-5 at "99%" |
| 4 | Per-file diversity cap | 4.8 → 9.7 distinct files per 10 hits; MRR 0.757 → 0.784 |
| 5 | Batched `read_code(ranges=[...])` | 150 read calls hit 75 distinct files — half were re-reads; tokens correlate with turn count at r = 0.887 |
| 6 | Deterministic question router + `CN_SEED_MODE` | seed costs ~1,699 tokens, re-sent every turn (~13.6k over 8 turns); only 2.8/10 seeded chunks came from a file the agent opened |
| 7 | Drop `call_tree` from the default tool spec | 0 uses across 25 tasks (`tags_lookup` 0, `find_references` 2, `search` **2**) |
| 8 | 7 lookup tasks added to the benchmark | the suite had none, so the router was unmeasurable |

---

## Measured and rejected

### Token-aware chunk splitting

The defect is real and large: fastembed's `all-MiniLM-L6-v2` truncates at
**128 tokens** — not the documented 256/512 — so **73.3% of indexed chunks
overflow and 61.3% of indexed content never reaches the encoder**
(uv 76.6%/61.9%, vikunja 69.3%/66.0%).

Splitting chunks to fit recovers 100% of that content and measured **worse**:

| index | recall@1 | MRR |
|---|---|---|
| line-capped chunks (current) | 9/14 | **0.717** |
| token-capped chunks | 6/14 | 0.561 |
| token-capped + diversity cap | 6/14 | 0.608 |

A code chunk's head — signature plus docstring, exactly what survives
truncation today — carries most of the identifying signal; body fragments are
low-signal near-duplicates that crowd distinct files out of the top-k (distinct
files per 10 hits fell 4.3 → 3.5). The line cap is *accidentally right* for a
128-token encoder.

`split_oversize_chunks()` is kept and tested behind
`CN_SPLIT_OVERSIZE_CHUNKS=1`, because it becomes the correct behaviour with a
long-context code embedding model. Index growth is ~3–6× (flask 3.6×,
httpx 2.9×, uv ~5.7×).

### Retrieval quality is not the turn-count bottleneck

Bucketing the run by where the seed ranked the gold file:

| seed rank of gold file | n | mean cn API calls |
|---|---|---|
| 1 (top hit) | 10 | 8.1 |
| 2–3 | 5 | 9.0 |
| 4–10 | 5 | 7.4 |
| missed entirely | 5 | 8.8 |

The agent takes ~8 turns whether retrieval nailed it or missed completely.
Retrieval quality and turn count are separate problems, and turn count is what
costs money — hence items 5 and 6 above.

---

## Still open

| Priority | Item | Why |
|---|---|---|
| 🔴 | Re-run the A/B with capped greps and confirm the honest baseline | every number above except this one is measured; the end-to-end token/time effect is not |
| 🟠 | Long-context code embedding model (`nomic-embed-code`, VoyageCode3) | the only real fix for 61% content loss; unlocks chunk splitting |
| 🟠 | Benchmark has no accuracy headroom (25/25 both arms) | add harder multi-hop tasks, or the suite can only measure cost |
| 🟡 | Encourage parallel tool calls in one turn | cn issues 1.51 tools per tool-calling turn vs baseline 1.63 |
| 🟡 | Cross-encoder reranking | precision on top of the fixed ranking |
| 🟢 | HyDE for seed search | bridges conversational queries to code identifiers |

### Carried over from the previous plan, still true

- Markdown heading regex vs fenced code blocks — **fixed** (fence tracking is in
  `extract_markdown`).
- `content_lower` dead code — **fixed** (content matches now boost).
- Judge retry logic — verify it survived; two false failures were traced to it.
