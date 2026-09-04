#!/usr/bin/env python3
"""Evaluate the pre-flight routing heuristic against a wide, independently-labelled query set.

`route_question()` is a hand-written heuristic, and its only validation so far
was the 32 benchmark questions -- which were written by the same author as the
heuristic. That is circular: it measures whether the regex matches the sentences
the regex was written for.

This harness breaks the circularity two ways:

1. **Generation** -- queries come from an LLM told only what a developer asks a
   code-navigation tool, never which features the heuristic looks at, and asked
   for adversarial/ambiguous phrasings on purpose.
2. **Labelling** -- a second LLM pass labels each query from the *task*
   definition ("would pre-flight semantic retrieval earn its cost here?"),
   never from the heuristic's features, and never sees the heuristic's answer.

Agreement between the two is then a real measure. Disagreements are printed in
full, because on a decision this cheap the interesting output is the failure
list, not the headline percentage.

Usage:
    python eval/router_eval.py --generate 240      # generate + label + score
    python eval/router_eval.py --queries FILE      # re-score a saved set
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from codebase_navigator.ask import (
    call_chat_completions,
    load_llm_config,
    route_question,
)

DEFAULT_QUERIES_PATH = Path(__file__).parent / "router_queries.json"

# Deliberately says nothing about underscores, capitalisation, question stems, or
# any other feature the heuristic keys on.
GENERATION_PROMPT = """You are helping build a test set for a code-search tool.

Write {n} distinct questions a working developer would type at a tool that answers
questions about an unfamiliar {language} codebase ({repo_hint}).

CRITICAL - these must read like something typed into a terminal prompt:
- NO backticks, NO markdown, NO quotation marks around identifiers
- write identifiers bare, exactly as someone types them: parse_config not `parse_config`
- many should be all lowercase, with no question mark
- some should have typos or inconsistent capitalisation

Use real {language} vocabulary and naming conventions ({naming}), and refer to
things that actually exist in software of this kind.

Cover the full realistic range:
- questions naming one specific function, type, constant or file
- questions about how a mechanism, flow or subsystem works
- questions that name something specific but really ask how it behaves
- comparisons between two named things
- vague one-word or two-word questions
- long rambling questions with several clauses
- questions phrased as commands rather than questions
- questions about configuration, tests, build files and documentation

Vary phrasing heavily. Do not number them. Do not explain them.
Return ONLY a JSON array of strings."""

# Defines the decision by its COST, not by sentence shape, so the labeller is
# reasoning about the same tradeoff the router faces rather than pattern-matching.
LABEL_PROMPT = """A code-navigation agent must decide, before answering, whether to run an
expensive pre-flight semantic (embedding) search over the repository and paste the
top results into its first prompt.

That pre-flight dump costs roughly 1700 tokens and is re-sent on every subsequent
turn of the session, so over a typical session it costs well over 10000 tokens.

The agent ALWAYS has these alternatives available at no up-front cost:
- exact symbol lookup in a ctags index
- ripgrep literal/regex search
- reading specific file line ranges
- calling the semantic search itself later, if it turns out to need it

Answer LOOKUP if the question names a specific identifier or file concrete enough
that an exact-name lookup or a literal grep would land on the answer directly, so
the pre-flight dump would be wasted.

Answer CONCEPTUAL if answering needs understanding of a mechanism, flow,
relationship or design spread across code, where ranked semantic context would
genuinely save the agent turns.

Question: {question}

Reply with exactly one word: LOOKUP or CONCEPTUAL"""

# Weighted toward Python, Go and Rust. The first pass drifted heavily to
# JavaScript/Express (77 of 211 queries against 1 for Rust and 0 for Go), which
# made the agreement figure a measure of one language's phrasing conventions.
REPO_SPECS = [
    ("Python", "a web framework such as Flask", "snake_case functions, PascalCase classes"),
    ("Python", "an async HTTP client such as httpx", "snake_case, dunder methods, async def"),
    ("Python", "an API framework such as FastAPI", "snake_case, decorators, type hints"),
    ("Go", "a task-tracking web service", "MixedCaps funcs, exported Capitalised names, err vars"),
    ("Go", "a container orchestration daemon", "interfaces, struct receivers, ctx params"),
    ("Rust", "a package manager such as uv", "snake_case fns, PascalCase structs/traits, crates"),
    ("Rust", "an async runtime such as tokio", "modules, impl blocks, lifetimes, Result types"),
    ("JavaScript", "a web framework such as Express", "camelCase, middleware, module.exports"),
]


def _chat(config, prompt: str, max_tokens: int = 2048) -> str:
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": max_tokens,
    }
    data = call_chat_completions(config.endpoint, config.api_key, payload)
    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content") or ""


def _parse_json_array(text: str) -> list[str]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # Fall back to line scraping so one malformed batch does not lose the run.
    out = []
    for line in text.splitlines():
        line = line.strip().strip(",").strip('"').strip()
        line = re.sub(r"^[-*\d.)\s]+", "", line)
        if len(line) > 8 and not line.startswith(("{", "[", "}")):
            out.append(line)
    return out


def generate_queries(config, total: int) -> list[tuple[str, str]]:
    """Return (question, language) pairs so the language balance is auditable."""
    per_repo = max(1, total // len(REPO_SPECS))
    collected: list[tuple[str, str]] = []

    def one(spec: tuple[str, str, str]) -> list[tuple[str, str]]:
        language, hint, naming = spec
        try:
            batch = _parse_json_array(
                _chat(
                    config,
                    GENERATION_PROMPT.format(
                        n=per_repo, language=language, repo_hint=hint, naming=naming
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  generation failed for {language}/{hint}: {exc}", file=sys.stderr)
            return []
        return [(q, language) for q in batch]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for batch in pool.map(one, REPO_SPECS):
            collected.extend(batch)

    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for q, lang in collected:
        # Strip any markdown the model added despite instructions, so the set
        # reflects how a person actually types rather than how an LLM formats.
        q = q.replace("`", "").strip()
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append((q, lang))
    return deduped


def label_queries(config, queries: list[str]) -> list[str | None]:
    def one(q: str) -> str | None:
        try:
            # Generous budget: the candidate model reasons internally before
            # emitting content, and a tight cap returns an empty string (or a
            # truncated "LOOK") for every single query.
            reply = _chat(config, LABEL_PROMPT.format(question=q), max_tokens=384)
        except Exception:  # noqa: BLE001
            return None
        upper = reply.strip().upper()
        # Check CONCEPTUAL first: it is the longer word and a truncated reply can
        # contain neither, so order matters more than membership.
        if "CONCEPTUAL" in upper:
            return "conceptual"
        if "LOOKUP" in upper or upper.endswith("LOOK"):
            return "lookup"
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, queries))


def score(
    queries: list[str],
    labels: list[str | None],
    languages: list[str] | None = None,
) -> dict[str, Any]:
    languages = languages or ["?"] * len(queries)
    triples = [(q, ll, lg) for q, ll, lg in zip(queries, labels, languages) if ll]
    by_lang: dict[str, list[bool]] = {}
    for q, truth, lg in triples:
        by_lang.setdefault(lg, []).append(route_question(q) == truth)
    pairs = [(q, ll) for q, ll, _ in triples]
    preds = [route_question(q) for q, _ in pairs]

    cm = Counter((truth, pred) for (_, truth), pred in zip(pairs, preds))
    n = len(pairs)
    correct = sum(v for (t, p), v in cm.items() if t == p)

    def prf(cls: str) -> tuple[float, float, int]:
        tp = cm[(cls, cls)]
        fp = sum(v for (t, p), v in cm.items() if p == cls and t != cls)
        fn = sum(v for (t, p), v in cm.items() if t == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return prec, rec, tp + fn

    disagreements = [
        {"question": q, "labelled": truth, "heuristic": pred}
        for (q, truth), pred in zip(pairs, preds)
        if truth != pred
    ]

    return {
        "per_language": {lg: (sum(v), len(v)) for lg, v in sorted(by_lang.items())},
        "n_labelled": n,
        "n_unlabelled": len(queries) - n,
        "accuracy": correct / n if n else 0.0,
        "confusion": {f"{t}->{p}": v for (t, p), v in sorted(cm.items())},
        "lookup": prf("lookup"),
        "conceptual": prf("conceptual"),
        "disagreements": disagreements,
    }


def report(res: dict[str, Any]) -> None:
    print(f"\n{'=' * 72}")
    print("Routing heuristic vs independent LLM labels")
    print(f"{'=' * 72}")
    print(f"  labelled queries      {res['n_labelled']}  (unlabelled: {res['n_unlabelled']})")
    print(f"  agreement             {res['accuracy']:.1%}")
    for cls in ("lookup", "conceptual"):
        p, r, support = res[cls]
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  {cls:11}  precision {p:.2f}  recall {r:.2f}  F1 {f1:.2f}  (n={support})")
    if res.get("per_language"):
        print("\n  agreement by language:")
        for lg, (ok, tot) in res["per_language"].items():
            print(f"    {lg:12} {ok:3}/{tot:<3} {100 * ok / tot:5.1f}%")
    print("\n  confusion (label -> heuristic):")
    for k, v in res["confusion"].items():
        print(f"    {k:28} {v}")

    dis = res["disagreements"]
    if dis:
        # The cost of the two error directions is not symmetric, so split them.
        missed = [d for d in dis if d["labelled"] == "conceptual"]
        wasted = [d for d in dis if d["labelled"] == "lookup"]
        print(f"\n  MISSED RETRIEVAL ({len(missed)}) - routed lookup, wanted context")
        print("    cost: agent may spend an extra turn, can still call `search`")
        for d in missed[:15]:
            print(f"    - {d['question'][:96]}")
        print(f"\n  WASTED SEED ({len(wasted)}) - routed conceptual, grep would have done")
        print("    cost: ~1700 tokens, re-sent every turn")
        for d in wasted[:15]:
            print(f"    - {d['question'][:96]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", type=int, default=0, help="Generate N queries with the LLM")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--relabel", action="store_true", help="Re-label an existing query set in place"
    )
    args = ap.parse_args()

    config = load_llm_config(folder=PROJECT_DIR)
    if not config.api_key:
        print("No API key found; set OPENROUTER_API_KEY or configure api_key.", file=sys.stderr)
        return 1

    if args.generate:
        print(f"Generating ~{args.generate} queries with {config.model} ...")
        pairs = generate_queries(config, args.generate)
        queries = [q for q, _ in pairs]
        langs = [ll for _, ll in pairs]
        print(f"  generated {len(queries)} unique queries")
        by_lang = Counter(langs)
        print(f"  language balance: {dict(by_lang)}")
        backticked = sum(1 for q in queries if "`" in q)
        print(f"  containing backticks: {backticked} (should be 0)")
        print("Labelling with an independent prompt ...")
        labels = label_queries(config, queries)
        payload = [
            {"question": q, "language": lang, "label": ll}
            for q, lang, ll in zip(queries, langs, labels)
        ]
        out = args.out or args.queries
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"  saved to {out}")
    else:
        if not args.queries.is_file():
            print(f"No query set at {args.queries}; run with --generate N", file=sys.stderr)
            return 1
        payload = json.loads(args.queries.read_text(encoding="utf-8"))
        queries = [x["question"] for x in payload]
        labels = [x.get("label") for x in payload]
        languages = [x.get("language", "?") for x in payload]
        if args.relabel or not any(labels):
            print(f"Labelling {len(queries)} queries ...")
            labels = label_queries(config, queries)
            out = args.out or args.queries
            out.write_text(
                json.dumps(
                    [
                        {"question": q, "language": lg, "label": ll}
                        for q, lg, ll in zip(queries, languages, labels)
                    ],
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(f"  saved to {out}")

    if args.generate:
        languages = langs
    report(score(queries, labels, languages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
