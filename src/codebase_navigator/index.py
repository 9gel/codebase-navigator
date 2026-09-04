"""LanceDB vector index management and hybrid semantic search."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lancedb

if TYPE_CHECKING:
    from fastembed import TextEmbedding

from lancedb.index import FTS

from .config import (
    DOC_SCHEMA,
    EMBEDDING_MODEL_NAME,
    VECTOR_DIM,
    get_cache_dir,
    silence_stdio,
)
from .extractor import DocExtractor
from .tags import get_available_files

COMMON_STOPWORDS = {
    "what",
    "is",
    "a",
    "an",
    "the",
    "in",
    "on",
    "of",
    "for",
    "to",
    "and",
    "or",
    "how",
    "why",
    "where",
    "which",
    "does",
    "do",
    "can",
}

# Queries that are genuinely asking about prose -- a glossary term, a guide, a
# README section. Code-first ordering must not demote documentation for these,
# or "what is a Retail Destination" returns retail.py instead of GLOSSARY.md.
DOC_SEEKING_RE = re.compile(
    r"\b(what\s+(?:is|are)\s+(?:a|an|the)\b"
    r"|what\s+does\s+\w+\s+mean"
    r"|definition\s+of|define\b|glossary|terminology|nomenclature"
    r"|documentation|readme|changelog|release\s+notes"
    r"|contributing|licence|license"
    r"|getting\s+started|tutorial|guide\b)",
    re.IGNORECASE,
)


# See VectorIndex.split_oversize_chunks: splitting to fit the encoder window is
# a measured regression with a 128-token model, and a prerequisite with a
# long-context one.
SPLIT_OVERSIZE_CHUNKS = os.environ.get("CN_SPLIT_OVERSIZE_CHUNKS", "").lower() in {
    "1",
    "true",
    "yes",
}

# Retrieval returned 4.8 distinct files per 10 hits because several chunks of one
# file monopolised the list. Capping chunks-per-file raised that to 9.7 and lifted
# MRR 0.757 -> 0.784 (recall@10 21 -> 23 of 25) at no cost.
MAX_CHUNKS_PER_FILE = 1


def is_doc_seeking(query: str) -> bool:
    """True when the query asks about documentation rather than implementation."""
    return bool(DOC_SEEKING_RE.search(query or ""))


_SHARED_MODEL: TextEmbedding | None = None
_SHARED_MODEL_LOCK = threading.Lock()
_EMBEDDING_INFERENCE_LOCK = threading.Lock()


class IndexModelMismatch(RuntimeError):
    """Raised when an existing index was built with a different embedding model."""


class VectorIndex:
    """LanceDB index manager with FastEmbed ONNX embeddings and hybrid re-ranking."""

    def __init__(self, folder: Path, custom_index_dir: str | None = None):
        self.folder = folder
        self.cache_dir = get_cache_dir(folder, custom_index_dir)
        self.db_dir = self.cache_dir / "lancedb"
        self.meta_file = self.cache_dir / "files_meta.json"
        self.db = lancedb.connect(str(self.db_dir))
        self._model: TextEmbedding | None = None
        self._max_tokens: int | None = None
        self._count_tok: Any = None
        self._ensure_table()

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            global _SHARED_MODEL
            if _SHARED_MODEL is None:
                with _SHARED_MODEL_LOCK:
                    if _SHARED_MODEL is None:
                        from fastembed import TextEmbedding

                        with silence_stdio():
                            _SHARED_MODEL = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
            self._model = _SHARED_MODEL
        return self._model

    @property
    def embed_max_tokens(self) -> int:
        """Hard token ceiling the embedding model truncates at.

        fastembed's all-MiniLM-L6-v2 build truncates at 128 tokens, not the 256
        or 512 the model card implies. Chunks were capped by *line count* only
        (50 lines Python, 35 generic), so 73% of indexed chunks overflowed and
        61% of all indexed content never reached the encoder at all.
        """
        if self._max_tokens is None:
            limit = 128
            try:
                trunc = getattr(self.model.model.tokenizer, "truncation", None)
                if isinstance(trunc, dict) and trunc.get("max_length"):
                    limit = int(trunc["max_length"])
            except Exception:
                pass
            self._max_tokens = limit
        return self._max_tokens

    @property
    def _counting_tokenizer(self):
        """A truncation-free clone of the encoder's tokenizer, for measuring length.

        The live tokenizer has truncation enabled at max_length, so encoding a
        3,000-token chunk with it returns 128 and every chunk looks compliant.
        Clone it and disable truncation on the copy -- mutating the shared one
        would change what the ONNX model actually receives.
        """
        if self._count_tok is not None:
            return self._count_tok
        try:
            from tokenizers import Tokenizer

            src = self.model.model.tokenizer
            clone = Tokenizer.from_str(src.to_str())
            # Both must go: truncation clamps long chunks to max_length, and
            # padding inflates every short chunk to max_length. With either left
            # on, encode() returns exactly max_length for all input and the
            # length check is meaningless.
            clone.no_truncation()
            clone.no_padding()
            self._count_tok = clone
        except Exception:
            self._count_tok = False
        return self._count_tok

    def _count_tokens(self, text: str) -> int:
        tok = self._counting_tokenizer
        if tok:
            try:
                return len(tok.encode(text).ids)
            except Exception:
                pass
        # WordPiece averages ~2.75 chars/token on code (measured across the
        # six benchmark repos); 2.6 keeps the estimate slightly conservative.
        return int(len(text) / 2.6) + 1

    def split_oversize_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split chunks that exceed the encoder window into line-aligned windows.

        OFF by default -- enable with CN_SPLIT_OVERSIZE_CHUNKS=1.

        all-MiniLM-L6-v2 truncates at 128 tokens, so 73% of indexed chunks
        overflow and 61% of indexed content never reaches the encoder. Splitting
        recovers 100% of that content, but measured on the benchmark it makes
        retrieval *worse*: MRR fell 0.717 -> 0.561 over flask/express/httpx, and
        stayed behind (0.608) even with the per-file diversity cap applied. The
        reason is that a code chunk's head -- signature plus docstring, which is
        exactly what survives truncation today -- carries most of the identifying
        signal, while body fragments are low-signal near-duplicates that crowd
        distinct files out of the top-k.

        So the line cap is accidentally right for a 128-token encoder. This
        becomes the correct behaviour when swapping in a long-context code
        embedding model (nomic-embed-code and similar allow 8k), where there is
        no truncation to hide behind; hence it is kept and tested, not deleted.
        """
        cap = max(32, int(self.embed_max_tokens * 0.85))  # headroom for special tokens
        out: list[dict[str, Any]] = []
        tok = self._counting_tokenizer

        def count(text: str) -> int:
            if tok:
                try:
                    return max(1, len(tok.encode(text).ids))
                except Exception:
                    pass
            return max(1, int(len(text) / 2.6) + 1)

        for ch in chunks:
            content = ch.get("content", "")
            # Fast path: at >=1 char per token this cannot overflow.
            if len(content) <= cap or self._count_tokens(content) <= cap:
                out.append(ch)
                continue

            lines = content.split("\n")
            has_header = bool(lines and (lines[0].startswith("File:") or lines[0].startswith("#")))
            header = lines[0] if has_header else ""
            body = lines[1:] if has_header else lines
            header_t = self._count_tokens(header) if header else 0
            budget = max(16, cap - header_t)

            # A single minified line can exceed the whole budget on its own, so
            # hard-split any such line on character boundaries before windowing.
            units: list[tuple[str, int]] = []
            for ln in body:
                lt = count(ln)
                if lt <= budget:
                    units.append((ln, lt))
                    continue
                approx_chars = max(1, int(len(ln) * budget / lt))
                for i in range(0, len(ln), approx_chars):
                    piece = ln[i : i + approx_chars]
                    units.append((piece, count(piece)))

            windows: list[list[str]] = []
            cur: list[str] = []
            cur_t = 0
            for text, lt in units:
                if cur and cur_t + lt > budget:
                    windows.append(cur)
                    # Carry a little context forward, but only while it leaves
                    # room for real content -- an oversized carry would flush
                    # immediately and spin out empty windows.
                    carry: list[str] = []
                    carry_t = 0
                    for prev in reversed(cur):
                        pt = count(prev)
                        if carry_t + pt > budget // 4:
                            break
                        carry.insert(0, prev)
                        carry_t += pt
                    cur = carry
                    cur_t = carry_t
                cur.append(text)
                cur_t += lt
            if cur:
                windows.append(cur)

            if len(windows) <= 1:
                out.append(ch)
                continue

            for i, win in enumerate(windows):
                sub = dict(ch)
                sub["content"] = "\n".join(([header] + win) if header else win)
                sub["title"] = (
                    ch.get("title", "") if i == 0 else f"{ch.get('title', '')} (part {i + 1})"
                )
                sub["id"] = hashlib.sha256(
                    f"{ch.get('path')}:{ch.get('start_line')}:{ch.get('title')}:{i}".encode()
                ).hexdigest()[:16]
                out.append(sub)

        return out

    def _embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[Any]:
        """Run the shared ONNX model without concurrent-session contention.

        Batches are formed from length-sorted input. The tokenizer pads to the
        longest sequence *in each batch*, so with a long-context encoder a single
        8k-token chunk forces every other chunk in its batch to be padded to 8k
        and burns compute on padding. Sorting groups similar lengths together;
        results are restored to the caller's order, and embeddings are per-item
        so the reordering cannot change them.
        """
        if progress_callback:
            progress_callback("Waiting for shared embedding model...")

        if len(texts) < 2:
            with _EMBEDDING_INFERENCE_LOCK:
                if progress_callback:
                    progress_callback("Embedding search query...")
                kwargs = {"batch_size": batch_size} if batch_size is not None else {}
                return list(self.model.embed(texts, **kwargs))

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ordered = [texts[i] for i in order]

        with _EMBEDDING_INFERENCE_LOCK:
            if progress_callback:
                progress_callback("Embedding search query...")
            kwargs = {"batch_size": batch_size} if batch_size is not None else {}
            embedded = list(self.model.embed(ordered, **kwargs))

        restored: list[Any] = [None] * len(texts)
        for slot, vec in zip(order, embedded):
            restored[slot] = vec
        return restored

    def _ensure_table(self):
        try:
            self.table = self.db.open_table("documents")
        except Exception:
            self.table = self.db.create_table("documents", schema=DOC_SCHEMA, mode="create")
            return

        # An index built with a different embedding model has a different vector
        # width, and LanceDB reports that as an opaque "no vector column" error at
        # query time. Detect it here and say what actually needs to happen.
        try:
            field = self.table.schema.field("vector")
            existing_dim = getattr(field.type, "list_size", None)
        except Exception:
            existing_dim = None

        if existing_dim and existing_dim != VECTOR_DIM:
            raise IndexModelMismatch(
                f"Index at {self.db_dir} was built with a {existing_dim}-dimensional "
                f"embedding model, but the configured model {EMBEDDING_MODEL_NAME} "
                f"produces {VECTOR_DIM} dimensions.\n"
                f"Re-index with `cn sync --force`, or pin the previous model via "
                f"CN_EMBEDDING_MODEL."
            )

    def _ensure_fts_index(self):
        """Create or update full-text search index on content column if rows exist."""
        try:
            if len(self.table) > 0:
                self.table.create_index("content", config=FTS(), replace=True)
        except Exception:
            pass

    def load_meta(self) -> dict[str, dict[str, Any]]:
        if self.meta_file.exists():
            try:
                return json.loads(self.meta_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save_meta(self, meta: dict[str, dict[str, Any]]):
        self.meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def sync(self, force: bool = False) -> tuple[int, int, int]:
        """Incrementally sync all available files into LanceDB.

        Returns (files_updated, chunks_indexed, files_pruned).
        """
        extractor = DocExtractor(self.folder)
        code_files, doc_files = get_available_files(self.folder)
        all_files = code_files + doc_files

        meta = {} if force else self.load_meta()
        current_rel_paths: set[str] = set()

        files_to_index: list[tuple[Path, list[dict[str, Any]]]] = []
        all_chunks_to_embed: list[dict[str, Any]] = []

        for fpath in all_files:
            try:
                rel_p = str(fpath.relative_to(self.folder))
            except ValueError:
                rel_p = str(fpath)
            current_rel_paths.add(rel_p)

            stat = fpath.stat()
            last_meta = meta.get(rel_p)
            if (
                not force
                and last_meta
                and last_meta.get("mtime") == stat.st_mtime
                and last_meta.get("size") == stat.st_size
            ):
                continue

            if fpath.suffix.lower() in {".md", ".markdown", ".rst", ".adoc", ".org"}:
                chunks = extractor.extract_markdown(fpath)
            else:
                chunks = extractor.extract_code_doc(fpath)

            if chunks:
                if SPLIT_OVERSIZE_CHUNKS:
                    chunks = self.split_oversize_chunks(chunks)
                files_to_index.append((fpath, chunks))
                all_chunks_to_embed.extend(chunks)

        # Prune deleted files
        deleted_paths = [p for p in meta.keys() if p not in current_rel_paths]
        if deleted_paths:
            for p in deleted_paths:
                try:
                    escaped_p = p.replace('"', '\\"')
                    self.table.delete(f'path = "{escaped_p}"')
                except Exception:
                    pass
                meta.pop(p, None)

        if force:
            try:
                self.db.drop_table("documents")
            except Exception:
                pass
            self.table = self.db.create_table("documents", schema=DOC_SCHEMA, mode="create")
        else:
            for fpath, _ in files_to_index:
                try:
                    rel_p = str(fpath.relative_to(self.folder))
                except ValueError:
                    rel_p = str(fpath)
                escaped_p = rel_p.replace('"', '\\"')
                try:
                    self.table.delete(f'path = "{escaped_p}"')
                except Exception:
                    pass

        if all_chunks_to_embed:
            texts = [c["content"] for c in all_chunks_to_embed]
            embeddings = self._embed(texts, batch_size=256)
            for chunk, vec in zip(all_chunks_to_embed, embeddings):
                chunk["vector"] = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            self.table.add(all_chunks_to_embed)

        if files_to_index or force:
            self._ensure_fts_index()

        for fpath, chunks in files_to_index:
            try:
                rel_p = str(fpath.relative_to(self.folder))
            except ValueError:
                rel_p = str(fpath)
            stat = fpath.stat()
            meta[rel_p] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "chunks": len(chunks),
            }

        self.save_meta(meta)
        return (len(files_to_index), len(all_chunks_to_embed), len(deleted_paths))

    def update_single_file(self, fpath: Path) -> int:
        """Incrementally update one file in LanceDB."""
        try:
            rel_p = str(fpath.relative_to(self.folder))
        except ValueError:
            rel_p = str(fpath)

        escaped_p = rel_p.replace('"', '\\"')
        try:
            self.table.delete(f'path = "{escaped_p}"')
        except Exception:
            pass

        if not fpath.exists():
            meta = self.load_meta()
            meta.pop(rel_p, None)
            self.save_meta(meta)
            self._ensure_fts_index()
            return 0

        extractor = DocExtractor(self.folder)
        if fpath.suffix.lower() in {".md", ".markdown", ".rst", ".adoc", ".org"}:
            chunks = extractor.extract_markdown(fpath)
        else:
            chunks = extractor.extract_code_doc(fpath)

        if chunks:
            if SPLIT_OVERSIZE_CHUNKS:
                chunks = self.split_oversize_chunks(chunks)
            texts = [c["content"] for c in chunks]
            embeddings = self._embed(texts, batch_size=64)
            for chunk, vec in zip(chunks, embeddings):
                chunk["vector"] = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            self.table.add(chunks)

        self._ensure_fts_index()

        meta = self.load_meta()
        stat = fpath.stat()
        meta[rel_p] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "chunks": len(chunks),
        }
        self.save_meta(meta)
        return len(chunks)

    def search(
        self,
        query: str,
        limit: int = 5,
        doc_type: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        code_first: bool = True,
        max_per_file: int | None = MAX_CHUNKS_PER_FILE,
    ) -> list[dict[str, Any]]:
        """True hybrid semantic vector + BM25 FTS search with Reciprocal Rank Fusion."""
        if len(self.table) == 0:
            return []

        q_vec = self._embed([query], progress_callback=progress_callback)[0]
        if hasattr(q_vec, "tolist"):
            q_vec = q_vec.tolist()
        else:
            q_vec = list(q_vec)
        # The per-file diversity cap discards candidates, so the pool must be
        # deeper than `limit` or the tail of the result list comes up short.
        fetch_limit = max(limit * 6, 40)
        if progress_callback:
            progress_callback("Querying LanceDB index...")

        # Normalize doc_type filter
        norm_type = None
        if doc_type and doc_type != "all":
            norm_type = (
                "markdown"
                if doc_type in ["md", "markdown"]
                else "code_doc"
                if doc_type in ["code", "code_doc"]
                else doc_type
            )

        # 1. Vector Search
        search_query = self.table.search(q_vec).metric("cosine").limit(fetch_limit)
        if norm_type:
            search_query = search_query.where(f'doc_type = "{norm_type}"')

        try:
            vector_results = search_query.to_list()
        except Exception:
            vector_results = []

        # 2. BM25 / FTS Search
        clean_terms = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]+", query)
            if len(w) >= 2 and w.lower() not in COMMON_STOPWORDS
        ]
        clean_phrase = " ".join(clean_terms)

        fts_results: list[dict[str, Any]] = []
        if clean_terms:
            fts_query_str = clean_phrase if clean_phrase else " ".join(clean_terms)
            try:
                fts_query = self.table.search(fts_query_str, query_type="fts").limit(fetch_limit)
                if norm_type:
                    fts_query = fts_query.where(f'doc_type = "{norm_type}"')
                fts_results = fts_query.to_list()
            except Exception:
                fts_results = []

        # 3. Reciprocal Rank Fusion. k=20 rather than the textbook 60: with only a few
        # dozen candidates a large k flattens the rank curve so the top hit barely
        # outscores the tail. FTS is weighted slightly higher because code questions
        # usually name a real identifier.
        k_rrf = 20
        w_fts = 1.2
        merged_candidates: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = {}

        for rank, r in enumerate(vector_results):
            cid = r.get("id") or f"{r.get('path')}:{r.get('start_line')}"
            merged_candidates[cid] = r
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank + 1))

        for rank, r in enumerate(fts_results):
            cid = r.get("id") or f"{r.get('path')}:{r.get('start_line')}"
            if cid not in merged_candidates:
                merged_candidates[cid] = r
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_fts / (k_rrf + rank + 1))

        if not merged_candidates:
            return []

        # Normalization factor: best possible score is rank 0 in both result lists.
        max_rrf = (1.0 + w_fts) / (k_rrf + 1)

        # Identifier-aware boosts are proportional to the share of query terms matched,
        # so a chunk matching 3/3 terms outranks one matching 1/3 instead of both
        # colliding at the old 0.99 ceiling.
        n_terms = len(clean_terms) or 1

        scored_results: list[dict[str, Any]] = []
        for cid, r in merged_candidates.items():
            # Rank fusion is the ranking signal. The old code took
            # max(vec_base, norm_rrf * 0.90); because raw cosine proximity sits at
            # 0.7-0.95 for almost any pair while norm_rrf tops out at 0.90, the
            # vector term always won and the BM25 half of the "hybrid" search was
            # silently discarded. RRF is rank-based and scale-free -- use it directly.
            score = rrf_scores.get(cid, 0.0) / max_rrf

            title_lower = r.get("title", "").lower()
            content_lower = r.get("content", "").lower()
            path_lower = r.get("path", "").lower()

            title_hits = sum(1 for t in clean_terms if t in title_lower)
            path_hits = sum(1 for t in clean_terms if t in path_lower)
            content_hits = sum(1 for t in clean_terms if t in content_lower)

            score += 0.25 * (title_hits / n_terms)
            score += 0.20 * (path_hits / n_terms)
            score += 0.10 * (content_hits / n_terms)

            # Exact phrase match in title
            if clean_phrase and clean_phrase in title_lower:
                score += 0.15

            # No min(0.99, ...) clamp: saturating at the cap collapsed distinct
            # candidates into ties and made the score useless as a confidence signal.

            scored_results.append(
                {
                    "score": round(score, 3),
                    "base_score": round(rrf_scores.get(cid, 0.0) / max_rrf, 3),
                    "path": r["path"],
                    "abs_path": r["abs_path"],
                    "doc_type": r["doc_type"],
                    "title": r["title"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "content": r["content"],
                }
            )

        scored_results.sort(key=lambda x: x["score"], reverse=True)

        if max_per_file and max_per_file > 0:
            # Diversify before truncating to `limit`: several chunks of the same
            # file crowd out other candidate files, and the agent only needs one
            # anchor per file to decide whether to open it.
            seen: dict[str, int] = {}
            diversified = []
            for r in scored_results:
                n = seen.get(r["path"], 0)
                if n >= max_per_file:
                    continue
                seen[r["path"]] = n + 1
                diversified.append(r)
            scored_results = diversified

        if code_first and not norm_type and not is_doc_seeking(query):
            # Doc-heavy repos drown code in prose: FastAPI indexes 15,839 markdown
            # chunks against 5,689 code chunks, so 8.6 of every 10 unfiltered hits
            # were documentation. Float code above markdown while keeping the prose
            # as a tail rather than dropping it, so doc questions still resolve.
            code = [r for r in scored_results if r.get("doc_type") == "code_doc"]
            other = [r for r in scored_results if r.get("doc_type") != "code_doc"]
            scored_results = code + other

        return scored_results[:limit]
