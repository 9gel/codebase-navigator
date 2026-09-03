"""LanceDB vector index management and hybrid semantic search."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lancedb

if TYPE_CHECKING:
    from fastembed import TextEmbedding

from lancedb.index import FTS

from .config import (
    DOC_SCHEMA,
    EMBEDDING_MODEL_NAME,
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


class VectorIndex:
    """LanceDB index manager with FastEmbed ONNX embeddings and hybrid re-ranking."""

    def __init__(self, folder: Path, custom_index_dir: str | None = None):
        self.folder = folder
        self.cache_dir = get_cache_dir(folder, custom_index_dir)
        self.db_dir = self.cache_dir / "lancedb"
        self.meta_file = self.cache_dir / "files_meta.json"
        self.db = lancedb.connect(str(self.db_dir))
        self._model: TextEmbedding | None = None
        self._ensure_table()

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            with silence_stdio():
                self._model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        return self._model

    def _ensure_table(self):
        try:
            self.table = self.db.open_table("documents")
        except Exception:
            self.table = self.db.create_table("documents", schema=DOC_SCHEMA, mode="create")

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
            embeddings = list(self.model.embed(texts, batch_size=256))
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
            texts = [c["content"] for c in chunks]
            embeddings = list(self.model.embed(texts, batch_size=64))
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
    ) -> list[dict[str, Any]]:
        """True hybrid semantic vector + BM25 FTS search with Reciprocal Rank Fusion."""
        if len(self.table) == 0:
            return []

        q_vec = list(self.model.embed([query]))[0]
        if hasattr(q_vec, "tolist"):
            q_vec = q_vec.tolist()
        else:
            q_vec = list(q_vec)
        fetch_limit = max(limit * 4, 25)

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

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        k_rrf = 60
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
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank + 1))

        if not merged_candidates:
            return []

        # Normalization factor for RRF: max theoretical score with rank 0 in both is 2 / (k_rrf + 1)
        max_rrf = 2.0 / (k_rrf + 1)

        scored_results: list[dict[str, Any]] = []
        for cid, r in merged_candidates.items():
            dist = r.get("_distance", None)
            if dist is not None:
                vec_base = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            else:
                vec_base = 0.50

            # Base score combining vector proximity and RRF rank
            norm_rrf = rrf_scores.get(cid, 0.0) / max_rrf
            base_score = max(vec_base, norm_rrf * 0.90)
            score = base_score

            title_lower = r.get("title", "").lower()
            content_lower = r.get("content", "").lower()

            # Exact phrase match in title
            if clean_phrase and clean_phrase in title_lower:
                score += 0.12
            # Individual term matches in title
            for term in clean_terms:
                if term in title_lower:
                    score += 0.04
                # Content keyword match boost (+0.02 per term)
                elif term in content_lower:
                    score += 0.02

            score = min(0.99, score)

            scored_results.append(
                {
                    "score": round(score, 3),
                    "base_score": round(base_score, 3),
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
        return scored_results[:limit]
