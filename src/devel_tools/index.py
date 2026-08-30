"""LanceDB vector index management and hybrid semantic search."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import lancedb
from sentence_transformers import SentenceTransformer

from .config import (
    DOC_SCHEMA,
    EMBEDDING_MODEL_NAME,
    get_cache_dir,
    silence_stdio,
)
from .extractor import DocExtractor
from .tags import get_available_files

COMMON_STOPWORDS = {
    "what", "is", "a", "an", "the", "in", "on", "of", "for", "to",
    "and", "or", "how", "why", "where", "which", "does", "do", "can",
}


class VectorIndex:
    """LanceDB index manager with SentenceTransformer embeddings and hybrid re-ranking."""

    def __init__(self, folder: Path, custom_index_dir: str | None = None):
        self.folder = folder
        self.cache_dir = get_cache_dir(folder, custom_index_dir)
        self.db_dir = self.cache_dir / "lancedb"
        self.meta_file = self.cache_dir / "files_meta.json"
        self.db = lancedb.connect(str(self.db_dir))
        self._model: SentenceTransformer | None = None
        self._ensure_table()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            with silence_stdio():
                try:
                    self._model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._model

    def _ensure_table(self):
        try:
            self.table = self.db.open_table("documents")
        except Exception:
            self.table = self.db.create_table("documents", schema=DOC_SCHEMA, mode="create")

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
            embeddings = self.model.encode(
                texts,
                batch_size=256,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            for chunk, vec in zip(all_chunks_to_embed, embeddings):
                chunk["vector"] = vec.tolist()

            self.table.add(all_chunks_to_embed)

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
            return 0

        extractor = DocExtractor(self.folder)
        if fpath.suffix.lower() in {".md", ".markdown", ".rst", ".adoc", ".org"}:
            chunks = extractor.extract_markdown(fpath)
        else:
            chunks = extractor.extract_code_doc(fpath)

        if chunks:
            texts = [c["content"] for c in chunks]
            embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            for chunk, vec in zip(chunks, embeddings):
                chunk["vector"] = vec.tolist()
            self.table.add(chunks)

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
        """Hybrid semantic vector search with keyword & title match re-ranking."""
        q_vec = self.model.encode(query, normalize_embeddings=True).tolist()
        fetch_limit = max(limit * 4, 20)
        search_query = self.table.search(q_vec).metric("cosine").limit(fetch_limit)

        if doc_type and doc_type != "all":
            norm_type = "markdown" if doc_type in ["md", "markdown"] else "code_doc" if doc_type in ["code", "code_doc"] else doc_type
            search_query = search_query.where(f'doc_type = "{norm_type}"')

        try:
            raw_results = search_query.to_list()
        except Exception:
            raw_results = []

        if not raw_results:
            return []

        # Extract significant query terms
        clean_terms = [
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]+", query)
            if len(w) >= 2 and w.lower() not in COMMON_STOPWORDS
        ]
        clean_phrase = " ".join(clean_terms)

        scored_results: list[dict[str, Any]] = []
        for r in raw_results:
            dist = r.get("_distance", 0.0)
            base_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            score = base_score

            title_lower = r.get("title", "").lower()
            content_lower = r.get("content", "").lower()
            dtype = r.get("doc_type", "")

            # 1. Exact phrase match in title
            if clean_phrase and clean_phrase in title_lower:
                score += 0.12
            # 2. Individual term matches in title
            for term in clean_terms:
                if term in title_lower:
                    score += 0.04

            # 3. Term definitions or documentation boost
            if dtype == "markdown":
                score += 0.04

            score = min(0.99, score)

            scored_results.append({
                "score": round(score, 3),
                "base_score": round(base_score, 3),
                "path": r["path"],
                "abs_path": r["abs_path"],
                "doc_type": r["doc_type"],
                "title": r["title"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "content": r["content"],
            })

        # Re-sort by boosted hybrid score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:limit]
