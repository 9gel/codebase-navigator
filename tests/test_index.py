import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import codebase_navigator.index as index_module
from codebase_navigator.index import VectorIndex


def test_vector_index_lifecycle(tmp_path: Path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    f1 = doc_dir / "intro.md"
    f1.write_text("""# Introduction

This document provides a comprehensive overview of the system architecture and components.
""")

    index_dir = tmp_path / ".custom-idx"
    idx = VectorIndex(tmp_path, custom_index_dir=str(index_dir))

    # Sync
    u_files, u_chunks, p_files = idx.sync()
    assert u_files == 1
    assert u_chunks >= 1
    assert p_files == 0

    # Search
    results = idx.search("system architecture overview")
    assert len(results) >= 1
    assert "intro.md" in results[0]["path"]

    # Incremental sync (no changes)
    u_files2, u_chunks2, p_files2 = idx.sync()
    assert u_files2 == 0
    assert u_chunks2 == 0

    # Delete file and prune
    f1.unlink()
    u_files3, u_chunks3, p_files3 = idx.sync()
    assert p_files3 == 1


def test_vector_indexes_share_one_embedding_model_across_threads(monkeypatch):
    created = 0

    class FakeTextEmbedding:
        def __init__(self, model_name: str):
            nonlocal created
            assert model_name == index_module.EMBEDDING_MODEL_NAME
            time.sleep(0.02)
            created += 1

    monkeypatch.setattr(index_module, "_SHARED_MODEL", None)
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    indexes = []
    for _ in range(4):
        index = object.__new__(VectorIndex)
        index._model = None
        indexes.append(index)

    with ThreadPoolExecutor(max_workers=4) as pool:
        models = list(pool.map(lambda index: index.model, indexes))

    assert created == 1
    assert all(model is models[0] for model in models)


def test_embedding_inference_is_serialized_across_workers():
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    class FakeModel:
        def embed(self, texts, **_kwargs):
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return [[float(len(text))] for text in texts]

    indexes = []
    for _ in range(4):
        index = object.__new__(VectorIndex)
        index._model = FakeModel()
        indexes.append(index)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda index: index._embed(["query"]), indexes))

    assert maximum_active == 1
    assert results == [[[5.0]]] * 4


def test_parallel_vector_searches_complete_on_one_index(tmp_path: Path):
    (tmp_path / "guide.md").write_text(
        "# Request hooks\n\nBefore-request hooks run before dispatch.\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    VectorIndex(tmp_path, custom_index_dir=str(index_dir)).sync()
    indexes = [VectorIndex(tmp_path, custom_index_dir=str(index_dir)) for _ in range(4)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(index.search, "before request hooks") for index in indexes]
        results = [future.result(timeout=10) for future in futures]

    assert all(result for result in results)
    assert all(result[0]["path"] == "guide.md" for result in results)
