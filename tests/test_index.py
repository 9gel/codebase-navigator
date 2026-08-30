import pytest
from pathlib import Path
from devel_tools.index import VectorIndex


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
