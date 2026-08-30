import pytest
from pathlib import Path
from devel_tools.index import VectorIndex


def test_glossary_term_boosted_search(tmp_path: Path):
    glossary = tmp_path / "GLOSSARY.md"
    glossary.write_text("""# Glossary

## Domain Concepts

**Retail destination** — **a place that holds shops**: a mall, an arcade, a market, a shopping centre.
Not a shop. One destination usually contains several shops and may span more than one building.

**Shop** — one retail unit inside a destination.
""")

    code_file = tmp_path / "retail.py"
    code_file.write_text('''"""Retail processing domain logic."""

def calculate_retail_metrics():
    """Calculate aggregate retail performance numbers."""
    return {}
''')

    index_dir = tmp_path / ".idx"
    idx = VectorIndex(tmp_path, custom_index_dir=str(index_dir))
    idx.sync()

    results = idx.search("What is a Retail Destination", limit=3)
    assert len(results) >= 1
    # Top result should be the Glossary term definition
    top = results[0]
    assert "GLOSSARY.md" in top["path"]
    assert "Retail destination" in top["title"]
    assert top["score"] >= 0.85
