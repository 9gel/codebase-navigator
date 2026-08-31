import pytest
from pathlib import Path
from codebase_navigator.extractor import DocExtractor


def test_extract_markdown_sections(tmp_path: Path):
    extractor = DocExtractor(tmp_path)
    md_file = tmp_path / "README.md"
    md_file.write_text("""# Project Title

Introduction to the project with sufficient text to qualify as a chunk.

## Architecture

Detailed architecture overview explaining how components communicate.
""")

    chunks = extractor.extract_markdown(md_file)
    assert len(chunks) >= 2
    titles = [c["title"] for c in chunks]
    assert any("Project Title" in t for t in titles)
    assert any("Architecture" in t for t in titles)


def test_extract_markdown_terms(tmp_path: Path):
    extractor = DocExtractor(tmp_path)
    glossary_file = tmp_path / "GLOSSARY.md"
    glossary_file.write_text("""# Glossary

## Terms

**Retail destination** — **a place that holds shops**: a mall, an arcade, a market.
Not a shop. One destination usually contains several shops.

**Shop** — one retail unit inside a destination. Counted, not published as its own row.
""")

    chunks = extractor.extract_markdown(glossary_file)
    titles = [c["title"] for c in chunks]

    assert any("Retail destination" in t for t in titles)
    assert any("Shop" in t for t in titles)

    term_chunk = next(c for c in chunks if "Retail destination" in c["title"])
    assert "a place that holds shops" in term_chunk["content"]


def test_extract_python_docstrings(tmp_path: Path):
    extractor = DocExtractor(tmp_path)
    py_file = tmp_path / "service.py"
    py_file.write_text('''"""Service module documentation for testing."""

class Calculator:
    """A simple calculator class for arithmetic operations."""

    def add(self, a: int, b: int) -> int:
        """Add two integers together and return sum."""
        return a + b
''')

    chunks = extractor.extract_code_doc(py_file)
    assert len(chunks) >= 2
    titles = [c["title"] for c in chunks]
    assert any("module docstring" in t for t in titles)
    assert any("Calculator" in t for t in titles)
    assert any("add" in t for t in titles)
