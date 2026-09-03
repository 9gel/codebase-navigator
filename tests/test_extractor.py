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
    # Check body inclusion and contextual metadata
    calc_chunk = next(c for c in chunks if "Calculator" in c["title"])
    assert "Language: Python" in calc_chunk["content"]
    assert "return a + b" in calc_chunk["content"]


def test_extract_markdown_ignores_code_fences(tmp_path: Path):
    extractor = DocExtractor(tmp_path)
    md_file = tmp_path / "GUIDE.md"
    md_file.write_text("""# Real Section

Here is some documentation.

```bash
# This is a comment inside a shell script
echo "hello world"
```

## Another Section

Content here.
""")
    chunks = extractor.extract_markdown(md_file)
    titles = [c["title"] for c in chunks]
    assert not any("This is a comment" in t for t in titles)
    assert any("Real Section" in t for t in titles)
    assert any("Another Section" in t for t in titles)


def test_extract_multilang_code_structure(tmp_path: Path):
    extractor = DocExtractor(tmp_path)

    # Rust
    rs_file = tmp_path / "lib.rs"
    rs_file.write_text("""pub struct WorkerPool {
    size: usize,
}

pub fn spawn_worker(id: u32) -> Result<(), String> {
    println!("worker spawned");
    Ok(())
}
""")
    rs_chunks = extractor.extract_code_doc(rs_file)
    rs_titles = [c["title"] for c in rs_chunks]
    assert any("WorkerPool" in t for t in rs_titles)
    assert any("spawn_worker" in t for t in rs_titles)
    assert "Language: Rust" in rs_chunks[0]["content"]

    # Go
    go_file = tmp_path / "main.go"
    go_file.write_text("""package main

type Config struct {
    Host string
}

func HandleRequest(w ResponseWriter, r *Request) {
    w.WriteHeader(200)
}
""")
    go_chunks = extractor.extract_code_doc(go_file)
    go_titles = [c["title"] for c in go_chunks]
    assert any("Config" in t for t in go_titles)
    assert any("HandleRequest" in t for t in go_titles)
    assert "Language: Go" in go_chunks[0]["content"]

    # TypeScript
    ts_file = tmp_path / "index.ts"
    ts_file.write_text("""export interface UserSession {
    token: string;
}

export const authenticate = async (token: string) => {
    return true;
};
""")
    ts_chunks = extractor.extract_code_doc(ts_file)
    ts_titles = [c["title"] for c in ts_chunks]
    assert any("UserSession" in t for t in ts_titles)
    assert any("authenticate" in t for t in ts_titles)
    assert "Language: TypeScript" in ts_chunks[0]["content"]
