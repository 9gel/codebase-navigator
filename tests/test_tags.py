import pytest
from pathlib import Path
from codebase_navigator.tags import get_available_files, TagsManager


def test_get_available_files(tmp_path: Path):
    # Setup test file tree
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    py_file = src_dir / "app.py"
    py_file.write_text("def hello(): pass\n")

    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    md_file = doc_dir / "guide.md"
    md_file.write_text("# Guide\nHello world\n")

    # Ignored directory
    ignore_dir = tmp_path / "node_modules" / "pkg"
    ignore_dir.mkdir(parents=True)
    (ignore_dir / "index.js").write_text("console.log(1);")

    code_files, doc_files = get_available_files(tmp_path)

    assert py_file in code_files
    assert md_file in doc_files
    assert not any("node_modules" in str(p) for p in code_files)


def test_parse_tag_line(tmp_path: Path):
    mgr = TagsManager(tmp_path)
    tag_line = 'my_func\tsrc/app.py\t/^def my_func():$/;"\tf\tline:42'
    parsed = mgr._parse_tag_line(tag_line, tmp_path)

    assert parsed is not None
    assert parsed["symbol"] == "my_func"
    assert parsed["path"] == "src/app.py"
    assert parsed["line"] == 42
    assert parsed["kind"] == "f"
