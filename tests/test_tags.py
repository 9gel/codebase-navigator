from pathlib import Path

from codebase_navigator.tags import TagsManager, get_available_files


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


def test_custom_tag_file_resolves_paths_against_repository(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "app.py"
    source.write_text("def custom_symbol():\n    pass\n", encoding="utf-8")
    tag_file = tmp_path / "run" / "indexes" / "repo" / ".tags"
    tag_file.parent.mkdir(parents=True)
    tag_file.write_text(
        'custom_symbol\tapp.py\t/^def custom_symbol():$/;"\tf\tline:1\n',
        encoding="utf-8",
    )

    matches = TagsManager(repo, tag_file=tag_file).lookup_symbol("custom_symbol", exact=True)

    assert len(matches) == 1
    assert matches[0]["path"] == "app.py"
    assert matches[0]["abs_path"] == str(source.resolve())
