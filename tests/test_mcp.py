from pathlib import Path
from codebase_navigator.mcp_server import (
    resolve_repository_root,
    codebase_search,
    codebase_read,
    codebase_tags,
    codebase_references,
    codebase_call_tree,
    codebase_grep,
)
from codebase_navigator.tags import TagsManager
from codebase_navigator.index import VectorIndex


def test_mcp_tools_and_root_resolution(tmp_path: Path):
    # Setup test file
    src = tmp_path / "app.py"
    src.write_text(
        """def launch_server(port):
    print("Serving on port", port)

def start():
    launch_server(8080)
""",
        encoding="utf-8",
    )

    doc = tmp_path / "README.md"
    doc.write_text("# App Server\n\nHandles HTTP traffic and launches worker processes.")

    # 1. Test root resolution with explicit parameter
    resolved = resolve_repository_root(str(tmp_path))
    assert resolved == tmp_path.resolve()

    # 2. Test codebase_search
    idx = VectorIndex(tmp_path)
    idx.sync()
    tags = TagsManager(tmp_path)
    tags.generate()

    search_out = codebase_search("HTTP traffic", repo_root=str(tmp_path))
    assert "README.md" in search_out or "App Server" in search_out

    # 3. Test codebase_read
    read_out = codebase_read("app.py", start_line=1, end_line=3, repo_root=str(tmp_path))
    assert "launch_server" in read_out
    assert "Serving on port" in read_out

    # 4. Test codebase_tags
    tags_out = codebase_tags("launch_server", exact=True, repo_root=str(tmp_path))
    assert "launch_server" in tags_out

    # 5. Test codebase_references
    refs_out = codebase_references("launch_server", repo_root=str(tmp_path))
    assert "Definition:" in refs_out or "Usage/Caller:" in refs_out

    # 6. Test codebase_call_tree
    tree_out = codebase_call_tree("launch_server", repo_root=str(tmp_path))
    assert "Call Tree for `launch_server`" in tree_out

    # 7. Test codebase_grep
    grep_out = codebase_grep("8080", repo_root=str(tmp_path))
    assert "app.py" in grep_out
    assert "8080" in grep_out
