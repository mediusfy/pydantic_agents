from unittest.mock import patch

import pytest

from pydantic_agents import mcp_server


@pytest.mark.parametrize(
    "project_arg,expected_project",
    [
        ("", "alpha"),
        ("custom", "custom"),
    ],
)
def test_seed_namespaces_ids_and_tags_project(tmp_path, project_arg, expected_project):
    root = tmp_path / "alpha"
    root.mkdir()
    (root / "README.md").write_text("hello")

    with patch.object(mcp_server.vector_db, "add_documents") as add:
        out = mcp_server.seed_project_context(str(root), project_arg)

    assert "Indexed 1 files" in out
    _, docs, metas, ids = add.call_args.args
    assert docs == ["hello"]
    assert ids == [f"{expected_project}:file:readme.md"]
    assert metas[0]["project"] == expected_project


def test_seed_skips_vendored_dirs(tmp_path):
    root = tmp_path / "beta"
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "CONTRIBUTING.md").write_text("third party")
    (root / "docs").mkdir()
    (root / "docs" / "ARCHITECTURE.md").write_text("ours")

    with patch.object(mcp_server.vector_db, "add_documents") as add:
        mcp_server.seed_project_context(str(root))

    ids = [c.args[3][0] for c in add.call_args_list]
    assert ids == ["beta:file:docs::architecture.md"]
