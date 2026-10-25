import os
from unittest.mock import patch

import pytest

from pydantic_agents import mcp_server
from pydantic_agents.core.settings import settings


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
        out = mcp_server.seed_project_files(str(root), project_arg)

    assert "Indexed 1 files" in out
    _, docs, metas, ids = add.call_args.args
    assert docs == ["hello"]
    assert ids == [f"{expected_project}:file:readme.md#0"]
    assert metas[0]["project"] == expected_project


def test_seed_skips_vendored_dirs(tmp_path):
    root = tmp_path / "beta"
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "CONTRIBUTING.md").write_text("third party")
    (root / "docs").mkdir()
    (root / "docs" / "ARCHITECTURE.md").write_text("ours")

    with patch.object(mcp_server.vector_db, "add_documents") as add:
        mcp_server.seed_project_files(str(root))

    ids = [c.args[3][0] for c in add.call_args_list]
    assert ids == ["beta:file:docs::architecture.md#0"]


@pytest.mark.parametrize(
    "text,max_chars,expected_chunks",
    [
        ("short", 900, 1),
        ("a\n\n" * 3 + "b", 900, 1),
        ("\n\n".join(["p" * 400] * 3), 900, 2),
        ("word " * 500, 900, 3),
        ("", 900, 0),
    ],
)
def test_chunk_text_respects_limit(text, max_chars, expected_chunks):
    chunks = mcp_server._chunk_text(text, max_chars)
    assert len(chunks) == expected_chunks
    assert all(len(c) <= max_chars for c in chunks)


# --- the MCP tool only reads inside the allowed roots ---------------------------------------------------------------


def _project(base, name="gamma"):
    root = base / name
    root.mkdir()
    (root / "README.md").write_text("hello")
    return root


def test_the_tool_indexes_a_project_inside_seed_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "seed_roots", str(tmp_path))
    root = _project(tmp_path)

    with patch.object(mcp_server.vector_db, "add_documents") as add:
        out = mcp_server.seed_project_context(str(root))

    assert "Indexed 1 files" in out
    assert add.call_args.args[3] == ["gamma:file:readme.md#0"]


def test_the_tool_defaults_to_the_output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "seed_roots", "")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "out")
    (tmp_path / "out").mkdir()
    inside = _project(tmp_path / "out")
    outside = _project(tmp_path, "elsewhere")

    with patch.object(mcp_server.vector_db, "add_documents") as add:
        assert "Indexed 1 files" in mcp_server.seed_project_context(str(inside))
        assert mcp_server.seed_project_context(str(outside)).startswith("Error:")

    assert add.call_count == 1


def test_the_tool_accepts_several_roots(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(settings, "seed_roots", os.pathsep.join([str(first), str(second)]))

    with patch.object(mcp_server.vector_db, "add_documents"):
        assert "Indexed 1 files" in mcp_server.seed_project_context(str(_project(second)))


@pytest.mark.parametrize(
    "template",
    ["{other}", "{allowed}/../other", "/", "{parent}"],
    ids=["sibling", "dotdot", "slash", "parent"],
)
def test_the_tool_refuses_paths_outside_the_roots(tmp_path, monkeypatch, template):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    other = _project(tmp_path, "other")
    monkeypatch.setattr(settings, "seed_roots", str(allowed))
    path = template.format(other=other, allowed=allowed, parent=tmp_path)

    with patch.object(mcp_server.vector_db, "add_documents") as add:
        out = mcp_server.seed_project_context(path)

    assert out.startswith("Error: project_path is outside")
    add.assert_not_called()
    assert str(tmp_path) not in out  # the message does not echo the roots or the path


def test_a_symlink_leaving_the_project_is_not_indexed(tmp_path):
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "CONTRIBUTING.md").write_text("private notes")
    root = _project(tmp_path)
    (root / "CONTRIBUTING.md").symlink_to(secret / "CONTRIBUTING.md")

    with patch.object(mcp_server.vector_db, "add_documents") as add:
        out = mcp_server.seed_project_files(str(root))

    assert "Indexed 1 files" in out  # only the README
    assert all("private notes" not in c.args[1][0] for c in add.call_args_list)


@pytest.mark.parametrize(
    ("rel", "filename", "domain"),
    [
        (".agents/rules/style.md", "style.md", "conventions"),
        ("docs/conventions.md", "conventions.md", "conventions"),
        ("docs/adr-001.md", "adr-001.md", "adrs"),
        ("docs/architecture.md", "architecture.md", "adrs"),
        (".agents/workflows/ship.md", "ship.md", "project"),
        ("readme.md", "README.md", "project"),
        ("notes.md", "notes.md", "architecture"),
    ],
)
def test_domain_follows_the_path(rel, filename, domain):
    assert mcp_server._seed_domain(rel, filename) == domain


def test_empty_files_are_not_indexed(tmp_path):
    root = tmp_path / "delta"
    root.mkdir()
    (root / "README.md").write_text("  \n")

    with patch.object(mcp_server.vector_db, "add_documents") as add:
        out = mcp_server.seed_project_files(str(root))

    assert out == "Indexed 0 files into vector DB"
    add.assert_not_called()


def test_a_failing_file_is_reported_and_the_rest_continue(tmp_path):
    root = tmp_path / "epsilon"
    root.mkdir()
    (root / "README.md").write_text("one")
    (root / "CONTRIBUTING.md").write_text("two")

    def add(_domain, _chunks, _metas, ids):
        if "contributing" in ids[0]:
            raise RuntimeError("db down")

    with patch.object(mcp_server.vector_db, "add_documents", side_effect=add):
        out = mcp_server.seed_project_files(str(root))

    assert "Indexed 1 files" in out  # the README
    assert "Errors:" in out
    assert "db down" in out
