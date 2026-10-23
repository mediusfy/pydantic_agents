"""Path-traversal and argument-injection tests for the model-driven tools."""

import shutil
from unittest.mock import MagicMock

import pytest

from pydantic_agents.agents.service_engineer import run_gofmt_check
from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.tool_assignments import get_tool_categories
from pydantic_agents.core.tools import frontend_tools, gobuild
from pydantic_agents.core.tools.fs_tools import list_dir, read_file, write_file
from pydantic_agents.core.tools.registry import get_tools_for_agent
from pydantic_agents.core.tools.search import search_codebase, search_files
from pydantic_agents.core.utils import safe_path_join


@pytest.fixture
def root(tmp_path, monkeypatch):
    project = tmp_path / "repo"
    project.mkdir()
    (project / "main.go").write_text("package main\n")
    (tmp_path / "secret.txt").write_text("top secret")
    sibling = tmp_path / "repo-evil"
    sibling.mkdir()
    (sibling / "loot.go").write_text("package loot\n")
    monkeypatch.setenv("ORCHESTRATION_CODEBASE_ROOT", str(project))
    monkeypatch.chdir(project)
    return project


def test_safe_path_join_rejects_sibling_prefix(tmp_path):
    (tmp_path / "repo").mkdir()
    with pytest.raises(PathTraversalError):
        safe_path_join(tmp_path / "repo", "../repo-evil/x")


@pytest.mark.parametrize(
    "filepath, ok",
    [
        ("main.go", True),
        ("../secret.txt", False),
        ("../repo-evil/loot.go", False),
        ("/etc/passwd", False),
    ],
)
def test_read_file_is_sandboxed(root, filepath, ok):
    out = read_file(filepath)
    assert out.startswith("Error") != ok
    assert "top secret" not in out


@pytest.mark.parametrize("directory", ["..", "/", "../repo-evil"])
def test_list_dir_rejects_escape(root, directory):
    assert list_dir(directory).startswith("Error")


@pytest.mark.parametrize(
    "filepath, written",
    [
        ("out/new.go", True),
        ("../escaped.txt", False),
        ("/tmp/agent-sandbox-escape.txt", False),
    ],
)
def test_write_file_is_sandboxed(root, filepath, written):
    out = write_file(filepath, "x")
    assert out.startswith("Successfully") == written
    assert not (root.parent / "escaped.txt").exists()


def test_write_file_is_not_granted_by_default():
    from pydantic_agents.core import tool_assignments

    for agent in tool_assignments.AGENT_TOOL_CATEGORIES:
        granted = {name for name, _ in get_tools_for_agent(get_tool_categories(agent))}
        assert "write_file" not in granted, agent


@pytest.mark.parametrize("glob", ["../*", "/etc/*", "../repo-evil/*.go"])
def test_search_files_rejects_escape(root, glob):
    assert search_files(MagicMock(), glob) == []


def test_search_files_finds_in_root(root):
    assert search_files(MagicMock(), "*.go") == ["main.go"]


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
@pytest.mark.parametrize(
    "pattern, path",
    [
        ("--pre=touch pwned", "."),
        ("package", "../repo-evil"),
    ],
)
async def test_search_codebase_blocks_injection_and_escape(root, pattern, path):
    result = await search_codebase(MagicMock(), pattern, path)
    assert not (root / "pwned").exists()
    assert not any("loot.go" in line for line in result)


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
async def test_search_codebase_still_matches(root):
    assert await search_codebase(MagicMock(), "package", ".") == [str(root / "main.go")]


async def test_gofmt_check_rejects_outside_root(root):
    out = run_gofmt_check(MagicMock(), "../repo-evil/loot.go")
    assert out.startswith("ERROR")
    assert "package loot" not in out


@pytest.mark.parametrize("name", ["../escape.go", "/tmp/agent-sandbox-escape.go"])
async def test_go_build_rejects_traversal(name):
    assert (await gobuild.run_go_build({name: "package x"})).startswith("BUILD FAILED")
    assert (await gobuild.run_go_vet({name: "package x"})).startswith("VET FAILED")


@pytest.mark.parametrize("name", ["../Evil.vue", "/tmp/agent-sandbox-escape.vue"])
async def test_validate_frontend_rejects_traversal(name):
    assert (await frontend_tools.validate_frontend_code(name, "<template/>")).startswith("VALIDATION FAILED")


def test_fs_tool_categories():
    from pydantic_agents.core.tools.registry import tool_registry

    assert tool_registry["read_file"].categories == ["fs_read"]
    assert tool_registry["list_dir"].categories == ["fs_read"]
    assert tool_registry["write_file"].categories == ["fs_write"]


@pytest.mark.parametrize("agent", ["backend", "security", "cto"])
def test_search_files_resolves_for_search_agents(agent):
    names = {name for name, _ in get_tools_for_agent(get_tool_categories(agent))}
    assert "search_files" in names
    assert "read_file" in names
