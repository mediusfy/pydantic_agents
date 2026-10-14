"""Tests for the read_file / list_directory sandboxing on the service_engineer agent.

The agent's file-reading tools are exposed to LLM-driven input and were
previously unrestricted — an MCP caller could be prompt-injected into
reading /etc/passwd or the project's .env. Post-fix, both tools refuse any
path that escapes ORCHESTRATION_CODEBASE_ROOT (defaulting to CWD).
"""

import os
from unittest.mock import MagicMock

import pytest

from pydantic_agents.agents.service_engineer import _resolve_sandboxed, list_directory, read_file
from pydantic_agents.core.exceptions import PathTraversalError


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point ORCHESTRATION_CODEBASE_ROOT at a tmp directory and populate it."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.go").write_text("package main\nfunc main() {}\n")
    (tmp_path / "README.md").write_text("# project\n")
    monkeypatch.setenv("ORCHESTRATION_CODEBASE_ROOT", str(tmp_path))
    return tmp_path


def test_resolve_sandboxed_accepts_relative_in_root(sandbox):
    resolved = _resolve_sandboxed("src/main.go")
    assert resolved == (sandbox / "src" / "main.go").resolve()


def test_resolve_sandboxed_accepts_absolute_in_root(sandbox):
    resolved = _resolve_sandboxed(str(sandbox / "README.md"))
    assert resolved == (sandbox / "README.md").resolve()


def test_resolve_sandboxed_rejects_dotdot_traversal(sandbox):
    with pytest.raises(PathTraversalError):
        _resolve_sandboxed("../../../etc/passwd")


def test_resolve_sandboxed_rejects_absolute_outside_root(sandbox):
    # /etc/passwd exists on POSIX hosts; on Windows substitute any path
    # outside the sandbox.
    outside = "/etc/passwd" if os.name == "posix" else "C:\\Windows\\System32\\drivers\\etc\\hosts"
    with pytest.raises(PathTraversalError):
        _resolve_sandboxed(outside)


@pytest.mark.asyncio
async def test_read_file_returns_content_in_sandbox(sandbox):
    ctx = MagicMock()
    content = await read_file(ctx, "src/main.go")
    assert "package main" in content


@pytest.mark.asyncio
async def test_read_file_refuses_traversal(sandbox):
    ctx = MagicMock()
    result = await read_file(ctx, "../../../etc/passwd")
    assert result.startswith("ERROR:")
    assert "escapes sandbox root" in result


@pytest.mark.asyncio
async def test_read_file_missing_returns_error_not_traceback(sandbox):
    ctx = MagicMock()
    result = await read_file(ctx, "src/does_not_exist.go")
    assert result.startswith("ERROR: File not found")


@pytest.mark.asyncio
async def test_read_file_truncates_large_files(sandbox, tmp_path):
    # Write a 300 KiB file (above _READ_FILE_MAX_BYTES = 256 KiB)
    large = sandbox / "large.txt"
    large.write_text("a" * (300 * 1024))
    ctx = MagicMock()
    content = await read_file(ctx, "large.txt")
    assert content.endswith("(truncated at 262144 bytes)")
    assert len(content) < 300 * 1024 + 100  # at most marker overhead


def test_list_directory_lists_sandbox_contents(sandbox):
    ctx = MagicMock()
    listing = list_directory(ctx, ".")
    assert "src/" in listing.splitlines()
    assert "README.md" in listing.splitlines()


def test_list_directory_refuses_traversal(sandbox):
    ctx = MagicMock()
    result = list_directory(ctx, "../../")
    assert result.startswith("ERROR:")
    assert "escapes sandbox root" in result


def test_sandbox_default_is_cwd_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCHESTRATION_CODEBASE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.txt").write_text("hi")
    # An in-CWD path resolves; an out-of-CWD path raises.
    assert _resolve_sandboxed("x.txt") == (tmp_path / "x.txt").resolve()
    with pytest.raises(PathTraversalError):
        _resolve_sandboxed("../escape")
