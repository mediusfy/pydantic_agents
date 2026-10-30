"""execute_test_suite writes model-chosen file names, so it must stay inside its temp dir."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core.tools.test_tools import execute_test_suite


def _proc(returncode=0, out=b"ok", err=b""):
    return MagicMock(returncode=returncode, communicate=AsyncMock(return_value=(out, err)), kill=MagicMock())


@pytest.mark.parametrize(
    "test_name, target_name",
    [
        ("../escape_test.go", "svc.go"),
        ("svc_test.go", "../escape.go"),
        ("/tmp/agent-sandbox-escape_test.go", "svc.go"),
        ("svc_test.go", "/tmp/agent-sandbox-escape.go"),
    ],
)
async def test_traversal_in_file_names_is_rejected_before_anything_runs(test_name, target_name):
    with patch("asyncio.create_subprocess_exec", AsyncMock()) as spawn:
        result = await execute_test_suite(test_name, "package x", target_name, "package x")
    assert result.startswith("TEST EXECUTION FAILED")
    spawn.assert_not_awaited()


@pytest.mark.parametrize(
    "test_name, target_name",
    [("bad\x00name_test.go", "svc.go"), ("svc_test.go", "bad\x00.go"), ("pkg", "pkg/svc.go")],
)
async def test_unwritable_names_return_a_failure_instead_of_raising(test_name, target_name):
    with patch("asyncio.create_subprocess_exec", AsyncMock()) as spawn:
        result = await execute_test_suite(test_name, "package x", target_name, "package x")
    assert result.startswith("TEST EXECUTION FAILED")
    spawn.assert_not_awaited()


@pytest.mark.parametrize(
    "returncode, expected_prefix",
    [(0, "SUCCESS"), (1, "FAILED")],
)
async def test_vitest_path_receives_the_written_test_file(returncode, expected_prefix):
    spawn = AsyncMock(return_value=_proc(returncode))
    with patch("asyncio.create_subprocess_exec", spawn):
        result = await execute_test_suite("a.spec.ts", "test('x')", "a.ts", "export const a = 1")

    args = spawn.await_args.args
    assert args[:3] == ("npx", "vitest", "run")
    assert args[3].endswith("a.spec.ts")
    assert result.startswith(expected_prefix)


async def test_go_tests_run_go_test_in_the_temp_module():
    spawn = AsyncMock(return_value=_proc(0, b"PASS"))
    with patch("asyncio.create_subprocess_exec", spawn):
        result = await execute_test_suite("svc_test.go", "package svc", "svc.go", "package svc")

    assert spawn.await_args.args[:2] == ("go", "test")
    assert result == "SUCCESS\nPASS"


async def test_timeouts_kill_the_process():
    proc = _proc()
    proc.communicate = AsyncMock(side_effect=TimeoutError)
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        result = await execute_test_suite("a.spec.ts", "x", "a.ts", "y")
    assert result == "TEST EXECUTION FAILED: Timed out."
    proc.kill.assert_called_once()


async def test_the_event_loop_is_not_blocked_by_file_writes():
    # Writes go through asyncio.to_thread, so a coroutine scheduled alongside still makes progress.
    ticks = []

    async def ticker():
        for _ in range(3):
            ticks.append(1)
            await asyncio.sleep(0)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=_proc())):
        await asyncio.gather(execute_test_suite("a.spec.ts", "x", "a.ts", "y"), ticker())
    assert len(ticks) == 3


async def test_subprocesses_do_not_inherit_secret_env_vars(monkeypatch):
    monkeypatch.setenv("LITELLM_VIRTUAL_KEY", "sk-secret-litellm")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-openai")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret@db/prod")
    spawn = AsyncMock(return_value=_proc(0, b"PASS"))
    with patch("asyncio.create_subprocess_exec", spawn):
        await execute_test_suite("a.spec.ts", "test('x')", "a.ts", "export const a = 1")

    passed_env = spawn.await_args.kwargs["env"]
    assert "LITELLM_VIRTUAL_KEY" not in passed_env
    assert "OPENAI_API_KEY" not in passed_env
    assert "DATABASE_URL" not in passed_env


def test_sanitized_subprocess_env_allowlist_and_trust_boundary_pins(monkeypatch):
    from pydantic_agents.core.settings import settings
    from pydantic_agents.core.utils import _SAFE_SUBPROCESS_ENV_KEYS, sanitized_subprocess_env

    # Dangerous Go/Node/XDG override vars must not be in the allowlist or leak through.
    forbidden = (
        "GOPRIVATE",
        "GONOPROXY",
        "GONOSUMDB",
        "GOFLAGS",
        "GOENV",
        "NODE_PATH",
        "NODE_OPTIONS",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    )
    for key in forbidden:
        assert key not in _SAFE_SUBPROCESS_ENV_KEYS
        monkeypatch.setenv(key, "attacker-controlled")

    monkeypatch.setenv("GOPROXY", "http://internal-proxy.corp.local")
    monkeypatch.setenv("GOSUMDB", "off")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(settings, "untrusted_goproxy", "https://proxy.golang.org,direct")
    monkeypatch.setattr(settings, "untrusted_gosumdb", "sum.golang.org")

    env = sanitized_subprocess_env()

    for key in forbidden:
        if key != "GOENV":
            assert key not in env
    assert env["GOENV"] == "off"
    assert env["GOPROXY"] == "https://proxy.golang.org,direct"
    assert env["GOSUMDB"] == "sum.golang.org"
    assert env["PATH"] == "/usr/bin:/bin"
    assert set(env).issubset(_SAFE_SUBPROCESS_ENV_KEYS | {"GOENV", "GOPROXY", "GOSUMDB"})
