"""Go module setup is awaited and time-limited, and unwritable file names make tools return a failure."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core.tools import frontend_tools, go_common, gobuild
from pydantic_agents.core.tools.test_tools import execute_test_suite


class _Recorder:
    """Fake create_subprocess_exec that records when each process starts and when it has been waited for."""

    def __init__(self, hang_on=None):
        self.events: list[str] = []
        self.hang_on = hang_on
        self.procs: list[MagicMock] = []

    async def __call__(self, *args, **_kwargs):
        label = " ".join(args[:3])
        self.events.append(f"start {label}")
        recorder = self

        async def communicate():
            if recorder.hang_on and recorder.hang_on in label:
                await asyncio.sleep(5)
            await asyncio.sleep(0)  # yield, so an un-awaited process would let later steps overtake it
            recorder.events.append(f"done {label}")
            return b"ok", b""

        proc = MagicMock(returncode=0, communicate=communicate, kill=MagicMock(), wait=AsyncMock())
        recorder.procs.append(proc)
        return proc


async def test_go_test_starts_only_after_mod_init_and_tidy_finished():
    spawn = _Recorder()
    with patch("asyncio.create_subprocess_exec", spawn):
        result = await execute_test_suite("svc_test.go", "package svc", "svc.go", "package svc")

    assert result.startswith("SUCCESS")
    assert spawn.events.index("done go mod init") < spawn.events.index("start go mod tidy")
    assert spawn.events.index("done go mod tidy") < spawn.events.index("start go test -v")


@pytest.mark.parametrize(
    "hang_on, expected",
    [("mod init", "'go mod init' timed out"), ("mod tidy", "'go mod tidy' timed out")],
)
async def test_slow_go_module_setup_fails_the_run_instead_of_hanging(monkeypatch, hang_on, expected):
    monkeypatch.setattr(go_common, "GO_MOD_INIT_TIMEOUT", 0.05)
    monkeypatch.setattr(go_common, "GO_MOD_TIDY_TIMEOUT", 0.05)
    spawn = _Recorder(hang_on=hang_on)
    with patch("asyncio.create_subprocess_exec", spawn):
        result = await execute_test_suite("svc_test.go", "package svc", "svc.go", "package svc")

    assert result.startswith("TEST EXECUTION FAILED")
    assert expected in result
    assert not any("go test" in e for e in spawn.events)
    hung = spawn.procs[-1]
    hung.kill.assert_called_once()  # the stuck process is killed...
    hung.wait.assert_awaited_once()  # ...and reaped, not left as a zombie


async def test_missing_go_toolchain_is_a_failure_message():
    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError("go"))):
        test_result = await execute_test_suite("svc_test.go", "package svc", "svc.go", "package svc")
        build_result = await gobuild.run_go_build({"main.go": "package main"})
        vet_result = await gobuild.run_go_vet({"main.go": "package main"})

    assert test_result == "TEST EXECUTION FAILED: the 'go' toolchain is not installed."
    assert build_result == "BUILD FAILED: the 'go' toolchain is not installed."
    assert vet_result == "VET FAILED: the 'go' toolchain is not installed."


BAD_NAMES = [
    pytest.param("bad\x00name.go", id="nul-byte"),
    pytest.param("../escape.go", id="traversal"),
    pytest.param("/tmp/agent-leftover-escape.go", id="absolute"),
    pytest.param("a/b.go", id="parent-is-a-file"),
    pytest.param("x" * 4096 + ".go", id="name-too-long"),
]


@pytest.mark.parametrize("name", BAD_NAMES)
async def test_go_tools_return_a_failure_for_invalid_names(name):
    files = {"a": "package a", name: "package x"}  # "a" is a file, so "a/b.go" cannot be created
    with patch("asyncio.create_subprocess_exec", AsyncMock()) as spawn:
        build = await gobuild.run_go_build(files)
        vet = await gobuild.run_go_vet(files)

    assert build.startswith("BUILD FAILED")
    assert vet.startswith("VET FAILED")
    spawn.assert_not_awaited()


# "parent-is-a-file" needs a second file to collide with, which a single-file tool never has.
SINGLE_FILE_BAD_NAMES = [p for p in BAD_NAMES if p.id != "parent-is-a-file"]


@pytest.mark.parametrize("name", SINGLE_FILE_BAD_NAMES)
async def test_validate_frontend_code_returns_a_failure_for_invalid_names(name):
    with patch("asyncio.create_subprocess_exec", AsyncMock()) as spawn:
        result = await frontend_tools.validate_frontend_code(name, "<template/>")

    assert result.startswith("VALIDATION FAILED")
    spawn.assert_not_awaited()


async def test_validate_frontend_code_still_runs_the_linter_for_a_good_name():
    proc = MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"", b"")))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
        result = await frontend_tools.validate_frontend_code("components/Button.vue", "<template><b/></template>")

    assert result == "SUCCESS"
    assert spawn.await_args.args[:2] == ("npx", "eslint")
    assert spawn.await_args.args[2].endswith("components/Button.vue")
