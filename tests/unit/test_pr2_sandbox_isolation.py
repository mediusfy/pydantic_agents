"""Unit tests for PR 2: bubblewrap namespace sandbox, fail-closed SANDBOX_BACKEND=bwrap, offline npx --no-install
runners, process-group timeout reaping in frontend_tools, and hardened Dockerfile invariants."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core import sandbox, verifier
from pydantic_agents.core.exceptions import SandboxUnavailableError
from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.settings import settings
from pydantic_agents.core.tools import frontend_tools, go_common, gobuild, test_tools


@pytest.fixture(autouse=True)
def _clean_sandbox_cache(monkeypatch):
    sandbox.clear_sandbox_probe_cache()
    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)
    monkeypatch.setattr(settings, "sandbox_backend", "auto")
    yield
    sandbox.clear_sandbox_probe_cache()


# ── 1. core/sandbox.py: bwrap argv, mounts, network policy, and probe ───────


def test_build_bwrap_command_isolates_namespaces_root_proc_and_network(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    gopath = tmp_path / "go-mod-cache"
    gopath.mkdir()

    argv_offline, env_offline = sandbox.build_bwrap_command(
        ["go", "test", "./..."],
        cwd=work,
        allow_network=False,
        bwrap_bin="/usr/bin/bwrap",
        gopath_dir=str(gopath),
    )

    assert argv_offline[:2] == ["/usr/bin/bwrap", "--unshare-all"]
    assert "--share-net" not in argv_offline
    assert "--die-with-parent" in argv_offline
    assert "--new-session" in argv_offline
    assert argv_offline[argv_offline.index("--uid") : argv_offline.index("--uid") + 4] == [
        "--uid",
        "65534",
        "--gid",
        "65534",
    ]
    assert argv_offline[argv_offline.index("--tmpfs") : argv_offline.index("--tmpfs") + 2] == ["--tmpfs", "/"]
    assert "--proc" in argv_offline and "/proc" in argv_offline
    assert "/etc/resolv.conf" not in argv_offline
    # GOPATH is mounted read-only during offline test execution so untrusted tests cannot poison cached modules
    ro_idx = [i for i, tok in enumerate(argv_offline) if tok == "--ro-bind" and argv_offline[i + 1] == str(gopath)]
    assert len(ro_idx) == 1
    # Workspace is mounted after --tmpfs /tmp and separated by '--'
    sep = argv_offline.index("--")
    assert argv_offline[sep + 1 :] == ["go", "test", "./..."]
    assert sandbox.unwrap_sandbox_command(argv_offline) == ("go", "test", "./...")
    assert env_offline["HOME"] == str(sandbox._SANDBOX_HOME)
    assert env_offline["GOCACHE"] == str(sandbox._SANDBOX_GOCACHE)
    assert env_offline["TMPDIR"] == str(sandbox._SANDBOX_TMP)
    assert env_offline["GOPATH"] == str(gopath)

    # Network-enabled mode (used strictly for `go mod tidy`): adds --share-net and writable GOPATH bind
    new_gopath = tmp_path / "new-gopath"
    argv_net, _ = sandbox.build_bwrap_command(
        ["go", "mod", "tidy"],
        cwd=work,
        allow_network=True,
        bwrap_bin="/usr/bin/bwrap",
        gopath_dir=str(new_gopath),
    )
    assert "--share-net" in argv_net
    assert new_gopath.is_dir()
    rw_idx = [i for i, tok in enumerate(argv_net) if tok == "--bind" and argv_net[i + 1] == str(new_gopath)]
    assert len(rw_idx) == 1


def test_build_bwrap_command_binds_toolchains_installed_outside_system_paths(tmp_path):
    prefix = tmp_path / "hostedtoolcache" / "go" / "1.26" / "x64"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    exe = bin_dir / "go"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()

    args = sandbox._toolchain_mount_args("go", str(bin_dir))
    assert args == ["--ro-bind", str(prefix), str(prefix)]

    # A symlinked launcher (e.g. /opt/node/bin/npx -> ../lib/node_modules/npm/bin/npx-cli.js) binds both prefixes.
    real_pkg = tmp_path / "node" / "lib" / "node_modules" / "npm" / "bin"
    real_pkg.mkdir(parents=True)
    (real_pkg / "npx-cli.js").write_text("")
    (real_pkg / "npx-cli.js").chmod(0o755)
    node_bin = tmp_path / "node" / "bin"
    node_bin.mkdir()
    (node_bin / "npx").symlink_to(real_pkg / "npx-cli.js")
    linked = sandbox._toolchain_mount_args("npx", str(node_bin))
    assert linked[:3] == ["--ro-bind", str(tmp_path / "node"), str(tmp_path / "node")]
    assert str(real_pkg.parent) in linked

    # System toolchains and unknown programs add nothing.
    assert sandbox._toolchain_mount_args("sh", "/usr/bin:/bin") == []
    assert sandbox._toolchain_mount_args("definitely-not-a-binary", str(bin_dir)) == []


def test_build_bwrap_command_skips_duplicate_gopath_when_equal_to_cwd(tmp_path):
    argv, _ = sandbox.build_bwrap_command(
        ["go", "build", "./..."],
        cwd=tmp_path,
        allow_network=False,
        bwrap_bin="/usr/bin/bwrap",
        gopath_dir=str(tmp_path),
    )
    bind_targets = [argv[i + 1] for i, tok in enumerate(argv) if tok in ("--bind", "--ro-bind")]
    assert bind_targets.count(str(tmp_path.resolve())) == 1


def test_mount_or_symlink_handles_usrmerge_links_and_real_dirs(tmp_path):
    real_dir = tmp_path / "usr_bin"
    real_dir.mkdir()
    link_dir = tmp_path / "bin_link"
    link_dir.symlink_to("usr_bin")
    missing = tmp_path / "does_not_exist"

    args = sandbox._mount_or_symlink_args([str(real_dir), str(link_dir), str(missing)])
    assert args == ["--ro-bind", str(real_dir), str(real_dir), "--symlink", "usr_bin", str(link_dir)]


def test_probe_bwrap_caches_success_failure_and_oserror(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda _name, **_kw: None)
    assert sandbox.probe_bwrap() == (False, "'bwrap' binary is not installed")

    runs: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        runs.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    assert sandbox.probe_bwrap("/fake/bwrap") == (True, "")
    assert sandbox.probe_bwrap("/fake/bwrap") == (True, "")
    assert len(runs) == 1  # second call served from cache

    sandbox.clear_sandbox_probe_cache()
    monkeypatch.setattr(
        sandbox.subprocess,
        "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 1, "", "Operation not permitted"),
    )
    assert sandbox.probe_bwrap("/fake/bwrap") == (False, "Operation not permitted")

    sandbox.clear_sandbox_probe_cache()
    monkeypatch.setattr(sandbox.subprocess, "run", MagicMock(side_effect=OSError("exec format error")))
    ok, reason = sandbox.probe_bwrap("/fake/bwrap")
    assert ok is False and "exec format error" in reason


def test_resolve_sandbox_execution_modes_and_validation(tmp_path, monkeypatch, caplog):
    with pytest.raises(ValueError, match="must not be empty"):
        sandbox.resolve_sandbox_execution([], cwd=tmp_path)

    monkeypatch.setenv("SANDBOX_BACKEND", "invalid-mode")
    with pytest.raises(SandboxUnavailableError, match="Unknown SANDBOX_BACKEND"):
        sandbox.resolve_sandbox_execution(["go", "version"], cwd=tmp_path)

    # Explicit 'none' bypasses bwrap probe
    monkeypatch.setenv("SANDBOX_BACKEND", "none")
    argv, _ = sandbox.resolve_sandbox_execution(["go", "version"], cwd=tmp_path)
    assert argv == ["go", "version"]

    # Fail-closed 'bwrap' raises SandboxUnavailableError when bwrap is missing or denied
    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    monkeypatch.setattr(sandbox.shutil, "which", lambda _n, **_kw: None)
    with pytest.raises(SandboxUnavailableError, match="unavailable"):
        sandbox.resolve_sandbox_execution(["go", "version"], cwd=tmp_path)

    # 'bwrap' succeeds and wraps when probe succeeds
    monkeypatch.setattr(
        sandbox.shutil, "which", lambda name, **_kw: "/usr/bin/bwrap" if name == "bwrap" else "/usr/bin/true"
    )
    monkeypatch.setattr(sandbox.subprocess, "run", lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    sandbox.clear_sandbox_probe_cache()
    wrapped, _ = sandbox.resolve_sandbox_execution(["go", "test", "./..."], cwd=tmp_path)
    assert wrapped[0] == "/usr/bin/bwrap"
    assert sandbox.unwrap_sandbox_command(wrapped) == ("go", "test", "./...")

    # 'auto' logs once (at ERROR on Linux) when bwrap fails
    sandbox.clear_sandbox_probe_cache()
    monkeypatch.setenv("SANDBOX_BACKEND", "auto")
    monkeypatch.setattr(sandbox.sys, "platform", "linux")
    monkeypatch.setattr(sandbox.shutil, "which", lambda _n, **_kw: None)
    with caplog.at_level("WARNING", logger="pydantic_agents.core.sandbox"):
        out1, _ = sandbox.resolve_sandbox_execution(["go", "version"], cwd=tmp_path)
        out2, _ = sandbox.resolve_sandbox_execution(["go", "version"], cwd=tmp_path)
    assert out1 == ["go", "version"] and out2 == ["go", "version"]
    assert caplog.text.count("bwrap namespace sandbox unavailable") == 1


def test_unwrap_sandbox_command_passthrough_when_not_bwrap():
    assert sandbox.unwrap_sandbox_command(("go", "test", "./...")) == ("go", "test", "./...")
    assert sandbox.unwrap_sandbox_command(("bwrap", "--version")) == ("bwrap", "--version")


# ── 2. Fail-closed SandboxUnavailableError handling across all 5 callers ────


@pytest.mark.asyncio
async def test_all_execution_tools_report_sandbox_unavailable_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    monkeypatch.setattr(sandbox.shutil, "which", lambda _n, **_kw: None)

    # 1. execute_test_suite (Go & Vitest)
    go_res = await test_tools.execute_test_suite("svc_test.go", "package svc", "svc.go", "package svc")
    assert go_res.startswith("TEST EXECUTION FAILED: Sandbox backend 'bwrap' is unavailable")

    js_res = await test_tools.execute_test_suite("a.spec.ts", "test('a')", "a.ts", "export const a = 1")
    assert js_res.startswith("TEST EXECUTION FAILED: Sandbox backend 'bwrap' is unavailable")

    # 2. gobuild (run_go_build & run_go_vet) — both via init_and_tidy and direct _run_build/_run_vet
    build_res = await gobuild.run_go_build({"main.go": "package main"})
    assert build_res.startswith("BUILD FAILED: Sandbox backend 'bwrap' is unavailable")
    assert (await gobuild._run_build(str(tmp_path))).startswith("BUILD FAILED: Sandbox backend 'bwrap'")
    assert (await gobuild._run_vet(str(tmp_path))).startswith("VET FAILED: Sandbox backend 'bwrap'")

    # 3. verifier.verify_artifacts
    art = Artifact(agent_name="backend", files={"go.mod": "module m\n\ngo 1.26\n", "a.go": "package m\n"})
    rep = await verifier.verify_artifacts([art])
    assert rep.passed is False
    assert "Sandbox backend 'bwrap' is unavailable" in rep.checks[0].output

    # 4. frontend_tools.validate_frontend_code
    fe_res = await frontend_tools.validate_frontend_code("App.vue", "<template><div/></template>")
    assert fe_res.startswith("VALIDATION FAILED: Sandbox backend 'bwrap' is unavailable")


@pytest.mark.asyncio
async def test_go_mod_tidy_gets_network_while_go_mod_init_and_tests_do_not(monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    monkeypatch.setattr(
        sandbox.shutil, "which", lambda name, **_kw: "/usr/bin/bwrap" if name == "bwrap" else "/usr/bin/true"
    )
    monkeypatch.setattr(sandbox.subprocess, "run", lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, "", ""))

    recorded_argvs: list[tuple[str, ...]] = []

    async def fake_exec(*args, **_kwargs):
        recorded_argvs.append(args)
        return MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"PASS", b"")), kill=MagicMock())

    with patch("asyncio.create_subprocess_exec", fake_exec):
        out = await test_tools.execute_test_suite("svc_test.go", "package svc", "svc.go", "package svc")

    assert out.startswith("SUCCESS")
    assert len(recorded_argvs) == 3
    init_argv, tidy_argv, test_argv = recorded_argvs
    assert "--share-net" not in init_argv
    assert "--share-net" in tidy_argv
    assert "--share-net" not in test_argv


@pytest.mark.asyncio
async def test_init_and_tidy_handles_sandbox_error_on_tidy_step(tmp_path, monkeypatch):
    calls = 0

    async def fake_run_cmd(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SandboxUnavailableError("bwrap tidy blocked")

    monkeypatch.setattr(go_common, "run_cmd", fake_run_cmd)
    err = await go_common.init_and_tidy(str(tmp_path))
    assert err == "bwrap tidy blocked"


# ── 3. frontend_tools.py: offline npx, isolated temp cwd, and timeout reap ──


@pytest.mark.asyncio
async def test_validate_frontend_code_isolates_cwd_and_reaps_timeouts(monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "none")
    server_cwd = os.getcwd()
    observed_cwds: list[str] = []

    # Case 1: eslint fails (returncode=1), vue-tsc succeeds (returncode=0) -> strict eslint error returned
    calls = 0

    async def fake_eslint_then_tsc(*_args, **kwargs):
        nonlocal calls
        calls += 1
        observed_cwds.append(kwargs["cwd"])
        if calls == 1:
            return MagicMock(returncode=1, communicate=AsyncMock(return_value=(b"lint err", b"")))
        return MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"", b"")))

    with patch("asyncio.create_subprocess_exec", fake_eslint_then_tsc):
        res = await frontend_tools.validate_frontend_code("Button.vue", "<template/>")
    assert res.startswith("VALIDATION FAILED:\nLinter Errors:\nlint err")
    assert all(cwd != server_cwd for cwd in observed_cwds)

    # Case 2: both eslint and vue-tsc fail -> both Compiler and Linter errors returned
    async def both_fail(*_args, **_kwargs):
        return MagicMock(returncode=1, communicate=AsyncMock(return_value=(b"bad", b"detail")))

    with patch("asyncio.create_subprocess_exec", both_fail):
        res_both = await frontend_tools.validate_frontend_code("Button.vue", "<template/>")
    assert "Compiler Errors:" in res_both and "Linter Errors:" in res_both

    # Case 3: linter timeout calls kill_and_reap
    hung_proc = MagicMock(communicate=AsyncMock(side_effect=TimeoutError), kill=MagicMock(), wait=AsyncMock())
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=hung_proc)):
        res_timeout = await frontend_tools.validate_frontend_code("Button.vue", "<template/>")
    assert res_timeout == "VALIDATION FAILED: Linter timed out."
    hung_proc.kill.assert_called_once()
    hung_proc.wait.assert_awaited_once()

    # Case 4: compiler timeout calls kill_and_reap
    step = 0

    async def eslint_fail_then_tsc_hang(*_args, **_kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return MagicMock(returncode=1, communicate=AsyncMock(return_value=(b"lint", b"")))
        return hung_proc

    hung_proc.reset_mock()
    with patch("asyncio.create_subprocess_exec", eslint_fail_then_tsc_hang):
        res_tsc_timeout = await frontend_tools.validate_frontend_code("Button.vue", "<template/>")
    assert res_tsc_timeout == "VALIDATION FAILED: Compiler timed out."
    hung_proc.kill.assert_called_once()

    # Case 5: missing npx binary handled cleanly
    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError("npx"))):
        res_missing = await frontend_tools.validate_frontend_code("Button.vue", "<template/>")
    assert "not installed" in res_missing


# ── 4. Dockerfile invariants: Go 1.26, bubblewrap, pinned npm, RO /opt ──────


def test_dockerfile_pins_go_126_frontend_tools_bubblewrap_and_readonly_opt():
    dockerfile = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text(encoding="utf-8")

    assert f"FROM golang:{settings.go_version}-bookworm AS gotoolchain" in dockerfile
    assert "bubblewrap" in dockerfile
    for pkg in ("eslint@9.39.5", "typescript@5.9.3", "vue-tsc@3.3.14", "vitest@4.1.12"):
        assert pkg in dockerfile
    assert "--ignore-scripts" in dockerfile
    assert "chmod -R a-w /opt/pydantic-agents" in dockerfile
    assert "chown agent:agent /opt/pydantic-agents" not in dockerfile
    assert "WORKDIR /var/lib/pydantic-agents" in dockerfile
    assert "ORCHESTRATION_CODEBASE_ROOT=/var/lib/pydantic-agents/workspace" in dockerfile
