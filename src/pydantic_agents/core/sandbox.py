"""Namespace and filesystem sandbox for untrusted build, lint, and test subprocesses.

Untrusted model-generated code (``go test``, ``go build``, ``go vet``, ``vitest``, ``eslint``, ``vue-tsc``) must not
share a PID, mount, or network namespace with the MCP server: on Linux a child running under the same UID and PID
namespace can read ``/proc/<ppid>/environ`` (exfiltrating ``LITELLM_VIRTUAL_KEY``, ``DATABASE_URL``, and Langfuse keys)
or reach internal cluster services.

When ``bwrap`` (bubblewrap) is available and unprivileged namespaces are permitted by the kernel:
- ``--unshare-all`` isolates user (mapped to ``65534:65534``), mount, PID, IPC, UTS, and network namespaces
  (``--share-net`` is granted ONLY when ``allow_network=True`` for ``go mod tidy`` before any untrusted code runs).
- ``--tmpfs /`` starts from an empty root so ``/opt/pydantic-agents``, ``/var/lib/pydantic-agents``, and ``/home/agent``
  do not exist inside the sandbox at all; only system toolchain paths are bound read-only and the ephemeral ``cwd``
  workspace is bound read-write.
- ``--proc /proc`` mounts a fresh procfs for the child PID namespace so the MCP server's PID and ``environ`` are
  invisible.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic_agents.core.exceptions import SandboxUnavailableError
from pydantic_agents.core.settings import settings
from pydantic_agents.core.utils import sanitized_subprocess_env

logger = logging.getLogger(__name__)

_SYSTEM_RO_PATHS: tuple[str, ...] = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc/alternatives",
    "/etc/ssl",
    "/etc/ca-certificates",
    "/etc/pki",
)
_NETWORK_RO_PATHS: tuple[str, ...] = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
)

# Private scratch space inside the sandbox. A fresh tmpfs is mounted at the host's temp-dir path (normally /tmp), so
# nothing the host keeps there is visible to the child; the ephemeral workspace, created by tempfile under that same
# path, is then bound back in on top of it. The directory is never shared with the host or other sandboxes.
_SANDBOX_TMP = Path(tempfile.gettempdir())
_SANDBOX_HOME = _SANDBOX_TMP / "home"
_SANDBOX_GOCACHE = _SANDBOX_TMP / "go-cache"
_SANDBOX_NPM_CACHE = _SANDBOX_TMP / "npm-cache"
_DEFAULT_GOPATH = _SANDBOX_TMP / "go"

_probe_cache: dict[str, tuple[bool, str]] = {}
_warned_auto_fallback: bool = False


def clear_sandbox_probe_cache() -> None:
    """Reset cached bwrap capability results (used by tests)."""
    global _warned_auto_fallback
    _probe_cache.clear()
    _warned_auto_fallback = False


def _mount_or_symlink_args(paths: Sequence[str]) -> list[str]:
    """Emit ``--symlink`` for usrmerge links (e.g. ``/bin -> usr/bin``) and ``--ro-bind`` for real paths."""
    args: list[str] = []
    for path in paths:
        if os.path.islink(path):
            args.extend(["--symlink", os.readlink(path), path])
        elif os.path.exists(path):
            args.extend(["--ro-bind", path, path])
    return args


def _system_mount_args(*, allow_network: bool) -> list[str]:
    paths = (*_SYSTEM_RO_PATHS, *_NETWORK_RO_PATHS) if allow_network else _SYSTEM_RO_PATHS
    return _mount_or_symlink_args(paths)


def _is_within(path: Path, roots: Sequence[str]) -> bool:
    return any(path == Path(root) or Path(root) in path.parents for root in roots)


def _toolchain_mount_args(program: str, search_path: str | None) -> list[str]:
    """Read-only binds for a toolchain installed outside the system paths (e.g. ``/opt/hostedtoolcache/go/...``).

    The image keeps Go under ``/usr/local/go`` (already covered by ``/usr``), but CI runners and dev machines often
    install toolchains under ``/opt`` or a home directory. Without this the sandboxed child fails with
    ``execvp go: No such file or directory``. The bound directory is the toolchain prefix: the parent of the ``bin``
    directory holding the executable, both as found on ``PATH`` and after resolving symlinks.
    """
    found = shutil.which(program, path=search_path)
    if not found:
        return []
    args: list[str] = []
    seen: set[Path] = set()
    for exe in (Path(found).absolute(), Path(os.path.realpath(found))):
        prefix = exe.parent.parent
        if prefix in seen or len(prefix.parts) < 3 or _is_within(prefix, _SYSTEM_RO_PATHS):
            continue
        seen.add(prefix)
        args.extend(["--ro-bind", str(prefix), str(prefix)])
    return args


def probe_bwrap(bwrap_bin: str | None = None) -> tuple[bool, str]:
    """Return ``(True, "")`` if ``bwrap`` can create unprivileged namespaces on this host, else ``(False, reason)``."""
    resolved = bwrap_bin or shutil.which("bwrap")
    if not resolved:
        return False, "'bwrap' binary is not installed"
    if resolved in _probe_cache:
        return _probe_cache[resolved]

    true_bin = shutil.which("true") or "/usr/bin/true"
    probe_cmd = [
        resolved,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--uid",
        "65534",
        "--gid",
        "65534",
        "--tmpfs",
        "/",
        *_system_mount_args(allow_network=False),
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--",
        true_bin,
    ]
    try:
        completed = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5.0, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        outcome = (False, str(exc))
        _probe_cache[resolved] = outcome
        return outcome

    if completed.returncode == 0:
        outcome = (True, "")
    else:
        detail = (completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip()
        outcome = (False, detail)
    _probe_cache[resolved] = outcome
    return outcome


def build_bwrap_command(
    cmd: Sequence[str],
    *,
    cwd: str | Path,
    allow_network: bool = False,
    bwrap_bin: str = "bwrap",
    gopath_dir: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build the ``bwrap`` argv and isolated environment for running ``cmd`` inside ``cwd``."""
    work_dir = str(Path(cwd).resolve())
    env = sanitized_subprocess_env()
    gopath = str(Path(gopath_dir or env.get("GOPATH") or _DEFAULT_GOPATH).resolve())
    scratch = str(_SANDBOX_TMP)

    argv: list[str] = [bwrap_bin, "--unshare-all"]
    if allow_network:
        argv.append("--share-net")
    argv.extend(
        [
            "--die-with-parent",
            "--new-session",
            "--uid",
            "65534",
            "--gid",
            "65534",
            "--tmpfs",
            "/",
            *_system_mount_args(allow_network=allow_network),
            *_toolchain_mount_args(cmd[0], env.get("PATH")),
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            scratch,
            "--dir",
            str(_SANDBOX_HOME),
            "--dir",
            str(_SANDBOX_GOCACHE),
            "--dir",
            str(_SANDBOX_NPM_CACHE),
        ]
    )

    if gopath != work_dir:
        if allow_network:
            # Only `go mod tidy` asks for the network, and it runs before any generated code executes (no build, test
            # or vet has run yet). Module downloads go through settings.untrusted_goproxy / untrusted_gosumdb, which
            # sanitized_subprocess_env() pins with GOENV=off so the module cannot redirect them.
            os.makedirs(gopath, exist_ok=True)
            argv.extend(["--bind", gopath, gopath])
        elif os.path.isdir(gopath):
            argv.extend(["--ro-bind", gopath, gopath])

    argv.extend(["--bind", work_dir, work_dir, "--chdir", work_dir, "--", *cmd])
    env.update(
        {
            "HOME": str(_SANDBOX_HOME),
            "TMPDIR": scratch,
            "TEMP": scratch,
            "TMP": scratch,
            "GOCACHE": str(_SANDBOX_GOCACHE),
            "NPM_CONFIG_CACHE": str(_SANDBOX_NPM_CACHE),
            "GOPATH": gopath,
        }
    )
    return argv, env


def _effective_backend() -> str:
    raw = os.environ.get("SANDBOX_BACKEND")
    return (raw if raw is not None else settings.sandbox_backend).strip().lower()


def resolve_sandbox_execution(
    cmd: Sequence[str],
    *,
    cwd: str | Path,
    allow_network: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Resolve ``(argv, env)`` according to ``SANDBOX_BACKEND`` ('auto' | 'bwrap' | 'none')."""
    global _warned_auto_fallback

    if not cmd:
        raise ValueError("cmd must not be empty")
    mode = _effective_backend()
    if mode == "none":
        return list(cmd), sanitized_subprocess_env()
    if mode not in ("auto", "bwrap"):
        raise SandboxUnavailableError(f"Unknown SANDBOX_BACKEND {mode!r}; expected 'auto', 'bwrap', or 'none'.")

    bwrap_bin = shutil.which("bwrap")
    ok, reason = probe_bwrap(bwrap_bin)
    if ok and bwrap_bin:
        return build_bwrap_command(cmd, cwd=cwd, allow_network=allow_network, bwrap_bin=bwrap_bin)
    if mode == "bwrap":
        raise SandboxUnavailableError(f"Sandbox backend 'bwrap' is unavailable: {reason}")

    if not _warned_auto_fallback:
        _warned_auto_fallback = True
        # On Linux (where production runs) losing the namespace sandbox re-exposes /proc/<ppid>/environ, so it is an
        # error; pin SANDBOX_BACKEND=bwrap in production to fail closed instead. On macOS dev machines it is expected.
        log_fn = logger.error if sys.platform == "linux" else logger.debug
        log_fn(
            "bwrap namespace sandbox unavailable (%s); falling back to sanitized-env subprocess isolation. "
            "Set SANDBOX_BACKEND=bwrap in production to fail closed.",
            reason,
        )
    return list(cmd), sanitized_subprocess_env()


def unwrap_sandbox_command(args: Sequence[str]) -> tuple[str, ...]:
    """Return the inner command from an argv that may be wrapped in ``bwrap ... -- <cmd>``."""
    if args and os.path.basename(args[0]) == "bwrap":
        for idx, token in enumerate(args):
            if token == "--":
                return tuple(args[idx + 1 :])
    return tuple(args)


async def spawn_untrusted_subprocess(
    *cmd: str,
    cwd: str | Path,
    allow_network: bool = False,
    stdout: int | None = asyncio.subprocess.PIPE,
    stderr: int | None = asyncio.subprocess.PIPE,
) -> asyncio.subprocess.Process:
    """Spawn an untrusted build/lint/test command inside the configured sandbox."""
    argv, env = resolve_sandbox_execution(cmd, cwd=cwd, allow_network=allow_network)
    return await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=env,
        start_new_session=True,
        stdout=stdout,
        stderr=stderr,
    )
