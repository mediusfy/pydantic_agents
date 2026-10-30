"""Go module setup shared by the Go tools (build, vet and test execution)."""

import asyncio

from pydantic_agents.core.utils import kill_and_reap, sanitized_subprocess_env

GO_MOD_INIT_TIMEOUT = 10.0
GO_MOD_TIDY_TIMEOUT = 30.0


async def run_cmd(*args: str, cwd: str, timeout_seconds: float = 30.0) -> None:
    """Run a command to completion with a timeout, discarding its output; kill and reap it if it times out."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        env=sanitized_subprocess_env(),
        start_new_session=True,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        await kill_and_reap(proc)
        raise


async def init_and_tidy(temp_dir: str) -> str | None:
    """Initialize a go module and run go mod tidy, waiting for both. Returns an error string or None."""
    try:
        await run_cmd("go", "mod", "init", "tempmodule", cwd=temp_dir, timeout_seconds=GO_MOD_INIT_TIMEOUT)
    except FileNotFoundError:
        return "the 'go' toolchain is not installed."
    except TimeoutError:
        return "'go mod init' timed out."
    try:
        await run_cmd("go", "mod", "tidy", cwd=temp_dir, timeout_seconds=GO_MOD_TIDY_TIMEOUT)
    except TimeoutError:
        return (
            "'go mod tidy' timed out — the generated code likely imports "
            "a private or non-existent module. Check import paths."
        )
    return None
