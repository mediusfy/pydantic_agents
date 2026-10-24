"""Go build and vet tools for shift-left validation during code generation."""

import asyncio
import contextlib
import tempfile

from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import WRITE_NAME_ERRORS, safe_write

GO_MOD_INIT_TIMEOUT = 10.0
GO_MOD_TIDY_TIMEOUT = 30.0
GO_BUILD_TIMEOUT = 60.0
GO_VET_TIMEOUT = 60.0


async def _run_cmd(*args: str, cwd: str, timeout: float = 30.0) -> None:
    """Run a command with timeout, discarding output."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise


async def _write_files(temp_dir: str, files: dict[str, str]) -> None:
    """Write a dict of file paths to content into temp_dir; raises (see WRITE_NAME_ERRORS) if a name escapes it or is invalid."""
    for filename, content in files.items():
        safe_write(temp_dir, filename, content)


async def init_and_tidy(temp_dir: str) -> str | None:
    """Initialize a go module and run go mod tidy, waiting for both. Returns an error string or None."""
    try:
        await _run_cmd("go", "mod", "init", "tempmodule", cwd=temp_dir, timeout=GO_MOD_INIT_TIMEOUT)
    except FileNotFoundError:
        return "the 'go' toolchain is not installed."
    except TimeoutError:
        return "'go mod init' timed out."
    try:
        await _run_cmd("go", "mod", "tidy", cwd=temp_dir, timeout=GO_MOD_TIDY_TIMEOUT)
    except TimeoutError:
        return (
            "'go mod tidy' timed out — the generated code likely imports "
            "a private or non-existent module. Check import paths."
        )
    return None


async def _run_build(temp_dir: str) -> str:
    """Run go build and return result string."""
    build_proc = await asyncio.create_subprocess_exec(
        "go",
        "build",
        "./...",
        cwd=temp_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(build_proc.communicate(), timeout=GO_BUILD_TIMEOUT)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            build_proc.kill()
        return "BUILD FAILED: 'go build' timed out (possible circular dependency or slow proxy)."

    if build_proc.returncode == 0:
        return "SUCCESS: Code compiled successfully."
    return f"BUILD FAILED:\n{stderr.decode()}"


async def _run_vet(temp_dir: str) -> str:
    """Run go vet and return result string."""
    vet_proc = await asyncio.create_subprocess_exec(
        "go",
        "vet",
        "./...",
        cwd=temp_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(vet_proc.communicate(), timeout=GO_VET_TIMEOUT)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            vet_proc.kill()
        return "VET FAILED: 'go vet' timed out."

    if vet_proc.returncode == 0:
        return "SUCCESS: Code passed go vet."
    return f"VET FAILED:\n{stderr.decode()}"


async def run_go_build(files: dict[str, str]) -> str:
    """
    Writes the provided Go files to a temporary directory and runs
    'go mod init', 'go mod tidy', and 'go build ./...'.

    Returns 'SUCCESS' if it compiles, or the compiler error output if it fails.

    Args:
        files: A dictionary mapping file paths (e.g., 'main.go') to their string content.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            await _write_files(temp_dir, files)
        except WRITE_NAME_ERRORS as e:
            return f"BUILD FAILED: cannot write the given files: {e}"
        err = await init_and_tidy(temp_dir)
        if err:
            return f"BUILD FAILED: {err}"
        return await _run_build(temp_dir)


async def run_go_vet(files: dict[str, str]) -> str:
    """
    Writes the provided Go files to a temporary directory and runs
    'go mod init', 'go mod tidy', and 'go vet ./...'.

    Returns 'SUCCESS' if it passes, or the vet error output if it fails.

    Args:
        files: A dictionary mapping file paths (e.g., 'main.go') to their string content.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            await _write_files(temp_dir, files)
        except WRITE_NAME_ERRORS as e:
            return f"VET FAILED: cannot write the given files: {e}"
        err = await init_and_tidy(temp_dir)
        if err:
            return f"VET FAILED: {err}"
        return await _run_vet(temp_dir)


register_tool(
    "run_go_build",
    run_go_build,
    "Compile Go code to check for syntax and import errors. Pass a dict of filename -> content.",
    ["go_build"],
)

register_tool(
    "run_go_vet",
    run_go_vet,
    "Run go vet on Go code to check for suspicious constructs. Pass a dict of filename -> content.",
    ["go_build"],
)
