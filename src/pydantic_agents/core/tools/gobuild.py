"""Go build and vet tools for shift-left validation during code generation."""

import asyncio
import contextlib
import tempfile

from pydantic_agents.core.tools.go_common import init_and_tidy
from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import WRITE_NAME_ERRORS, safe_write

GO_BUILD_TIMEOUT = 60.0
GO_VET_TIMEOUT = 60.0


async def _write_files(temp_dir: str, files: dict[str, str]) -> None:
    """Write a dict of file paths to content into temp_dir; raises (see WRITE_NAME_ERRORS) if a name escapes it or is invalid."""
    for filename, content in files.items():
        safe_write(temp_dir, filename, content)


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
