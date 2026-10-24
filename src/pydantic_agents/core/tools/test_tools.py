"""Test execution tools."""

import asyncio
import contextlib
import tempfile

from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import safe_write

TEST_EXECUTION_TIMEOUT = 60.0


async def execute_test_suite(
    test_file_name: str, test_source_code: str, target_code_file: str, target_source_code: str
) -> str:
    """
    Writes test files and target code files to a temporary workspace and runs tests.

    Args:
        test_file_name: The name of the test file (e.g. 'service_test.go' or 'Component.spec.ts').
        test_source_code: The string content of the test file.
        target_code_file: The name of the file being tested (e.g. 'service.go' or 'Component.vue').
        target_source_code: The string content of the target file.

    Returns:
        The full execution trace of the test suite (stdout and stderr).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write files
        try:
            test_path = await asyncio.to_thread(safe_write, temp_dir, test_file_name, test_source_code)
            await asyncio.to_thread(safe_write, temp_dir, target_code_file, target_source_code)
        except PathTraversalError as e:
            return f"TEST EXECUTION FAILED: {e}"

        # Determine the test runner based on file extension
        if test_file_name.endswith(".go"):
            # Go tests
            # We must init a module so the test can run
            await asyncio.create_subprocess_exec("go", "mod", "init", "tempmodule", cwd=temp_dir)
            await asyncio.create_subprocess_exec("go", "mod", "tidy", cwd=temp_dir)

            proc = await asyncio.create_subprocess_exec(
                "go",
                "test",
                "-v",
                "./...",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            # Assume JS/TS (vitest)
            proc = await asyncio.create_subprocess_exec(
                "npx",
                "vitest",
                "run",
                str(test_path),
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TEST_EXECUTION_TIMEOUT)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return "TEST EXECUTION FAILED: Timed out."

        output = f"{stdout.decode()}\n{stderr.decode()}".strip()

        if proc.returncode == 0:
            return f"SUCCESS\n{output}"
        else:
            return f"FAILED\n{output}"


register_tool(
    "execute_test_suite",
    execute_test_suite,
    "Executes a test suite against target code files to verify correctness. Returns the trace.",
    ["test_run"],
)
