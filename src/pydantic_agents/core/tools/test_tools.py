"""Test execution tools."""

import asyncio
import tempfile
from dataclasses import dataclass

from pydantic_agents.core.tools.go_common import init_and_tidy
from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import WRITE_NAME_ERRORS, kill_and_reap, safe_write, sanitized_subprocess_env

TEST_EXECUTION_TIMEOUT = 60.0


@dataclass(frozen=True)
class TestRunResult:
    """What running a test suite produced. ``passed`` is the only verdict: ``error`` says the suite could not be
    run at all (toolchain missing, module setup failed, timeout, unwritable names), which also counts as a failure."""

    __test__ = False

    passed: bool
    output: str
    error: str = ""

    def trace(self) -> str:
        """The legacy string form: SUCCESS/FAILED plus the output, or TEST EXECUTION FAILED plus the error."""
        if self.error:
            return f"TEST EXECUTION FAILED: {self.error}"
        return f"{'SUCCESS' if self.passed else 'FAILED'}\n{self.output}"


async def run_test_suite(files: dict[str, str], test_files: dict[str, str]) -> TestRunResult:
    """Write the source files and the test files into one temporary module and run its tests.

    Go test files (``*_test.go``) run with ``go test ./...`` against every source file given, so a suite that
    touches more than one file compiles; anything else runs with vitest.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            for name, content in {**files, **test_files}.items():
                await asyncio.to_thread(safe_write, temp_dir, name, content)
        except WRITE_NAME_ERRORS as e:
            return TestRunResult(False, "", f"cannot write the given files: {e}")

        go_tests = [name for name in test_files if name.endswith(".go")]
        if go_tests:
            err = await init_and_tidy(temp_dir)
            if err:
                return TestRunResult(False, "", err)
            cmd = ["go", "test", "-v", "-count=1", "./..."]
        else:
            cmd = ["npx", "vitest", "run", *(str(safe_write(temp_dir, n, c)) for n, c in test_files.items())]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=temp_dir,
                env=sanitized_subprocess_env(),
                start_new_session=True,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            return TestRunResult(False, "", f"test runner not installed: {e}")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TEST_EXECUTION_TIMEOUT)
        except TimeoutError:
            await kill_and_reap(proc)
            return TestRunResult(False, "", "Timed out.")

        output = f"{stdout.decode()}\n{stderr.decode()}".strip()
        return TestRunResult(proc.returncode == 0, output)


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
        The full execution trace of the test suite (stdout and stderr), starting with SUCCESS, FAILED or
        TEST EXECUTION FAILED.
    """
    result = await run_test_suite({target_code_file: target_source_code}, {test_file_name: test_source_code})
    return result.trace()


register_tool(
    "execute_test_suite",
    execute_test_suite,
    "Executes a test suite against target code files to verify correctness. Returns the trace.",
    ["test_run"],
)
