"""Frontend build and validation tools."""

import asyncio
import tempfile

from pydantic_agents.core.exceptions import SandboxUnavailableError
from pydantic_agents.core.sandbox import spawn_untrusted_subprocess
from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import WRITE_NAME_ERRORS, kill_and_reap, safe_write

FRONTEND_VALIDATE_TIMEOUT = 60.0
_MAX_OUTPUT_CHARS = 16_000


async def validate_frontend_code(file_name: str, source_code: str) -> str:
    """
    Validates a frontend component (e.g., Vue or TSX) by writing it to an isolated
    temporary directory and running offline 'npx --no-install eslint' / 'npx --no-install vue-tsc'.

    Args:
        file_name: The name of the file (e.g. 'Button.vue')
        source_code: The string content of the component.

    Returns:
        'SUCCESS' if validation passes, otherwise returns the compiler/linter error string.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            filepath = str(safe_write(temp_dir, file_name, source_code))
        except WRITE_NAME_ERRORS as e:
            return f"VALIDATION FAILED: cannot write {file_name!r}: {e}"

        try:
            proc = await spawn_untrusted_subprocess(
                "npx",
                "--no-install",
                "eslint",
                filepath,
                cwd=temp_dir,
                allow_network=False,
            )
        except SandboxUnavailableError as e:
            return f"VALIDATION FAILED: {e}"
        except FileNotFoundError:
            return "VALIDATION FAILED: frontend linter ('npx' / 'eslint') is not installed."

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=FRONTEND_VALIDATE_TIMEOUT)
        except TimeoutError:
            await kill_and_reap(proc)
            return "VALIDATION FAILED: Linter timed out."

        if proc.returncode == 0:
            return "SUCCESS"

        eslint_out = f"{stdout.decode(errors='replace')}\n{stderr.decode(errors='replace')}".strip()

        try:
            tsc_proc = await spawn_untrusted_subprocess(
                "npx",
                "--no-install",
                "vue-tsc",
                "--noEmit",
                filepath,
                cwd=temp_dir,
                allow_network=False,
            )
        except SandboxUnavailableError as e:
            return f"VALIDATION FAILED: {e}"
        except FileNotFoundError:
            return f"VALIDATION FAILED:\nLinter Errors:\n{eslint_out[-_MAX_OUTPUT_CHARS:]}"

        try:
            tsc_stdout, tsc_stderr = await asyncio.wait_for(tsc_proc.communicate(), timeout=FRONTEND_VALIDATE_TIMEOUT)
        except TimeoutError:
            await kill_and_reap(tsc_proc)
            return "VALIDATION FAILED: Compiler timed out."

        if tsc_proc.returncode == 0:
            # If tsc passes but eslint fails, we still return the eslint error to be strict
            return f"VALIDATION FAILED:\nLinter Errors:\n{eslint_out[-_MAX_OUTPUT_CHARS:]}"

        tsc_out = f"{tsc_stdout.decode(errors='replace')}\n{tsc_stderr.decode(errors='replace')}".strip()
        return (
            f"VALIDATION FAILED:\nCompiler Errors:\n{tsc_out[-_MAX_OUTPUT_CHARS:]}\n"
            f"Linter Errors:\n{eslint_out[-_MAX_OUTPUT_CHARS:]}"
        )


register_tool(
    "validate_frontend_code",
    validate_frontend_code,
    "Validates a Vue/TSX component using the local compiler/linter. Must be used before returning final code.",
    ["frontend_validation"],
)
