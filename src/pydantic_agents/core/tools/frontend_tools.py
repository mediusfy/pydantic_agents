"""Frontend build and validation tools."""

import asyncio
import contextlib
import os
import tempfile

from pydantic_agents.core.tools.registry import register_tool

FRONTEND_VALIDATE_TIMEOUT = 60.0


async def validate_frontend_code(file_name: str, source_code: str) -> str:
    """
    Validates a frontend component (e.g., Vue or TSX) by writing it to a
    temporary directory and running 'npx eslint' or 'npx vue-tsc'.

    Args:
        file_name: The name of the file (e.g. 'Button.vue')
        source_code: The string content of the component.

    Returns:
        'SUCCESS' if validation passes, otherwise returns the compiler/linter error string.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, file_name)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(source_code)

        # Determine command based on extension. Default to vue-tsc or eslint.
        # Running against the filepath but keeping CWD as the project root so it
        # inherits the project's tsconfig.json and node_modules.
        cwd = os.getcwd()

        proc = await asyncio.create_subprocess_exec(
            "npx",
            "eslint",
            filepath,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=FRONTEND_VALIDATE_TIMEOUT)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return "VALIDATION FAILED: Linter timed out."

        if proc.returncode == 0:
            return "SUCCESS"

        # Fallback for compilation
        tsc_proc = await asyncio.create_subprocess_exec(
            "npx",
            "vue-tsc",
            "--noEmit",
            filepath,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            tsc_stdout, tsc_stderr = await asyncio.wait_for(tsc_proc.communicate(), timeout=FRONTEND_VALIDATE_TIMEOUT)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                tsc_proc.kill()
            return "VALIDATION FAILED: Compiler timed out."

        if tsc_proc.returncode == 0:
            # If tsc passes but eslint fails, we still return the eslint error to be strict
            return f"VALIDATION FAILED:\nLinter Errors:\n{stdout.decode()}\n{stderr.decode()}"

        return f"VALIDATION FAILED:\nCompiler Errors:\n{tsc_stdout.decode()}\n{tsc_stderr.decode()}\nLinter Errors:\n{stdout.decode()}"


register_tool(
    "validate_frontend_code",
    validate_frontend_code,
    "Validates a Vue/TSX component using the local compiler/linter. Must be used before returning final code.",
    ["frontend", "validation"],
)
