import asyncio
import glob as pyglob
import os
import re
from pathlib import Path

from pydantic_ai import RunContext

from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import resolve_sandboxed, sandbox_root


async def search_codebase(ctx: RunContext[str], pattern: str, path: str = ".", include: str = "*.go") -> list[str]:
    """Search the codebase for a text pattern using ripgrep.

    Args:
        pattern: The text pattern or regex to search for.
        path: The directory path to search in (defaults to current directory ".").
        include: Glob pattern for files to include (e.g. "*.go", "*.py"). Defaults to "*.go".
    """
    try:
        try:
            full_path = resolve_sandboxed(path)
        except PathTraversalError:
            return ["Error: Path traversal detected. You can only search within the workspace."]

        # -e and -- stop model-chosen text from being parsed as rg flags (e.g. --pre=<cmd>).
        proc = await asyncio.create_subprocess_exec(
            "rg",
            "-l",
            f"--glob={include}",
            "-e",
            pattern,
            "--",
            str(full_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        except TimeoutError:
            proc.kill()
            return ["Error: Search timed out after 60 seconds. Path might be too large."]
        if proc.returncode == 0:
            return stdout.decode().strip().splitlines()
        elif proc.returncode == 1:
            return []
        else:
            return [f"rg error: {stderr.decode().strip()}"]
    except FileNotFoundError:
        return ["ripgrep (rg) not installed - consider installing it or use search_pattern_in_content"]
    except Exception as e:
        return [f"Search failed: {e}"]


async def search_pattern_in_content(ctx: RunContext[str], pattern: str, content: str) -> list[dict]:
    """Search a string for a regex pattern, returning line matches.

    Args:
        pattern: The regex pattern to search for.
        content: The actual multiline text content to search inside.
    """
    results = []
    for i, line in enumerate(content.splitlines(), 1):
        if re.search(pattern, line):
            results.append({"line": i, "content": line.strip()})
    return results


async def search_files(ctx: RunContext[str], glob: str) -> list[str]:
    """Search for files matching a glob pattern relative to the workspace (e.g. '**/*.go')."""
    root = sandbox_root()
    if os.path.isabs(glob) or ".." in Path(glob).parts:
        return []
    matches = pyglob.glob(glob, root_dir=root, recursive=True)
    return [m for m in matches if (root / m).resolve().is_relative_to(root)]


register_tool(
    "search_codebase",
    search_codebase,
    "Search the codebase for a text pattern using ripgrep",
    ["search", "security", "code_review"],
)
register_tool(
    "search_pattern_in_content",
    search_pattern_in_content,
    "Search a string for regex pattern, returns line matches",
    ["search", "security"],
)
register_tool("search_files", search_files, "Search the filesystem using a glob pattern", ["search", "code_review"])
