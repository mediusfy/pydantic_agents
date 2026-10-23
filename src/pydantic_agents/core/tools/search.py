import asyncio
import glob as pyglob
import os
import re
from typing import Any

from pydantic_ai import RunContext

from pydantic_agents.core.tools.registry import register_tool


async def search_codebase(ctx: RunContext[Any], pattern: str, path: str = ".", include: str = "*.go") -> list[str]:
    """Search the codebase for a text pattern using ripgrep.

    Args:
        pattern: The text pattern or regex to search for.
        path: The directory path to search in (defaults to current directory ".").
        include: Glob pattern for files to include (e.g. "*.go", "*.py"). Defaults to "*.go".
    """
    try:
        base_dir = os.path.realpath(".")
        full_path = os.path.realpath(path)
        if not full_path.startswith(base_dir):
            return ["Error: Path traversal detected. You can only search within the workspace."]

        proc = await asyncio.create_subprocess_exec(
            "rg",
            "-l",
            "--glob",
            include,
            pattern,
            full_path,
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


async def search_pattern_in_content(ctx: RunContext[Any], pattern: str, content: str) -> list[dict]:
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


async def search_files(ctx: RunContext[Any], glob: str) -> list[str]:
    return pyglob.glob(glob, recursive=True)


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
