import asyncio
import os

from pydantic_ai import RunContext

from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import safe_path_join


def _write_doc(full_path, content: str, *, overwrite: bool) -> None:
    os.makedirs(os.path.dirname(str(full_path)), exist_ok=True)
    if overwrite:
        with open(full_path, "w") as f:
            f.write(content)
    else:
        with open(full_path, "a") as f:
            f.write(f"\n\n{content}")


async def update_documentation(ctx: RunContext, file_path: str, content: str, mode: str = "append") -> str:
    """Updates project documentation. Use this to proactively document decisions or implementation details."""
    base_docs_dir = os.path.realpath("docs")
    safe_file = file_path.lstrip("/")
    try:
        full_path = safe_path_join(base_docs_dir, safe_file)
    except (PathTraversalError, ValueError) as e:
        return f"Error: {e}"

    await asyncio.to_thread(_write_doc, full_path, content, overwrite=(mode == "overwrite"))
    return f"Documentation updated: {file_path}"


register_tool("update_documentation", update_documentation, "Updates project documentation", ["documentation"])
