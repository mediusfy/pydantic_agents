import os

from pydantic_agents.core.exceptions import PathTraversalError
from pydantic_agents.core.tools.registry import register_tool
from pydantic_agents.core.utils import resolve_sandboxed, safe_write, sandbox_root

_READ_FILE_MAX_BYTES = 256 * 1024


def read_file(filepath: str) -> str:
    """Read the contents of a file from the repository. Useful for reusing existing code."""
    try:
        path = resolve_sandboxed(filepath)
    except PathTraversalError as e:
        return f"Error reading file: {e}"
    try:
        content = path.read_text(errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    if len(content) > _READ_FILE_MAX_BYTES:
        return content[:_READ_FILE_MAX_BYTES] + f"\n... (truncated at {_READ_FILE_MAX_BYTES} bytes)"
    return content


def list_dir(directory: str = ".") -> str:
    """List all files in the given directory. Useful for exploring the repository."""
    try:
        return "\n".join(os.listdir(resolve_sandboxed(directory)))
    except Exception as e:
        return f"Error listing directory: {e}"


def write_file(filepath: str, content: str) -> str:
    """Write or overwrite a file with the given content."""
    try:
        root = sandbox_root()
        # Re-validate through resolve_sandboxed so absolute paths inside the root are accepted.
        relative = resolve_sandboxed(filepath).relative_to(root)
        safe_write(root, str(relative), content)
        return f"Successfully wrote {len(content)} characters to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


# Read access is limited to agents that inspect code. write_file lives in its own category that no agent is
# assigned by default: Artifact-producing agents hand their files to the pipeline writer instead.
register_tool(
    "read_file",
    read_file,
    "Read the contents of a file from the repository. Useful for understanding existing codebase or finding reusable utilities.",
    ["fs_read"],
)

register_tool(
    "list_dir",
    list_dir,
    "List all files in the given directory. Useful for exploring the repository layout.",
    ["fs_read"],
)

register_tool(
    "write_file",
    write_file,
    "Write or overwrite a file with the given content.",
    ["fs_write"],
)
