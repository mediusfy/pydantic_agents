import glob
import os

from pydantic_agents.core.tools.registry import register_tool


def read_file(filepath: str) -> str:
    """Read the contents of a file from the repository. Useful for reusing existing code."""
    try:
        with open(filepath) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def list_dir(directory: str = ".") -> str:
    """List all files in the given directory. Useful for exploring the repository."""
    try:
        return "\n".join(os.listdir(directory))
    except Exception as e:
        return f"Error listing directory: {e}"


def search_files(pattern: str) -> str:
    """Search for files matching a glob pattern (e.g. '**/*.go')."""
    try:
        matches = glob.glob(pattern, recursive=True)
        return "\n".join(matches) if matches else "No files found."
    except Exception as e:
        return f"Error searching files: {e}"


def write_file(filepath: str, content: str) -> str:
    """Write or overwrite a file with the given content."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


# File system tools are globally useful; register them across all common categories
categories = [
    "backend",
    "frontend",
    "test_eng",
    "implementation",
    "docs",
    "devops",
    "sre",
    "database",
    "search",
    "code_review",
    "ba",
    "architecture",
]

register_tool(
    "read_file",
    read_file,
    "Read the contents of a file from the repository. Useful for understanding existing codebase or finding reusable utilities.",
    categories,
)

register_tool(
    "write_file",
    write_file,
    "Write or overwrite a file with the given content.",
    categories,
)

register_tool(
    "list_dir",
    list_dir,
    "List all files in the given directory. Useful for exploring the repository layout.",
    categories,
)

register_tool(
    "search_files",
    search_files,
    "Search for files matching a glob pattern (e.g. '**/*.go' or 'src/components/*.vue').",
    categories,
)
