import os
from pathlib import Path
from typing import Any

from pydantic_core import to_json

from pydantic_agents.core.exceptions import PathTraversalError


def safe_path_join(base: str | Path, *parts: str) -> Path:
    """Join path components and verify the result stays within base.
    Raises PathTraversalError if the resolved path escapes base.
    """
    base_resolved = Path(base).resolve()
    full = base_resolved.joinpath(*parts).resolve()
    if not full.is_relative_to(base_resolved):
        raise PathTraversalError(f"Path traversal detected: {'/'.join(parts)} escapes {base}")
    return full


# What writing a model-chosen file name can raise: an escape (PathTraversalError), or an invalid name such as a NUL byte
# (ValueError), a directory, a missing parent or a name that is too long (OSError). Tools return a failure message for
# these instead of raising.
WRITE_NAME_ERRORS: tuple[type[BaseException], ...] = (PathTraversalError, OSError, ValueError)


def sandbox_root() -> Path:
    """Root that model-driven file tools may touch.

    Read from ORCHESTRATION_CODEBASE_ROOT (set by run_existing_code_pipeline), with
    CWD as the safe local-dev default.
    """
    return Path(os.environ.get("ORCHESTRATION_CODEBASE_ROOT", os.getcwd())).resolve()


def resolve_sandboxed(user_path: str) -> Path:
    """Resolve a model-chosen path (relative or absolute) and require it to stay inside the sandbox root."""
    root = sandbox_root()
    candidate = Path(user_path) if os.path.isabs(user_path) else root / user_path
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise PathTraversalError(f"Path '{user_path}' escapes sandbox root {root}")
    return candidate


def safe_write(base: str | Path, relative_path: str, content: str) -> Path:
    """Write content to a file at base/relative_path, preventing traversal."""
    full_path = safe_path_join(base, relative_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    return full_path


def serialize_output(result: Any) -> str:
    """Render an agent run result as text: strings as-is, models/lists/dicts as JSON.

    Delegating agents and MCP clients parse this, so a BaseModel must not be returned as its Python repr.
    """
    output = getattr(result, "output", result)
    if isinstance(output, str):
        return output
    return to_json(output, fallback=str).decode()


def _artifact_header(art: Any) -> str:
    story_id = art.metadata.get("user_story_id", "N/A")
    return f"=== Agent: {art.agent_name} | Story: {story_id} ==="


def _append_artifact_files(parts: list[str], art: Any, total_len: int, max_chars: int) -> int:
    """Append an artifact's files to ``parts`` until the character budget runs out; returns the new running length."""
    for path, content in art.files.items():
        snippet = f"--- {path} ---\n{content}"
        if total_len + len(snippet) > max_chars:
            snippet = snippet[: max_chars - total_len] + "\n... (truncated)"
        parts.append(snippet)
        total_len += len(snippet)
        if total_len >= max_chars:
            break
    return total_len


def format_artifacts(artifacts: list[Any], max_chars: int = 80_000, include_agent_header: bool = True) -> str:
    """Format a list of Artifacts into a single prompt-friendly string.

    Args:
        artifacts: List of Artifact objects (each has .agent_name, .files, .metadata).
        max_chars: Maximum total characters (truncates if exceeded).
        include_agent_header: Whether to prefix each artifact with agent/story info.

    Returns:
        Formatted string suitable for downstream agent prompts.
    """
    parts: list[str] = []
    total_len = 0
    for art in artifacts:
        if isinstance(art, Exception):
            snippet = f"[Error] {art}"
            parts.append(snippet)
            total_len += len(snippet)
            continue

        if include_agent_header:
            header = _artifact_header(art)
            if total_len + len(header) > max_chars:
                parts.append("... (truncated)")
                break
            parts.append(header)
            total_len += len(header)

        total_len = _append_artifact_files(parts, art, total_len, max_chars)
        if total_len >= max_chars:
            parts.append("... (artifact output truncated, review continues on remaining items)")
            break
    return "\n\n".join(parts)


def parse_code_blocks_to_files(text: str) -> dict[str, str]:
    """Parse markdown code blocks and extract filenames from headers or inline path comments."""
    import re

    files = {}
    lines = text.split("\n")
    i = 0
    current_file = None
    block_lines: list[str] = []
    in_block = False

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("```"):
            if in_block:
                if current_file:
                    files[current_file] = "\n".join(block_lines)
                in_block = False
                current_file = None
                block_lines = []
            else:
                in_block = True
                found_file = None
                for offset in range(1, 4):
                    if i - offset >= 0:
                        prev_line = lines[i - offset].strip()
                        prev_line_clean = prev_line.strip("*`#:").strip()
                        if (
                            ("." in prev_line_clean or "/" in prev_line_clean)
                            and " " not in prev_line_clean
                            and len(prev_line_clean) < 100
                        ):
                            found_file = prev_line_clean
                            break
                current_file = found_file
            i += 1
            continue

        if in_block:
            if not block_lines:
                m = re.match(r"^(?://|#|/\*|<!--)\s*([a-zA-Z0-9_\-\./]+)\s*(?:\*/|-->)?$", line)
                if m and not current_file:
                    current_file = m.group(1).strip()
                    i += 1
                    continue
            block_lines.append(lines[i])
        i += 1

    if not files:
        files["output.txt"] = text
    return files
