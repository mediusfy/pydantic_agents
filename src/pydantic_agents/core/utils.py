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
    if not str(full).startswith(str(base_resolved)):
        raise PathTraversalError(f"Path traversal detected: {'/'.join(parts)} escapes {base}")
    return full


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


def format_artifacts(artifacts: list[Any], max_chars: int = 80_000, include_agent_header: bool = True) -> str:
    """Format a list of Artifacts into a single prompt-friendly string.

    Args:
        artifacts: List of Artifact objects (each has .agent_name, .files, .metadata).
        max_chars: Maximum total characters (truncates if exceeded).
        include_agent_header: Whether to prefix each artifact with agent/story info.

    Returns:
        Formatted string suitable for downstream agent prompts.
    """
    parts = []
    total_len = 0
    for art in artifacts:
        if isinstance(art, Exception):
            snippet = f"[Error] {art}"
            parts.append(snippet)
            total_len += len(snippet)
            continue

        if include_agent_header:
            story_id = art.metadata.get("user_story_id", "N/A")
            header = f"=== Agent: {art.agent_name} | Story: {story_id} ==="
            if total_len + len(header) > max_chars:
                parts.append("... (truncated)")
                break
            parts.append(header)
            total_len += len(header)

        for path, content in art.files.items():
            snippet = f"--- {path} ---\n{content}"
            if total_len + len(snippet) > max_chars:
                remaining = max_chars - total_len
                snippet = snippet[:remaining] + "\n... (truncated)"
            parts.append(snippet)
            total_len += len(snippet)
            if total_len >= max_chars:
                break
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
