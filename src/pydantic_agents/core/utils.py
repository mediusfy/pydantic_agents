import os
import re
from collections.abc import Iterator
from contextvars import ContextVar
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

# Environment for untrusted build/test subprocesses (go test, go build, vitest, eslint) running model-generated code.
#
# This allowlist is a trust boundary, not just credential hygiene: anything forwarded here is visible to, and steers,
# code we did not write. So:
# - credentials (LITELLM_*, OPENAI_API_KEY, DATABASE_URL, cloud tokens, ...) are never forwarded;
# - variables that point the Go toolchain at private infrastructure are never forwarded (GOPRIVATE, GONOPROXY,
#   GONOSUMDB, GOFLAGS, GOINSECURE, GOAUTH), GOENV is forced off so the operator's `go env -w` file (which can set
#   all of those) is ignored, and GOPROXY/GOSUMDB are pinned from settings (`UNTRUSTED_GOPROXY`, default the public
#   proxy without `direct`, so a generated go.mod cannot make `go mod tidy` fetch from an internal VCS host or proxy);
# - NODE_PATH and XDG_CONFIG_HOME are not forwarded (module resolution / tool config from the operator's machine);
# - HOME and the cache locations stay, because go and npm need a writable cache; they hold build caches, not secrets.
# Add a variable only after asking what generated code could do with it.
_SAFE_SUBPROCESS_ENV_KEYS: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "TERM",
        "SYSTEMROOT",
        "GOPATH",
        "GOROOT",
        "GOCACHE",
        "GOMODCACHE",
        "CGO_ENABLED",
        "GOOS",
        "GOARCH",
        "NPM_CONFIG_CACHE",
        "npm_config_cache",
        "XDG_CACHE_HOME",
    }
)


def sanitized_subprocess_env() -> dict[str, str]:
    """A minimal environment for untrusted build/test subprocesses (see ``_SAFE_SUBPROCESS_ENV_KEYS``)."""
    from pydantic_agents.core.settings import settings

    env = {k: v for k, v in os.environ.items() if k in _SAFE_SUBPROCESS_ENV_KEYS}
    env["GOENV"] = "off"
    env["GOPROXY"] = settings.untrusted_goproxy
    env["GOSUMDB"] = settings.untrusted_gosumdb
    return env


# The codebase root of the pipeline running in the current task. A ContextVar (not an environment variable) so two
# concurrent pipelines on different codebases never see each other's root; tasks and threads started from the
# pipeline inherit it. ``run_existing_code_pipeline`` sets it for the duration of a run.
sandbox_root_var: ContextVar[Path | None] = ContextVar("sandbox_root", default=None)


def sandbox_root() -> Path:
    """Root that model-driven file tools may touch.

    The current pipeline's root (``sandbox_root_var``), else ORCHESTRATION_CODEBASE_ROOT (for a server started with a
    fixed codebase), else CWD as the safe local-dev default.
    """
    root = sandbox_root_var.get()
    if root is not None:
        return root.resolve()
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


_INLINE_PATH_COMMENT = re.compile(r"^(?://|#|/\*|<!--)\s*([a-zA-Z0-9_\-\./]+)\s*(?:\*/|-->)?$")


def _filename_above(lines: list[str], fence_index: int) -> str | None:
    """A file name written on one of the three lines above a code fence (the nearest wins), if any."""
    for offset in range(1, 4):
        if fence_index - offset < 0:
            continue
        candidate = lines[fence_index - offset].strip().strip("*`#:").strip()
        if ("." in candidate or "/" in candidate) and " " not in candidate and len(candidate) < 100:
            return candidate
    return None


def _fenced_blocks(lines: list[str]) -> Iterator[tuple[int, list[str]]]:
    """The (index of the opening fence, body lines) of every closed fenced block; an unclosed one is dropped."""
    start: int | None = None
    body: list[str] = []
    for i, raw in enumerate(lines):
        if raw.strip().startswith("```"):
            if start is None:
                start, body = i, []
            else:
                yield start, body
                start = None
        elif start is not None:
            body.append(raw)


def parse_code_blocks_to_files(text: str) -> dict[str, str]:
    """Parse markdown code blocks and extract filenames from headers or inline path comments."""
    lines = text.split("\n")
    files: dict[str, str] = {}
    for start, body in _fenced_blocks(lines):
        name = _filename_above(lines, start)
        if name is None and body and (match := _INLINE_PATH_COMMENT.match(body[0].strip())):
            name = match.group(1).strip()  # the comment names the file and is not part of its content
            body = body[1:]
        if name:
            files[name] = "\n".join(body)
    return files or {"output.txt": text}
