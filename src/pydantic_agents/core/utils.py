import asyncio
import contextlib
import os
import re
import signal
import time
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
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


MAX_ROTATED_ARCHIVES = 5


def rotate_log(path: str, max_bytes: int, keep: int = MAX_ROTATED_ARCHIVES) -> bool:
    """Rotate ``path`` to ``<path>.<unix time>.archive`` once it exceeds ``max_bytes``, and keep only the ``keep``
    newest archives. Rotation alone filled the disk: nothing ever deleted an archive. True when it rotated."""
    try:
        if os.path.getsize(path) <= max_bytes:
            return False
        os.rename(path, f"{path}.{int(time.time())}.archive")
    except OSError:
        return False
    directory, name = os.path.split(os.path.abspath(path))
    try:
        archives = sorted(f for f in os.listdir(directory) if f.startswith(f"{name}.") and f.endswith(".archive"))
        for stale in archives[:-keep] if keep > 0 else archives:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(directory, stale))
    except OSError:
        pass
    return True


async def kill_and_reap(proc: Any) -> None:
    """Kill an untrusted build/test subprocess together with its children, then wait for it.

    The command is started in its own session (``start_new_session=True``), so the group can be killed: ``go test``
    runs the test binary as a child, and killing only ``go`` left that binary spinning after a timeout. Waiting
    reaps the process instead of leaving a zombie.

    Guard ``isinstance(pid, int) and pid > 1`` before calling ``os.killpg``: in unit tests ``proc`` is often a
    ``MagicMock`` whose ``proc.pid`` is a child ``MagicMock`` that CPython coerces via ``MagicMock.__index__()`` to
    ``1``, which on Linux (`os.killpg(1, SIGKILL)`) sends ``SIGKILL`` to process group 1 and kills the test runner!
    """
    pid = getattr(proc, "pid", None)
    try:
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1:
            os.killpg(pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=5)


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


def _append_artifact_files(parts: list[str], art: Any, total_len: int, max_chars: int) -> tuple[int, int, int]:
    """Append an artifact's files to ``parts`` until the character budget runs out.

    Returns the new running length, how many of this artifact's files never made it in, and how many characters
    were cut — both from a file sliced mid-content and from the files never appended. The caller reports real
    loss instead of inferring it from the budget being reached, which is not the same thing: a batch can land
    exactly on the budget having lost nothing, and a single oversized artifact loses a lot while omitting no
    whole file at all.
    """
    files = list(art.files.items())
    omitted_chars = 0
    for index, (path, content) in enumerate(files):
        snippet = f"--- {path} ---\n{content}"
        if total_len + len(snippet) > max_chars:
            # max(0, ...): an [Error] entry is appended without a budget check, so the budget can already be
            # overspent here. A negative bound would silently slice to "" and leave a bare marker.
            kept = snippet[: max(0, max_chars - total_len)]
            omitted_chars += len(snippet) - len(kept)
            snippet = (kept + "\n... (truncated)") if kept else "... (omitted, no budget left)"
        parts.append(snippet)
        total_len += len(snippet)
        if total_len >= max_chars:
            remaining = files[index + 1 :]
            omitted_chars += sum(len(f"--- {p} ---\n{c}") for p, c in remaining)
            return total_len, len(remaining), omitted_chars
    return total_len, 0, omitted_chars


@dataclass(frozen=True)
class FormattedArtifacts:
    """``format_artifacts`` output together with what it had to leave out.

    A quality gate judging this text needs to know the view is partial: "no findings" on a truncated batch is not
    a pass, it is an absence of evidence, and treating the two alike is a silent false approval. (AP-105)
    """

    text: str
    truncated: bool = False
    omitted_artifacts: int = 0
    omitted_files: int = 0
    omitted_chars: int = 0


def _truncation_notice(total: int, omitted_artifacts: int, omitted_files: int, omitted_chars: int, budget: int) -> str:
    """One deterministic line stating what is missing. It must not suggest the review went on without it.

    The character count carries the weight here: one artifact larger than the budget omits no whole artifact and
    no whole file, so counts of 0 and 0 alone would read like a spurious alarm over ~20 KB of unseen code.
    """
    return (
        f"... (TRUNCATED at the {budget}-character budget: {omitted_artifacts} of {total} artifacts and "
        f"{omitted_files} further file(s) omitted, {omitted_chars} characters of code not shown. "
        f"This is an INCOMPLETE view of the code.)"
    )


def format_artifacts_report(
    artifacts: list[Any], max_chars: int = 80_000, include_agent_header: bool = True
) -> FormattedArtifacts:
    """``format_artifacts``, plus a count of everything the character budget forced out.

    Args:
        artifacts: List of Artifact objects (each has .agent_name, .files, .metadata).
        max_chars: Maximum total characters (truncates if exceeded).
        include_agent_header: Whether to prefix each artifact with agent/story info.

    Returns:
        The formatted text and how many artifacts and files it does not contain.
    """
    parts: list[str] = []
    total_len = 0
    omitted_files = 0
    omitted_chars = 0
    reached = 0
    for art in artifacts:
        if isinstance(art, Exception):
            snippet = f"[Error] {art}"
            parts.append(snippet)
            total_len += len(snippet)
            reached += 1
            continue

        if include_agent_header:
            header = _artifact_header(art)
            if total_len + len(header) > max_chars:
                break
            parts.append(header)
            total_len += len(header)

        total_len, dropped, cut = _append_artifact_files(parts, art, total_len, max_chars)
        omitted_files += dropped
        omitted_chars += cut
        reached += 1
        if total_len >= max_chars:
            break

    omitted = [a for a in artifacts[reached:] if not isinstance(a, Exception)]
    omitted_artifacts = len(artifacts) - reached
    omitted_chars += sum(len(f"--- {p} ---\n{c}") for a in omitted for p, c in a.files.items())
    # Truncation means content was actually lost — not merely that the budget was reached. A batch that lands
    # exactly on the budget having dropped nothing is complete, and must not be reported as a partial view or
    # every such run would be failed closed for no reason.
    truncated = bool(omitted_artifacts or omitted_files or omitted_chars)
    if truncated:
        parts.append(
            _truncation_notice(len(artifacts), omitted_artifacts, omitted_files, omitted_chars, max_chars)
        )
    return FormattedArtifacts(
        text="\n\n".join(parts),
        truncated=truncated,
        omitted_artifacts=omitted_artifacts,
        omitted_files=omitted_files,
        omitted_chars=omitted_chars,
    )


def format_artifacts(artifacts: list[Any], max_chars: int = 80_000, include_agent_header: bool = True) -> str:
    """Format a list of Artifacts into a single prompt-friendly string.

    Callers that act on a verdict over this text want ``format_artifacts_report`` instead: it reports whether the
    text is complete.
    """
    return format_artifacts_report(artifacts, max_chars, include_agent_header).text


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
