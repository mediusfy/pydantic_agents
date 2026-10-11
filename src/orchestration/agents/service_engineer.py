import os
import subprocess
from pathlib import Path

from pydantic_ai import Agent, RunContext

from orchestration.exceptions import PathTraversalError
from orchestration.models.core import Artifact
from orchestration.models.factory import create_model
from orchestration.settings import settings

_READ_FILE_MAX_BYTES = 256 * 1024


def _sandbox_root() -> Path:
    """Resolve the root that file-reading tools are allowed to touch.

    Read from ORCHESTRATION_CODEBASE_ROOT (set by run_existing_code_pipeline), with
    CWD as the safe local-dev default. Returning a Path lets the caller use
    relative_to() to enforce containment.
    """
    return Path(os.environ.get("ORCHESTRATION_CODEBASE_ROOT", os.getcwd())).resolve()


def _resolve_sandboxed(user_path: str) -> Path:
    root = _sandbox_root()
    candidate = (root / user_path).resolve() if not os.path.isabs(user_path) else Path(user_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise PathTraversalError(f"Path '{user_path}' escapes sandbox root {root}") from e
    return candidate


service_engineer_agent = Agent(
    create_model(settings.resolve_model(settings.backend_model)),
    system_prompt="""You are a Go Backend Service Engineer specializing in modifying EXISTING codebases. You are NOT a scaffolder.

Your goal is surgical, minimal changes to code that is already deployed and running.

## Mandatory Rules

1. **Understand before touching**: Read existing files first. Understand the full context, not just the change area. Identify existing patterns (handler layout, DI wiring, error handling, logging) — extend, don't rewrite.

2. **Library-first**: Use stdlib or already-imported dependencies. Never write a custom hashmap, retry loop, date parser, serializer, or validator.

3. **DRY**: If 3+ lines repeat, extract them. If a similar function exists nearby, reuse it. Do not create parallel patterns.

4. **Simple**: Flat > nested. Interfaces with exactly one implementation are not interfaces yet — inline them. No premature abstraction.

5. **YAGNI**: Do not add config knobs, feature flags, or generality nobody asked for. No "what if we someday need X" code paths.

6. **Handlers are thin**: Parse request → delegate to service → map response/error. No business logic in handlers.

7. **Error handling**: Use typed sentinel errors. Wrap with context via fmt.Errorf("%w: %v"). Use errors.Is for matching.

8. **Logging**: Use log/slog. Structured attributes only, no fmt.Print or log.Printf.

9. **Observability**: Add OpenTelemetry spans and attributes on every significant operation.

10. **Comments**: Explain "why", not "what". Remove "what" comments.

11. **Imports**: 3 groups (stdlib / third-party / local), alphabetized within each group.

## Output

Return an Artifact with ONLY the files that changed. Include the full file contents for each changed file. Do not include unchanged files.
""",
    output_type=Artifact,
    defer_model_check=True,
    retries=5,
)


@service_engineer_agent.tool
async def read_file(ctx: RunContext[str], filepath: str) -> str:
    """Read a file from the codebase (sandboxed to ORCHESTRATION_CODEBASE_ROOT) before modifying it."""
    try:
        path = _resolve_sandboxed(filepath)
    except PathTraversalError as e:
        return f"ERROR: {e}"
    if not path.exists() or not path.is_file():
        return f"ERROR: File not found: {filepath}"
    content = path.read_text(errors="replace")
    if len(content) > _READ_FILE_MAX_BYTES:
        return content[:_READ_FILE_MAX_BYTES] + f"\n... (truncated at {_READ_FILE_MAX_BYTES} bytes)"
    return content


@service_engineer_agent.tool
def list_directory(ctx: RunContext[str], dirpath: str) -> str:
    """List files in a directory (sandboxed to ORCHESTRATION_CODEBASE_ROOT) to understand project structure."""
    try:
        path = _resolve_sandboxed(dirpath)
    except PathTraversalError as e:
        return f"ERROR: {e}"
    if not path.exists() or not path.is_dir():
        return f"ERROR: Directory not found: {dirpath}"
    items = []
    for entry in sorted(path.iterdir()):
        suffix = "/" if entry.is_dir() else ""
        items.append(f"{entry.name}{suffix}")
    return "\n".join(items)


@service_engineer_agent.tool
def run_gofmt_check(ctx: RunContext[str], filepath: str) -> str:
    """Run gofmt on a file and return any formatting issues."""
    try:
        result = subprocess.run(["gofmt", "-d", filepath], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and not result.stdout:
            return "PASS: gofmt is clean"
        return result.stdout or result.stderr
    except FileNotFoundError:
        return "WARN: gofmt not found, skipping"
    except Exception as e:
        return f"ERROR: {e}"
