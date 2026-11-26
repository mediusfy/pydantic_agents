"""CTO Approval Gate — blocks implementation until CTO agent approves.

Usage in pipeline (main.py):
    from pydantic_agents.core.gates import CTOApprovalGate
    await CTOApprovalGate.apply(state)
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import tempfile
from contextvars import ContextVar
from pathlib import Path

from pydantic_agents.core.models.core import PipelineState

logger = logging.getLogger(__name__)

_APPROVAL_FILE = ".cto_approved"
# ORCHESTRATION_CTO_OVERRIDE=false (or 0, or empty) must not bypass the gate.
_TRUTHY = frozenset({"1", "true", "yes", "on"})
STATUS_WAITING_CTO = "waiting_cto_approval"
_APPROVAL_FILE_PATH: Path | None = None
_RUN_ID_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

# Per-task context carrying (expected_run_id, run_scoped_marker_path) into _poll_for_file_approval while keeping its
# positional signature unchanged for callers and tests that monkeypatch it.
_gate_poll_ctx: ContextVar[tuple[str, Path | None]] = ContextVar("cto_gate_poll_ctx", default=("", None))


def _marker_filename(run_id: str = "") -> str:
    """Marker filename: ``.cto_approved.<run_id>`` when bound to a run, else ``.cto_approved``."""
    if not run_id:
        return _APPROVAL_FILE
    if not (_RUN_ID_COMPONENT.fullmatch(run_id) and ".." not in run_id):
        from pydantic_agents.core.exceptions import PathTraversalError

        raise PathTraversalError(f"Invalid run_id {run_id!r} for approval marker")
    return f"{_APPROVAL_FILE}.{run_id}"


def _project_approval_path(project: str, run_id: str = "") -> Path:
    """The single place the marker's location is derived.

    ``project`` (and optional ``run_id``) arrive as MCP arguments, so they go through ``safe_path_join``: a
    ``project`` of ``../../tmp/x`` must neither read nor write a marker outside ``output_dir``. Both the read and
    the write path call this one helper, because writing the same path expression twice is what let the read path
    lose the guard the write path had (AP-110). Raises ``PathTraversalError`` on an escape.

    The imports stay function-local: ``settings`` and ``utils`` import back into this package.
    """
    from pydantic_agents.core.settings import settings
    from pydantic_agents.core.utils import safe_path_join

    return safe_path_join(settings.output_dir, project, _marker_filename(run_id))


def _approval_file_path(project: str, run_id: str = "") -> Path:
    """Return path for approval marker file inside the project output dir.

    The directory is created only after the path is checked, so a rejected traversal leaves nothing behind.
    """
    path = _project_approval_path(project, run_id=run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_text(path: Path, payload: str) -> None:
    """Write ``payload`` to ``path`` via sibling tempfile + ``os.replace`` so readers never observe a truncated file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{_APPROVAL_FILE}.", suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def write_approval(project: str, approved: bool = True, *, run_id: str = "", adr_id: str = "") -> Path:
    """Atomically write the structured approval marker for a project (optionally scoped to ``run_id``)."""
    path = _project_approval_path(project, run_id=run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"approved": bool(approved), "run_id": run_id, "adr_id": adr_id}) + "\n"
    _atomic_write_text(path, payload)
    return path


def consume_marker(filepath: Path, *, expected_run_id: str = "") -> bool | None:
    """``read_marker``, then delete the marker when it held a decision for ``expected_run_id``.

    A marker bound to another ``run_id`` returns ``None`` and is left untouched so concurrent runs on the same
    project cannot steal each other's approvals.
    """
    decision = read_marker(filepath, expected_run_id=expected_run_id)
    if decision is not None:
        with contextlib.suppress(OSError):
            filepath.unlink()
    return decision


def _log_unrecognised_marker(length: int) -> None:
    # The file is outside our control, so its content is not logged (it could forge or flood log lines).
    logger.info("CTO approval marker has unrecognised content (%d characters); still waiting.", length)


def _read_json_marker(raw: str, expected_run_id: str) -> bool | None:
    try:
        data = json.loads(raw)
    except ValueError:
        _log_unrecognised_marker(len(raw))
        return None
    if not isinstance(data, dict):
        _log_unrecognised_marker(len(raw))
        return None
    marker_run_id = data.get("run_id", "")
    if not isinstance(marker_run_id, str):
        _log_unrecognised_marker(len(raw))
        return None
    if marker_run_id and expected_run_id and marker_run_id != expected_run_id:
        return None
    approved = data.get("approved")
    if isinstance(approved, bool):
        return approved
    _log_unrecognised_marker(len(raw))
    return None


def read_marker(filepath: Path, *, expected_run_id: str = "") -> bool | None:
    """The decision in an approval marker: True (approved), False (explicitly rejected), or None.

    Empty files return ``None`` (never ``True``), preventing an in-progress shell truncation (`> .cto_approved`) from
    turning an intended rejection into an approval.
    """
    if not filepath.exists():
        return None
    try:
        raw = filepath.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    if raw.startswith(("{", "[")):
        return _read_json_marker(raw, expected_run_id)
    lowered = raw.lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    _log_unrecognised_marker(len(raw))
    return None


def _consume_candidates(candidates: tuple[Path, ...], expected_run_id: str) -> bool | None:
    for candidate in candidates:
        decision = consume_marker(candidate, expected_run_id=expected_run_id)
        if decision is not None:
            return decision
    return None


async def _poll_for_file_approval(filepath: Path, poll_interval: float = 10.0, timeout_seconds: float = 3600.0) -> bool:
    """Wait for a decision in the run-scoped or project `.cto_approved` marker file."""
    expected_run_id, run_scoped_path = _gate_poll_ctx.get()
    candidates: tuple[Path, ...] = (
        (run_scoped_path, filepath) if run_scoped_path and run_scoped_path != filepath else (filepath,)
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        decision = _consume_candidates(candidates, expected_run_id)
        if decision is True:
            logger.info("CTO approval marker file detected — proceeding.")
            return True
        if decision is False:
            logger.warning("CTO rejected the run via the marker file.")
            return False
        await asyncio.sleep(poll_interval)
    logger.warning("CTO approval timed out after %.0fs.", timeout_seconds)
    return False


def _log_gate_blocked(state: PipelineState) -> None:
    """Log that implementation is blocked and list the bottlenecks the CTO identified."""
    logger.warning("=" * 60)
    logger.warning("CTO APPROVAL GATE — Implementation blocked")
    logger.warning("=" * 60)
    blockers = [b.strip() for b in state.advisory_results.get("cto_blockers", "").split("\n") if b.strip()]
    if blockers:
        logger.warning("CTO-identified bottlenecks:")
        for blocker in blockers:
            logger.warning("  • %s", blocker)


async def apply(state: PipelineState, project: str = "default", timeout_seconds: float = 3600.0) -> bool:
    """CTO Approval Gate.

    Returns True if implementation should proceed, False if blocked.
    """
    if state.cto_approved:
        logger.info("CTO approval gate passed (approved_to_proceed=True).")
        return True

    _log_gate_blocked(state)

    if os.environ.get("ORCHESTRATION_CTO_OVERRIDE", "").strip().lower() in _TRUTHY:
        logger.warning("ORCHESTRATION_CTO_OVERRIDE set — bypassing gate.")
        return True

    from pydantic_agents.core.limits import run_id_var

    effective_run_id = state.run_id or run_id_var.get() or ""
    filepath = _approval_file_path(project)
    run_filepath = _approval_file_path(project, run_id=effective_run_id) if effective_run_id else None
    candidates: tuple[Path, ...] = (run_filepath, filepath) if run_filepath else (filepath,)

    decision = _consume_candidates(candidates, effective_run_id)
    if decision is True:
        logger.info("Approval marker file found — proceeding.")
        return True
    if decision is False:
        logger.error("CTO rejected the run via the marker file. Pipeline will stop.")
        return False

    target_display = run_filepath or filepath
    logger.info(
        "Pipeline paused. Approve by:\n"
        "  1. Set env ORCHESTRATION_CTO_OVERRIDE=true and rerun\n"
        "  2. Call approve_cto_gate(project_name=%r, run_id=%r, approved=True, adr_id=...) on /mcp/admin\n"
        "  3. Or write 'true' to %s (polling for %ds)",
        project,
        effective_run_id,
        target_display,
        timeout_seconds,
    )
    await _report_status(STATUS_WAITING_CTO)
    ctx_token = _gate_poll_ctx.set((effective_run_id, run_filepath))
    try:
        approved = await _poll_for_file_approval(filepath, timeout_seconds=timeout_seconds)
    finally:
        _gate_poll_ctx.reset(ctx_token)
    await _report_status("running")
    if not approved:
        logger.error("CTO approval not received within timeout. Pipeline will stop.")
    return approved


async def _report_status(status: str) -> None:
    """Show the wait in the run's status (a poll of check_pipeline_status says why nothing is happening)."""
    from pydantic_agents.core.limits import run_id_var

    run_id = run_id_var.get()
    if run_id is None:
        return
    try:
        from pydantic_agents.core.pipeline_jobs import set_run_status

        await set_run_status(run_id, status)
    except Exception as e:  # best-effort, like the run store itself
        logger.debug("Could not report status %s for run %s: %s", status, run_id, e)
