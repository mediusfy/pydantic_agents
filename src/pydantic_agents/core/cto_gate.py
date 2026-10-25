"""CTO Approval Gate — blocks implementation until CTO agent approves.

Usage in pipeline (main.py):
    from pydantic_agents.core.gates import CTOApprovalGate
    await CTOApprovalGate.apply(state)
"""

import asyncio
import logging
import os
from pathlib import Path

from pydantic_agents.core.models.core import PipelineState

logger = logging.getLogger(__name__)

_APPROVAL_FILE = ".cto_approved"
_APPROVAL_FILE_PATH: Path | None = None


def _approval_file_path(project: str) -> Path:
    """Return path for approval marker file inside the project output dir."""
    from pydantic_agents.core.settings import settings

    p = Path(str(settings.output_dir)) / project / _APPROVAL_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_approval(project: str, approved: bool = True) -> Path:
    """Write the approval marker for a project (what an MCP caller outside the pod cannot do with ``touch``).

    The project name is caller-chosen, so the path is checked to stay inside the output dir.
    """
    from pydantic_agents.core.settings import settings
    from pydantic_agents.core.utils import safe_path_join

    path = safe_path_join(settings.output_dir, project, _APPROVAL_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("true\n" if approved else "false\n")
    return path


def read_marker(filepath: Path) -> bool | None:
    """The decision in an approval marker: True (approved), False (explicitly rejected), None (no marker, or
    content that is neither, which keeps the gate waiting). An empty file counts as approved, as `touch` creates it."""
    if not filepath.exists():
        return None
    content = filepath.read_text().strip().lower()
    if content in ("true", "1", "yes", ""):
        return True
    if content in ("false", "0", "no"):
        return False
    # The file is outside our control: show a short printable prefix so it cannot forge or flood log lines.
    shown = "".join(ch if ch.isprintable() else "?" for ch in content[:40])
    logger.info("CTO approval marker has unrecognised content %r; still waiting.", shown)
    return None


async def _poll_for_file_approval(filepath: Path, poll_interval: float = 10.0, timeout_seconds: float = 3600.0) -> bool:
    """Wait for a decision in the `.cto_approved` marker file.

    The file can be written by `approve_cto_gate` (MCP), by `touch .cto_approved`, or by a UI. Returns True on
    approval, False on an explicit rejection or on timeout.
    """
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        decision = read_marker(filepath)
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

    if os.environ.get("ORCHESTRATION_CTO_OVERRIDE"):
        logger.warning("ORCHESTRATION_CTO_OVERRIDE set — bypassing gate.")
        return True

    # Check for a marker file left by an earlier decision (a rejection stops the run, it is not an approval)
    filepath = _approval_file_path(project)
    decision = read_marker(filepath)
    if decision is True:
        logger.info("Approval marker file found — proceeding.")
        return True
    if decision is False:
        logger.error("CTO rejected the run via the marker file. Pipeline will stop.")
        return False

    # Wait for external approval
    logger.info(
        "Pipeline paused. Approve by:\n"
        "  1. Set env ORCHESTRATION_CTO_OVERRIDE=true and rerun\n"
        "  2. Create file: %s\n"
        "  3. Wait for CTO re-evaluation (polling for %ds)",
        filepath,
        timeout_seconds,
    )
    approved = await _poll_for_file_approval(filepath, timeout_seconds=timeout_seconds)
    if not approved:
        logger.error("CTO approval not received within timeout. Pipeline will stop.")
    return approved
