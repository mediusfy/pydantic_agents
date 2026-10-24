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


async def _poll_for_file_approval(filepath: Path, poll_interval: float = 10.0, timeout: float = 3600.0) -> bool:
    """Wait for a `.cto_approved` marker file to exist.

    The file can be created by an external actor (e.g., CTO agent approval via
    `touch .cto_approved` or a UI button that writes the file). Returns True
    once detected, or False on timeout.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if filepath.exists():
            content = filepath.read_text().strip().lower()
            approved = content in ("true", "1", "yes", "")
            if not approved:
                logger.info("CTO approval marker file exists but content says no: %r", content)
            else:
                logger.info("CTO approval marker file detected — proceeding.")
                return True
        await asyncio.sleep(poll_interval)
    logger.warning("CTO approval timed out after %.0fs.", timeout)
    return False


async def apply(state: PipelineState, project: str = "default", timeout: float = 3600.0) -> bool:
    """CTO Approval Gate.

    Returns True if implementation should proceed, False if blocked.
    """
    if not state.cto_approved:
        logger.warning("=" * 60)
        logger.warning("CTO APPROVAL GATE — Implementation blocked")
        logger.warning("=" * 60)
        blockers_str = state.advisory_results.get("cto_blockers", "")
        if blockers_str:
            logger.warning("CTO-identified bottlenecks:")
            for b in blockers_str.split("\n"):
                if b.strip():
                    logger.warning("  • %s", b.strip())

        if os.environ.get("ORCHESTRATION_CTO_OVERRIDE"):
            logger.warning("ORCHESTRATION_CTO_OVERRIDE set — bypassing gate.")
            return True

        # Check for marker file override
        filepath = _approval_file_path(project)
        if filepath.exists():
            logger.info("Approval marker file found — proceeding.")
            return True

        # Wait for external approval
        logger.info(
            "Pipeline paused. Approve by:\n"
            "  1. Set env ORCHESTRATION_CTO_OVERRIDE=true and rerun\n"
            "  2. Create file: %s\n"
            "  3. Wait for CTO re-evaluation (polling for %ds)",
            filepath,
            timeout,
        )
        approved = await _poll_for_file_approval(filepath, timeout=timeout)
        if approved:
            return True

        logger.error("CTO approval not received within timeout. Pipeline will stop.")
        return False

    logger.info("CTO approval gate passed (approved_to_proceed=True).")
    return True
