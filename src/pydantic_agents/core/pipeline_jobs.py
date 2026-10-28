"""Pipeline jobs — async background task management for long-running pipelines.

This module provides lightweight in-process background task execution so MCP
tools can submit a pipeline run, return a run_id immediately, and let the
caller poll for completion. No Redis/DB dependency — simple asyncio task + dict.

``PIPELINE_RUNS`` is the process-local view (it holds the task handle). Every status change is also written
to the ``pipeline_runs`` table (``core/run_store.py``, best-effort), so after a restart ``get_run`` still finds
a run and reports one that was in flight as ``interrupted``; ``resume_pipeline_run`` restarts it from its
checkpoints.

A run's final status comes from the pipeline's final state, not from "execute() returned": ``completed`` only
when the gates approved, ``failed`` with the failure report otherwise (no stories, CTO rejection, retries
exhausted, or an exception).
"""

import asyncio
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import logfire

from pydantic_agents.core import run_store
from pydantic_agents.core.limits import run_id_var
from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import BLUEPRINTS, PipelineConfig
from pydantic_agents.core.pipeline.factory import build_pipeline

logger = logging.getLogger(__name__)

# In-memory store of pipeline runs. Keyed by run_id.
# Each entry: { "status": str, "project": str, "result": str | None,
#               "error": str | None, "started_at": str, "completed_at": str | None,
#               "elapsed_seconds": float | None, "_task": asyncio.Task | None }
PIPELINE_RUNS: dict[str, dict[str, Any]] = {}

_RUN_ID = re.compile(r"pipe_[0-9a-f]{12}")

STATUS_WAITING_CTO = "waiting_cto_approval"


async def set_run_status(run_id: str, status: str, **fields: Any) -> None:
    """Update a live run's status (and any other fields) in memory and in the store; unknown runs are ignored."""
    run = PIPELINE_RUNS.get(run_id)
    if run is None:
        return
    run["status"] = status
    run.update(fields)
    await run_store.save_run(run_id, run)


async def submit_pipeline_run(
    project_name: str,
    requirement: str,
    *,
    run_id: str | None = None,
    resume: bool = False,
    blueprint: str = "new_feature",
    framework: str | None = None,
) -> str:
    """Submit a pipeline run as a background asyncio task.

    Creates a new asyncio task running the pipeline the blueprint names (assembled via ``build_pipeline``),
    records its status in ``PIPELINE_RUNS``, and returns the run_id immediately. The caller polls ``get_run``.
    """
    if run_id is None:
        run_id = f"pipe_{uuid.uuid4().hex[:12]}"
    elif not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"Invalid run id {run_id!r}: expected pipe_ followed by 12 hex characters")
    if blueprint not in BLUEPRINTS:
        raise ValueError(f"Unknown blueprint {blueprint!r}. Available: {', '.join(BLUEPRINTS)}")
    live = PIPELINE_RUNS.get(run_id, {}).get("_task")
    if live is not None and not live.done():
        raise RuntimeError(f"Run {run_id} still has a live task; cancel it before resubmitting")

    PIPELINE_RUNS[run_id] = {
        "status": "pending",
        "project": project_name,
        "requirement": requirement,
        "blueprint": blueprint,
        "framework": framework or BLUEPRINTS[blueprint].get("framework", "legacy"),
        "result": None,
        "error": None,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "elapsed_seconds": None,
    }

    # Fire-and-forget background task; stash the handle so cancel_pipeline_run
    # can reach it.
    await run_store.save_run(run_id, PIPELINE_RUNS[run_id])
    task = asyncio.create_task(_run_pipeline_in_background(run_id, project_name, requirement, resume))
    PIPELINE_RUNS[run_id]["_task"] = task

    return run_id


def _outcome(project_name: str, state: Any) -> tuple[str, str | None, str | None]:
    """(status, result, error) from a pipeline's final state. ``None`` (a pipeline that returns nothing) counts as
    completed for backwards compatibility with wrappers that do not report a state."""
    if not isinstance(state, PipelineState):
        return "completed", f"Pipeline completed for project '{project_name}'.", None
    if state.approved:
        return (
            "completed",
            f"Pipeline completed for project '{project_name}'. Artifacts written to generated-app/{project_name}/.",
            None,
        )
    if state.failure_report is not None:
        return "failed", None, f"{state.failure_report.source}: {state.failure_report.trace}"
    return "failed", None, "The pipeline finished without approval (gate retries exhausted)."


async def _run_pipeline_in_background(run_id: str, project_name: str, requirement: str, resume: bool = False) -> None:
    """Run the pipeline in a background task, updating ``PIPELINE_RUNS`` and the run store."""
    start_time = time.time()
    run = PIPELINE_RUNS[run_id]

    with logfire.span("pipeline.background_run", run_id=run_id, project=project_name):
        run["status"] = "running"
        await run_store.save_run(run_id, run)
        run_id_var.set(run_id)  # this task's own context: sent to LiteLLM as the Langfuse session id
        try:
            config = PipelineConfig(
                project=project_name,
                step="all",
                blueprint=run.get("blueprint", "new_feature"),
                framework=run.get("framework") or "legacy",
                resume=resume,
                auto_yes=True,
                requirement=requirement,
                run_id=run_id,
            )
            pipeline = build_pipeline(config)
            final_state = await pipeline.execute()
        except asyncio.CancelledError:
            run["status"] = "cancelled"
            run["error"] = "Task was cancelled"
            # Re-raise so the asyncio task is marked cancelled, not completed.
            raise
        except Exception as e:
            logger.exception("Pipeline run %s failed: %s", run_id, e)
            run["status"] = "failed"
            run["error"] = f"{type(e).__name__}: {e}"
        else:
            run["status"], run["result"], run["error"] = _outcome(project_name, final_state)
        finally:
            elapsed = time.time() - start_time
            run["completed_at"] = datetime.now(UTC).isoformat()
            run["elapsed_seconds"] = elapsed
            # Shielded: a cancelled task must still record that it was cancelled.
            await asyncio.shield(run_store.save_run(run_id, run))


async def get_run(run_id: str) -> dict[str, Any] | None:
    """The run from memory, else from the database. A stored run that is still pending/running but has no live
    task in this process was cut off by a restart, so it is reported as ``interrupted``. That status exists only in
    the returned dict; the stored row keeps its last written status until the run is resumed or finishes."""
    run = PIPELINE_RUNS.get(run_id)
    if run is not None:
        return run
    stored = await run_store.load_run(run_id)
    if stored is not None and stored["status"] in ("pending", "running", STATUS_WAITING_CTO):
        stored["status"] = "interrupted"
        stored["error"] = "The server restarted while this run was in flight. Use resume_pipeline_run to continue it."
    return stored


def list_interrupted_runs() -> dict[str, dict[str, Any]]:
    """In-process runs left ``pending``/``running``/waiting-on-CTO with no live task — the same "cut off by a
    restart" condition ``get_run`` reports for a single id, scanned across every run this process knows about.
    Used by ``sre_autopilot`` to find runs worth auto-resuming; does not consult the database, so a run known only
    to another process (or from before this process's last restart) is not included."""
    return {
        run_id: run
        for run_id, run in PIPELINE_RUNS.items()
        if run.get("status") in ("pending", "running", STATUS_WAITING_CTO)
        and (run.get("_task") is None or run["_task"].done())
    }


async def resume_pipeline_run(run_id: str) -> str | None:
    """Restart an interrupted or failed run under the same id, resuming from its checkpoints.

    Returns the run id, or None when the run is unknown, still running, or already completed.
    """
    live = PIPELINE_RUNS.get(run_id, {}).get("_task")
    if live is not None and not live.done():
        return None  # a task is still alive (e.g. a cancel not yet reaped): keep its handle reachable
    run = await get_run(run_id)
    if run is None or run["status"] not in ("interrupted", "failed", "cancelled"):
        return None
    return await submit_pipeline_run(
        run["project"],
        run.get("requirement") or "",
        run_id=run_id,
        resume=True,
        blueprint=run.get("blueprint") or "new_feature",
        framework=run.get("framework"),
    )
