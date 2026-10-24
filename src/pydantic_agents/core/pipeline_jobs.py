"""Pipeline jobs — async background task management for long-running pipelines.

This module provides lightweight in-process background task execution so MCP
tools can submit a pipeline run, return a run_id immediately, and let the
caller poll for completion. No Redis/DB dependency — simple asyncio task + dict.

``PIPELINE_RUNS`` is the process-local view (it holds the task handle). Every status change is also written
to the ``pipeline_runs`` table (``core/run_store.py``, best-effort), so after a restart ``get_run`` still finds
a run and reports one that was in flight as ``interrupted``; ``resume_pipeline_run`` restarts it from its
checkpoints.
"""

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import logfire

from pydantic_agents.core import run_store
from pydantic_agents.core.limits import run_id_var
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.pipeline.factory import build_default_pipeline

logger = logging.getLogger(__name__)

# In-memory store of pipeline runs. Keyed by run_id.
# Each entry: { "status": str, "project": str, "result": str | None,
#               "error": str | None, "started_at": str, "completed_at": str | None,
#               "elapsed_seconds": float | None, "_task": asyncio.Task | None }
PIPELINE_RUNS: dict[str, dict[str, Any]] = {}


async def submit_pipeline_run(
    project_name: str, requirement: str, *, run_id: str | None = None, resume: bool = False
) -> str:
    """Submit a pipeline run as a background asyncio task.

    Creates a new asyncio task running ``OrgitrationPipeline`` (assembled via
    ``build_default_pipeline``), records its status in ``PIPELINE_RUNS``,
    and returns the run_id immediately.

    The caller should poll with ``PIPELINE_RUNS[run_id]["status"]`` to check
    completion.
    """
    run_id = run_id or f"pipe_{uuid.uuid4().hex[:12]}"

    PIPELINE_RUNS[run_id] = {
        "status": "pending",
        "project": project_name,
        "requirement": requirement,
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


async def _run_pipeline_in_background(run_id: str, project_name: str, requirement: str, resume: bool = False) -> None:
    """Run the ``OrgitrationPipeline`` in a background task, updating ``PIPELINE_RUNS``.

    Constructs a fully-staged pipeline via ``build_default_pipeline`` — the
    same factory used by the CLI in ``main.py`` — so both code paths share
    one definition of "the standard four-stage pipeline".
    """
    start_time = time.time()

    with logfire.span("pipeline.background_run", run_id=run_id, project=project_name):
        PIPELINE_RUNS[run_id]["status"] = "running"
        await run_store.save_run(run_id, PIPELINE_RUNS[run_id])
        run_id_var.set(run_id)  # this task's own context: sent to LiteLLM as the Langfuse session id
        try:
            config = PipelineConfig(
                project=project_name,
                step="all",
                blueprint="new_feature",
                resume=resume,
                auto_yes=True,
                requirement=requirement,
            )
            pipeline = build_default_pipeline(config)
            await pipeline.execute()
        except asyncio.CancelledError:
            PIPELINE_RUNS[run_id]["status"] = "cancelled"
            PIPELINE_RUNS[run_id]["error"] = "Task was cancelled"
            # Re-raise so the asyncio task is marked cancelled, not completed.
            raise
        except Exception as e:
            logger.exception("Pipeline run %s failed: %s", run_id, e)
            PIPELINE_RUNS[run_id]["status"] = "failed"
            PIPELINE_RUNS[run_id]["error"] = f"{type(e).__name__}: {e}"
        else:
            PIPELINE_RUNS[run_id]["status"] = "completed"
            PIPELINE_RUNS[run_id]["result"] = (
                f"Pipeline completed for project '{project_name}'. Artifacts written to generated-app/{project_name}/."
            )
        finally:
            elapsed = time.time() - start_time
            PIPELINE_RUNS[run_id]["completed_at"] = datetime.now(UTC).isoformat()
            PIPELINE_RUNS[run_id]["elapsed_seconds"] = elapsed
            # Shielded: a cancelled task must still record that it was cancelled.
            await asyncio.shield(run_store.save_run(run_id, PIPELINE_RUNS[run_id]))


async def get_run(run_id: str) -> dict[str, Any] | None:
    """The run from memory, else from the database. A stored run that is still pending/running but has no live
    task in this process was cut off by a restart, so it is reported as ``interrupted``."""
    run = PIPELINE_RUNS.get(run_id)
    if run is not None:
        return run
    stored = await run_store.load_run(run_id)
    if stored is not None and stored["status"] in ("pending", "running"):
        stored["status"] = "interrupted"
        stored["error"] = "The server restarted while this run was in flight. Use resume_pipeline_run to continue it."
    return stored


async def resume_pipeline_run(run_id: str) -> str | None:
    """Restart an interrupted or failed run under the same id, resuming from its checkpoints.

    Returns the run id, or None when the run is unknown, still running, or already completed.
    """
    run = await get_run(run_id)
    if run is None or run["status"] not in ("interrupted", "failed", "cancelled"):
        return None
    return await submit_pipeline_run(run["project"], run.get("requirement") or "", run_id=run_id, resume=True)
