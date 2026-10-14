"""Pipeline jobs — async background task management for long-running pipelines.

This module provides lightweight in-process background task execution so MCP
tools can submit a pipeline run, return a run_id immediately, and let the
caller poll for completion. No Redis/DB dependency — simple asyncio task + dict.

.. caution::

   ``PIPELINE_RUNS`` is process-local and lost on restart.  For durable job
   tracking, back this with the existing pgvector PostgreSQL connection
   (table: ``pipeline_runs``).  Not implemented — open ticket.
"""

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import logfire

from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.pipeline.factory import build_default_pipeline

logger = logging.getLogger(__name__)

# In-memory store of pipeline runs. Keyed by run_id.
# Each entry: { "status": str, "project": str, "result": str | None,
#               "error": str | None, "started_at": str, "completed_at": str | None,
#               "elapsed_seconds": float | None, "_task": asyncio.Task | None }
PIPELINE_RUNS: dict[str, dict[str, Any]] = {}


async def submit_pipeline_run(project_name: str, requirement: str) -> str:
    """Submit a pipeline run as a background asyncio task.

    Creates a new asyncio task running ``OrgitrationPipeline`` (assembled via
    ``build_default_pipeline``), records its status in ``PIPELINE_RUNS``,
    and returns the run_id immediately.

    The caller should poll with ``PIPELINE_RUNS[run_id]["status"]`` to check
    completion.
    """
    run_id = f"pipe_{uuid.uuid4().hex[:12]}"

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
    task = asyncio.create_task(_run_pipeline_in_background(run_id, project_name, requirement))
    PIPELINE_RUNS[run_id]["_task"] = task

    return run_id


async def _run_pipeline_in_background(run_id: str, project_name: str, requirement: str) -> None:
    """Run the ``OrgitrationPipeline`` in a background task, updating ``PIPELINE_RUNS``.

    Constructs a fully-staged pipeline via ``build_default_pipeline`` — the
    same factory used by the CLI in ``main.py`` — so both code paths share
    one definition of "the standard four-stage pipeline".
    """
    start_time = time.time()

    with logfire.span("pipeline.background_run", run_id=run_id, project=project_name):
        PIPELINE_RUNS[run_id]["status"] = "running"
        try:
            config = PipelineConfig(
                project=project_name,
                step="all",
                blueprint="new_feature",
                resume=False,
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
