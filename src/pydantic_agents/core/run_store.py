"""Durable record of background pipeline runs, in the pgvector PostgreSQL database (table ``pipeline_runs``).

``PIPELINE_RUNS`` in ``pipeline_jobs`` stays the in-process view (it holds the asyncio task handle); this store
is what survives a restart, so a poll after the pod restarted still finds the run and can say it was interrupted.
Persistence is best-effort, like the other database helpers: with no DATABASE_URL, or an unreachable database,
every call logs and returns None, and runs behave as before (memory only).
"""

import logging
from datetime import datetime
from typing import Any

from pydantic_agents.core import vector_db
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

_COLUMNS = ("status", "project", "requirement", "result", "error", "started_at", "completed_at", "elapsed_seconds")
# Per process: skips CREATE TABLE IF NOT EXISTS after the first success. Two concurrent first calls both run it,
# which is harmless because the statement is idempotent.
_table_ready = False


def enabled() -> bool:
    return bool(settings.database_url)


async def _ensure_table(conn: Any) -> None:
    global _table_ready
    if _table_ready:
        return
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id          TEXT PRIMARY KEY,
            status          TEXT NOT NULL,
            project         TEXT NOT NULL,
            requirement     TEXT,
            result          TEXT,
            error           TEXT,
            started_at      TIMESTAMPTZ,
            completed_at    TIMESTAMPTZ,
            elapsed_seconds DOUBLE PRECISION,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    _table_ready = True


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


async def save_run(run_id: str, run: dict[str, Any]) -> None:
    """Upsert the persistable fields of a run. Never raises."""
    if not enabled():
        return
    try:
        pool = await vector_db._get_pool()
        async with pool.acquire() as conn:
            await _ensure_table(conn)
            await conn.execute(
                """
                INSERT INTO pipeline_runs
                    (run_id, status, project, requirement, result, error, started_at, completed_at, elapsed_seconds)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (run_id) DO UPDATE SET
                    status = EXCLUDED.status, result = EXCLUDED.result, error = EXCLUDED.error,
                    completed_at = EXCLUDED.completed_at, elapsed_seconds = EXCLUDED.elapsed_seconds,
                    updated_at = now()
                """,
                run_id,
                run.get("status"),
                run.get("project"),
                run.get("requirement"),
                run.get("result"),
                run.get("error"),
                _ts(run.get("started_at")),
                _ts(run.get("completed_at")),
                run.get("elapsed_seconds"),
            )
    except Exception as exc:  # best-effort by design
        logger.warning("Could not persist pipeline run %s (%s: %s)", run_id, type(exc).__name__, exc)


async def recent_requirements(limit: int = 50) -> list[dict[str, Any]]:
    """Recent requirements across pipeline runs (run_id, project, requirement, status, completed_at), newest
    first, as plain dicts (ISO timestamps). Read-only, for the `product_strategist` agent's requirement-history
    signal (ADR-0001 / AP-82). Never raises; [] with no database or on error.
    """
    if not enabled():
        return []
    try:
        pool = await vector_db._get_pool()
        async with pool.acquire() as conn:
            await _ensure_table(conn)
            rows = await conn.fetch(
                """
                SELECT run_id, project, requirement, status, completed_at
                FROM pipeline_runs
                WHERE requirement IS NOT NULL
                ORDER BY completed_at DESC NULLS LAST
                LIMIT $1
                """,
                limit,
            )
    except Exception as exc:
        logger.warning("Could not load recent requirements (%s: %s)", type(exc).__name__, exc)
        return []
    out = []
    for row in rows:
        item = dict(row)
        item["completed_at"] = item["completed_at"].isoformat() if item.get("completed_at") else None
        out.append(item)
    return out


async def load_run(run_id: str) -> dict[str, Any] | None:
    """The stored run as a ``PIPELINE_RUNS``-shaped dict (ISO timestamps), or None. Never raises."""
    if not enabled():
        return None
    try:
        pool = await vector_db._get_pool()
        async with pool.acquire() as conn:
            await _ensure_table(conn)
            row = await conn.fetchrow("SELECT * FROM pipeline_runs WHERE run_id = $1", run_id)
    except Exception as exc:
        logger.warning("Could not load pipeline run %s (%s: %s)", run_id, type(exc).__name__, exc)
        return None
    if row is None:
        return None
    run = {k: row[k] for k in _COLUMNS}
    for key in ("started_at", "completed_at"):
        run[key] = run[key].isoformat() if run[key] else None
    return run
