"""Event Broker — asynchronous handoffs for heavy workloads.

ASYNC HANDOFFS FOR HEAVY WORKLOADS:
Instead of synchronous MCP calls that may timeout on long-running operations
(builds, test suites, deployments), the event broker provides an event-driven
model. The Orchestrator fires an event → worker agents pick it up → publish
a "Task Complete" event back.

This module provides a PostgreSQL-backed persistent event broker.
"""

import asyncio
import enum
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import logfire

from pydantic_agents.core.vector_db import _get_pool

logger = logging.getLogger(__name__)


class EventKind(enum.Enum):
    TASK_SUBMITTED = "task_submitted"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_TIMEOUT = "task_timeout"


@dataclass
class Event:
    kind: EventKind
    data: dict[str, Any]
    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


class PersistentEventBroker:
    """PostgreSQL-backed persistent event broker for async task handoffs.

    Handles task submission, status tracking, and completion notification.
    Worker agents check for pending tasks and publish completion events.
    """

    def __init__(self) -> None:
        self._tables_ensured = False

    async def _ensure_tables(self):
        if self._tables_ensured:
            return
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS broker_tasks (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    result TEXT,
                    created_at FLOAT NOT NULL,
                    completed_at FLOAT
                );
                CREATE TABLE IF NOT EXISTS broker_events (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    data JSONB NOT NULL,
                    timestamp FLOAT NOT NULL
                );
            """)
        self._tables_ensured = True

    async def submit(self, event: Event) -> str:
        """Submit a task event and return a task ID for polling."""
        await self._ensure_tables()
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO broker_tasks (id, status, created_at) VALUES ($1, $2, $3)", task_id, "pending", time.time()
            )
            await conn.execute(
                "INSERT INTO broker_events (id, kind, data, timestamp) VALUES ($1, $2, $3::jsonb, $4)",
                event.id,
                event.kind.value,
                json.dumps(event.data),
                event.timestamp,
            )

        with logfire.span("event_broker.submit", task_id=task_id, kind=event.kind.value, _level="info"):
            logger.info("Task %s submitted (%s)", task_id, event.kind.value)

        return task_id

    async def complete(self, task_id: str, result: str) -> None:
        """Mark a task as completed with a result."""
        await self._ensure_tables()
        pool = await _get_pool()
        async with pool.acquire() as conn:
            res = await conn.execute(
                "UPDATE broker_tasks SET status = 'completed', result = $1, completed_at = $2 WHERE id = $3",
                result,
                time.time(),
                task_id,
            )
            if res == "UPDATE 0":
                logger.warning("Cannot complete unknown task: %s", task_id)
                return

            evt = Event(kind=EventKind.TASK_COMPLETED, data={"task_id": task_id, "result": result})
            await conn.execute(
                "INSERT INTO broker_events (id, kind, data, timestamp) VALUES ($1, $2, $3::jsonb, $4)",
                evt.id,
                evt.kind.value,
                json.dumps(evt.data),
                evt.timestamp,
            )

        # Store in vector DB for future context retrieval
        try:
            from pydantic_agents.core.vector_db import vector_db

            store_result = result if isinstance(result, str) else str(result)
            vector_db.add_documents(
                "project",
                [f"Task {task_id} completed: {store_result}"],
                [{"task_id": task_id, "status": "completed", "source": "event_broker"}],
                [f"task:{task_id}"],
            )
        except Exception as e:
            logger.debug("Could not store task result in vector DB: %s", e)

        with logfire.span("event_broker.complete", task_id=task_id, _level="info"):
            logger.info("Task %s completed (stored in vector DB)", task_id)

    async def fail(self, task_id: str, error: str) -> None:
        """Mark a task as failed with an error message."""
        await self._ensure_tables()
        pool = await _get_pool()
        async with pool.acquire() as conn:
            res = await conn.execute(
                "UPDATE broker_tasks SET status = 'failed', result = $1, completed_at = $2 WHERE id = $3",
                error,
                time.time(),
                task_id,
            )
            if res == "UPDATE 0":
                logger.warning("Cannot fail unknown task: %s", task_id)
                return

            evt = Event(kind=EventKind.TASK_FAILED, data={"task_id": task_id, "error": error})
            await conn.execute(
                "INSERT INTO broker_events (id, kind, data, timestamp) VALUES ($1, $2, $3::jsonb, $4)",
                evt.id,
                evt.kind.value,
                json.dumps(evt.data),
                evt.timestamp,
            )

        with logfire.span("event_broker.fail", task_id=task_id, _level="warn"):
            logger.warning("Task %s failed: %s", task_id, error)

    async def get_status(self, task_id: str) -> dict[str, Any] | None:
        """Get the current status of a task."""
        await self._ensure_tables()
        pool = await _get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM broker_tasks WHERE id = $1", task_id)
            if not row:
                return None
            return {
                "status": row["status"],
                "result": row["result"] if row["status"] in ("completed", "failed") else "pending",
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "elapsed_seconds": round(time.time() - row["created_at"], 1),
            }

    async def wait_for(self, task_id: str, timeout_seconds: int = 300) -> dict[str, Any] | None:
        """Wait asynchronously for a task to complete via polling."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            status = await self.get_status(task_id)
            if not status:
                return None
            if status["status"] in ("completed", "failed"):
                return status
            await asyncio.sleep(2)

        # Timeout
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE broker_tasks SET status = 'timeout' WHERE id = $1 AND status = 'pending'", task_id
            )

        with logfire.span("event_broker.timeout", task_id=task_id, timeout=timeout_seconds, _level="warn"):
            logger.warning("Task %s timed out after %ds", task_id, timeout_seconds)

        return {
            "status": "timeout",
            "result": f"Timed out after {timeout_seconds}s",
            "created_at": status["created_at"] if status else start_time,
            "completed_at": None,
            "elapsed_seconds": timeout_seconds,
        }

    async def get_pending_tasks(self) -> list[dict[str, Any]]:
        """Get all pending tasks — for worker agents to discover work."""
        await self._ensure_tables()
        pool = await _get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, status, created_at FROM broker_tasks WHERE status = 'pending'")
            return [{"id": r["id"], "status": r["status"], "created_at": r["created_at"]} for r in rows]

    async def get_events(self, since_id: str | None = None) -> list[Event]:
        """Get all events since a given event ID."""
        await self._ensure_tables()
        pool = await _get_pool()
        async with pool.acquire() as conn:
            if since_id:
                # Find the timestamp of since_id
                row = await conn.fetchrow("SELECT timestamp FROM broker_events WHERE id = $1", since_id)
                if row:
                    rows = await conn.fetch(
                        "SELECT * FROM broker_events WHERE timestamp > $1 ORDER BY timestamp ASC", row["timestamp"]
                    )
                else:
                    rows = []
            else:
                rows = await conn.fetch("SELECT * FROM broker_events ORDER BY timestamp ASC")

            return [
                Event(
                    kind=EventKind(r["kind"]),
                    data=json.loads(r["data"]) if isinstance(r["data"], str) else r["data"],
                    id=r["id"],
                    timestamp=r["timestamp"],
                )
                for r in rows
            ]


# Singleton
event_broker = PersistentEventBroker()
