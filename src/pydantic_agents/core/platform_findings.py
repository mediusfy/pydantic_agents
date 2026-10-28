"""Structured findings written by the ``observer``/``sre_autopilot`` agents, in the pgvector PostgreSQL database
(table ``platform_findings``). Storage layer only — AP-77/ADR-0001 agents consume ``save_finding``/``list_findings``
through their own MCP tools; this module adds no MCP tool itself.

Persistence is best-effort, like the other database helpers: with no DATABASE_URL, or an unreachable database,
every call logs and returns, and the caller behaves as if nothing happened.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from pydantic_agents.core import vector_db
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

_table_ready = False

VALID_KINDS = frozenset({"regression", "flaky", "pattern", "health"})
VALID_STATUSES = frozenset({"open", "acknowledged", "resolved"})


def enabled() -> bool:
    return bool(settings.database_url)


async def _ensure_table(conn: Any) -> None:
    global _table_ready
    if _table_ready:
        return
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS platform_findings (
            id          TEXT PRIMARY KEY,
            kind        TEXT NOT NULL,
            severity    TEXT NOT NULL,
            agent       TEXT,
            summary     TEXT NOT NULL,
            evidence    JSONB,
            status      TEXT NOT NULL DEFAULT 'open',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            resolved_at TIMESTAMPTZ
        )
        """
    )
    _table_ready = True


async def save_finding(
    kind: str,
    severity: str,
    summary: str,
    *,
    agent: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> str | None:
    """Insert one finding and return its id, or None (no-op, logged) when the database is unavailable."""
    if kind not in VALID_KINDS:
        raise ValueError(f"Invalid finding kind {kind!r}: expected one of {sorted(VALID_KINDS)}")
    if not enabled():
        return None
    finding_id = f"find_{uuid.uuid4().hex[:12]}"
    try:
        pool = await vector_db._get_pool()
        async with pool.acquire() as conn:
            await _ensure_table(conn)
            await conn.execute(
                """
                INSERT INTO platform_findings (id, kind, severity, agent, summary, evidence, status)
                VALUES ($1, $2, $3, $4, $5, $6, 'open')
                """,
                finding_id,
                kind,
                severity,
                agent,
                summary,
                vector_db.json.dumps(evidence) if evidence is not None else None,
            )
    except Exception as exc:  # best-effort by design
        logger.warning("Could not persist platform finding (%s: %s)", type(exc).__name__, exc)
        return None
    return finding_id


async def list_findings(*, status: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Findings newest first, as plain dicts (ISO timestamps). Never raises; [] with no database."""
    if not enabled():
        return []
    clauses: list[str] = []
    args: list[Any] = []
    if status is not None:
        args.append(status)
        clauses.append(f"status = ${len(args)}")
    if kind is not None:
        args.append(kind)
        clauses.append(f"kind = ${len(args)}")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit)
    try:
        pool = await vector_db._get_pool()
        async with pool.acquire() as conn:
            await _ensure_table(conn)
            rows = await conn.fetch(
                f"SELECT * FROM platform_findings {where} ORDER BY created_at DESC LIMIT ${len(args)}", *args
            )
    except Exception as exc:
        logger.warning("Could not load platform findings (%s: %s)", type(exc).__name__, exc)
        return []
    out = []
    for row in rows:
        item = dict(row)
        for key in ("created_at", "resolved_at"):
            ts = item.get(key)
            item[key] = ts.isoformat() if isinstance(ts, datetime) else ts
        if item.get("evidence"):
            item["evidence"] = vector_db.json.loads(item["evidence"])
        out.append(item)
    return out
