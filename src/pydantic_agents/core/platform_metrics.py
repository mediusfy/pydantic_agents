"""Per-call agent performance/quality signal, in the pgvector PostgreSQL database (table ``platform_metrics``).

This is additive to ``core/telemetry.py``'s JSONL write, not a replacement: telemetry keeps writing locally for
offline use, and the MCP call path also upserts here so the ``observer`` agent can query trends across runs.
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


def enabled() -> bool:
    return bool(settings.database_url)


async def _ensure_table(conn: Any) -> None:
    global _table_ready
    if _table_ready:
        return
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS platform_metrics (
            id              TEXT PRIMARY KEY,
            agent           TEXT NOT NULL,
            duration        DOUBLE PRECISION,
            success         BOOLEAN NOT NULL,
            input_tokens    INTEGER,
            output_tokens   INTEGER,
            cost_usd        DOUBLE PRECISION,
            verdict         TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_platform_metrics_agent_created ON platform_metrics (agent, created_at DESC)"
    )
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_platform_metrics_created ON platform_metrics (created_at DESC)")
    _table_ready = True


def verdict_of(result: Any) -> str | None:
    """'passed'/'failed' from a QualityGateResult.passed or ReviewResult.approved output, else None."""
    output = getattr(result, "output", result)
    for field in ("passed", "approved"):
        value = getattr(output, field, None)
        if isinstance(value, bool):
            return "passed" if value else "failed"
    return None


def token_counts(result: Any) -> tuple[int | None, int | None]:
    """(input, output) tokens of a pydantic_ai run result, or (None, None) when it does not report usage."""
    try:
        usage = result.usage()
        counts = (usage.input_tokens, usage.output_tokens)
    except Exception:
        return None, None
    return tuple(c if isinstance(c, int) else None for c in counts)  # type: ignore[return-value]


async def record_call(
    agent: str,
    *,
    duration: float | None = None,
    success: bool = True,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    verdict: str | None = None,
) -> None:
    """Insert one row for an agent call. Never raises."""
    if not enabled():
        return

    async def _op(conn: Any) -> None:
        await _ensure_table(conn)
        await conn.execute(
            """
            INSERT INTO platform_metrics
                (id, agent, duration, success, input_tokens, output_tokens, cost_usd, verdict)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            f"pm_{uuid.uuid4().hex[:12]}",
            agent,
            duration,
            success,
            input_tokens,
            output_tokens,
            cost_usd,
            verdict,
        )

    try:
        await vector_db.with_connection(_op)
    except Exception as exc:  # best-effort by design
        logger.warning("Could not persist platform metric for %s (%s: %s)", agent, type(exc).__name__, exc)


async def recent_calls(agent: str | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
    """The most recent rows, newest first, as plain dicts (ISO timestamps). Never raises; [] with no database."""
    if not enabled():
        return []

    async def _op(conn: Any) -> Any:
        await _ensure_table(conn)
        if agent is None:
            return await conn.fetch("SELECT * FROM platform_metrics ORDER BY created_at DESC LIMIT $1", limit)
        else:
            return await conn.fetch(
                "SELECT * FROM platform_metrics WHERE agent = $1 ORDER BY created_at DESC LIMIT $2",
                agent,
                limit,
            )

    try:
        rows = await vector_db.with_connection(_op)
    except Exception as exc:
        logger.warning("Could not load platform metrics (%s: %s)", type(exc).__name__, exc)
        return []
    out = []
    for row in rows:
        item = dict(row)
        ts = item.get("created_at")
        item["created_at"] = ts.isoformat() if isinstance(ts, datetime) else ts
        out.append(item)
    return out
