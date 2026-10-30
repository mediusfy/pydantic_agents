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


USAGE_DEFAULT_DAYS = 7
USAGE_MAX_DAYS = 90

_USAGE_SQL = """
    SELECT agent,
           count(*)                                                  AS runs,
           count(*) FILTER (WHERE success)                           AS successes,
           percentile_cont(0.5)  WITHIN GROUP (ORDER BY duration)    AS p50_seconds,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY duration)    AS p95_seconds,
           count(input_tokens)                                       AS runs_with_tokens,
           sum(input_tokens)                                         AS input_tokens,
           sum(output_tokens)                                        AS output_tokens,
           count(*) FILTER (WHERE verdict = 'passed')                AS verdict_passed,
           count(*) FILTER (WHERE verdict = 'failed')                AS verdict_failed,
           max(created_at) FILTER (WHERE NOT success)                AS last_failure
    FROM platform_metrics
    WHERE created_at >= now() - make_interval(days => $1)
    {agent_filter}
    GROUP BY agent
    ORDER BY runs DESC, agent
"""


def clamp_days(days: Any) -> int:
    """``days`` as an int in 1..USAGE_MAX_DAYS; anything that is not a number falls back to the default."""
    try:
        value = int(days)
    except (TypeError, ValueError):
        return USAGE_DEFAULT_DAYS
    return max(1, min(USAGE_MAX_DAYS, value))


async def usage_by_agent(days: int, agent: str = "") -> list[dict[str, Any]]:
    """Per-agent aggregates over the last ``days`` days (already clamped by the caller), most-run agent first.

    Raises on a database error; ``usage_summary`` is the never-raising, text-returning wrapper.
    """

    async def _op(conn: Any) -> Any:
        await _ensure_table(conn)
        if agent:
            return await conn.fetch(_USAGE_SQL.format(agent_filter="AND agent = $2"), days, agent)
        return await conn.fetch(_USAGE_SQL.format(agent_filter=""), days)

    rows = await vector_db.with_connection(_op)
    return [dict(r) for r in rows]


def _fmt_seconds(value: Any) -> str:
    return f"{float(value):.1f}s" if value is not None else "n/a"


def _fmt_time(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value) if value is not None else "never"


def format_usage(rows: list[dict[str, Any]], days: int, agent: str = "") -> str:
    """Render ``usage_by_agent`` rows as plain text, saying so when token counts are missing rather than zero."""
    scope = f"agent '{agent}'" if agent else "all agents"
    if not rows:
        return f"No platform_metrics rows for {scope} in the last {days} day(s)."
    lines = [f"Agent usage for {scope}, last {days} day(s) (source: platform_metrics):"]
    for r in rows:
        runs = int(r["runs"])
        rate = 100.0 * int(r["successes"]) / runs if runs else 0.0
        with_tokens = int(r.get("runs_with_tokens") or 0)
        if with_tokens == 0:
            tokens = "tokens n/a (no run in this window reported token usage)"
        else:
            tokens = f"tokens in={int(r['input_tokens'] or 0)} out={int(r['output_tokens'] or 0)}"
            if with_tokens < runs:
                tokens += f" (only {with_tokens} of {runs} runs reported tokens; the rest are NULL, not zero)"
        lines.append(
            f"- {r['agent']}: runs={runs}, success={rate:.0f}%, "
            f"duration p50={_fmt_seconds(r['p50_seconds'])} p95={_fmt_seconds(r['p95_seconds'])}, {tokens}, "
            f"verdict passed={int(r['verdict_passed'])} failed={int(r['verdict_failed'])}, "
            f"last failure={_fmt_time(r['last_failure'])}"
        )
    return "\n".join(lines)


async def usage_summary(days: Any = USAGE_DEFAULT_DAYS, agent: str = "") -> str:
    """Human/agent-readable usage summary. Never raises: no database and database errors become a message."""
    window = clamp_days(days)
    name = (agent or "").strip()[:100]
    if not enabled():
        return "No database configured (DATABASE_URL is unset), so no usage data is recorded or available."
    try:
        rows = await usage_by_agent(window, name)
    except Exception as exc:
        logger.warning("Could not summarise platform metrics (%s: %s)", type(exc).__name__, exc)
        return f"Could not query platform_metrics ({type(exc).__name__}: {exc})."
    return format_usage(rows, window, name)
