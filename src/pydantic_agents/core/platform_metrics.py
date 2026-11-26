"""Per-call agent performance/quality signal, in the pgvector PostgreSQL database (table ``platform_metrics``).

This is additive to ``core/telemetry.py``'s JSONL write, not a replacement: telemetry keeps writing locally for
offline use, and the agent provider also inserts here so trends, spend (`cost_usd`), and per-run (`run_id`) costs can
be queried across runs. Persistence is best-effort, like the other database helpers: with no DATABASE_URL, or an
unreachable database, every call logs and returns, and the caller behaves as if nothing happened.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from pydantic_agents.core import vector_db
from pydantic_agents.core.limits import run_id_var
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)

_table_ready = False
# Heuristic: a configured model whose name contains one of these is billed at the cheaper sub-agent rate, anything
# else at the default rate. Costs are estimates from list rates in settings, not invoiced spend.
_SUBAGENT_MODEL_HINTS = ("subagent", "mini", "flash", "haiku")
_warned_unknown_model: set[str] = set()


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
            run_id          TEXT,
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
    await conn.execute("ALTER TABLE platform_metrics ADD COLUMN IF NOT EXISTS run_id TEXT")
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_platform_metrics_agent_created ON platform_metrics (agent, created_at DESC)"
    )
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_platform_metrics_created ON platform_metrics (created_at DESC)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_platform_metrics_run_id ON platform_metrics (run_id)")
    _table_ready = True


def verdict_of(result: Any) -> str | None:
    """'passed'/'failed' from structured gate/verdict outputs, else None."""
    output = getattr(result, "output", result)
    for field in ("passed", "approved", "approved_to_proceed", "all_passed"):
        value = getattr(output, field, None)
        if isinstance(value, bool):
            return "passed" if value else "failed"
    over_budget = getattr(output, "over_budget", None)
    if isinstance(over_budget, bool):
        return "failed" if over_budget else "passed"
    return None


def token_counts(result: Any) -> tuple[int | None, int | None]:
    """(input, output) tokens of a pydantic_ai run result, or (None, None) when it does not report usage."""
    try:
        usage = result.usage()
        counts = (usage.input_tokens, usage.output_tokens)
    except Exception:
        return None, None
    return tuple(c if isinstance(c, int) else None for c in counts)  # type: ignore[return-value]


def _agent_model_name(agent_name: str) -> str:
    if agent_name == "test_eng":
        return str(settings.test_model)
    configured = getattr(settings, f"{agent_name}_model", None)
    if configured:
        return str(configured)
    if agent_name not in _warned_unknown_model:
        _warned_unknown_model.add(agent_name)
        logger.warning(
            "platform_metrics: no '%s_model' setting for agent %r; estimating cost_usd at the default rate",
            agent_name,
            agent_name,
        )
    return ""


def estimate_cost_usd(
    agent_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    model_name: str | None = None,
) -> float | None:
    """Estimated USD cost for a call's token counts, or None when neither input nor output tokens were reported."""
    if input_tokens is None and output_tokens is None:
        return None
    resolved_model = (model_name if model_name is not None else _agent_model_name(agent_name)).lower()
    if any(hint in resolved_model for hint in _SUBAGENT_MODEL_HINTS):
        in_rate = settings.cost_subagent_input_per_1m_usd
        out_rate = settings.cost_subagent_output_per_1m_usd
    else:
        in_rate = settings.cost_default_input_per_1m_usd
        out_rate = settings.cost_default_output_per_1m_usd
    in_cost = max(0, input_tokens or 0) * in_rate / 1_000_000.0
    out_cost = max(0, output_tokens or 0) * out_rate / 1_000_000.0
    return round(in_cost + out_cost, 8)


async def record_call(
    agent: str,
    *,
    run_id: str | None = None,
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
    effective_run_id = run_id if run_id is not None else (run_id_var.get() or None)
    effective_cost = (
        cost_usd if cost_usd is not None else estimate_cost_usd(agent, input_tokens, output_tokens)
    )

    async def _op(conn: Any) -> None:
        await _ensure_table(conn)
        await conn.execute(
            """
            INSERT INTO platform_metrics
                (id, agent, run_id, duration, success, input_tokens, output_tokens, cost_usd, verdict)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            f"pm_{uuid.uuid4().hex[:12]}",
            agent,
            effective_run_id,
            duration,
            success,
            input_tokens,
            output_tokens,
            effective_cost,
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
           sum(cost_usd)                                             AS cost_usd,
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
        cost_val = r.get("cost_usd")
        cost_part = f", cost=${float(cost_val):.4f}" if cost_val is not None else ""
        lines.append(
            f"- {r['agent']}: runs={runs}, success={rate:.0f}%, "
            f"duration p50={_fmt_seconds(r['p50_seconds'])} p95={_fmt_seconds(r['p95_seconds'])}, {tokens}{cost_part}, "
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
