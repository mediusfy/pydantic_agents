"""platform_metrics.usage_summary and the agent_usage_summary MCP tool (read-only, parameterised, NULL-honest)."""

from datetime import UTC, datetime

import pytest

from pydantic_agents.core import platform_metrics
from pydantic_agents.core.settings import settings
from tests.unit.db_stub import stub_database

_ROW = {
    "agent": "review",
    "runs": 4,
    "successes": 3,
    "p50_seconds": 2.0,
    "p95_seconds": 9.5,
    "runs_with_tokens": 4,
    "input_tokens": 1000,
    "output_tokens": 200,
    "verdict_passed": 2,
    "verdict_failed": 1,
    "last_failure": datetime(2026, 10, 29, 12, 0, tzinfo=UTC),
}


class _Conn:
    calls: list[tuple[str, tuple]] = []
    rows: list[dict] = []

    def __init__(self, _db):
        pass

    async def execute(self, sql, *args):
        pass

    async def fetch(self, sql, *args):
        type(self).calls.append((sql, args))
        return type(self).rows


@pytest.fixture
def conn(monkeypatch):
    _Conn.calls = []
    _Conn.rows = [_ROW]
    stub_database(monkeypatch, platform_metrics, _Conn)
    return _Conn


@pytest.mark.parametrize(
    ("given", "expected"),
    [(7, 7), (0, 1), (-5, 1), (500, 90), (90, 90), ("30", 30), ("x", 7), (None, 7)],
)
def test_clamp_days(given, expected):
    assert platform_metrics.clamp_days(given) == expected


async def test_summary_reports_every_figure(conn):
    text = await platform_metrics.usage_summary(7)
    assert "review: runs=4, success=75%" in text
    assert "p50=2.0s p95=9.5s" in text
    assert "tokens in=1000 out=200" in text
    assert "verdict passed=2 failed=1" in text
    assert "last failure=2026-10-29T12:00:00+00:00" in text


async def test_sql_is_parameterised_and_clamped(conn):
    await platform_metrics.usage_summary(9999, "rev'; DROP TABLE platform_metrics;--")
    sql, args = conn.calls[0]
    assert args == (90, "rev'; DROP TABLE platform_metrics;--")
    assert "DROP" not in sql and "$2" in sql
    assert sql.lstrip().upper().startswith("SELECT")


async def test_no_agent_filter_binds_only_days(conn):
    await platform_metrics.usage_summary(3)
    sql, args = conn.calls[0]
    assert args == (3,)
    assert "$2" not in sql


async def test_null_tokens_are_reported_as_unknown_not_zero(conn):
    conn.rows = [{**_ROW, "runs_with_tokens": 0, "input_tokens": None, "output_tokens": None}]
    text = await platform_metrics.usage_summary()
    assert "tokens n/a" in text
    assert "in=0" not in text


async def test_partial_tokens_are_flagged(conn):
    conn.rows = [{**_ROW, "runs_with_tokens": 1}]
    text = await platform_metrics.usage_summary()
    assert "only 1 of 4 runs reported tokens" in text


async def test_never_failed_agent(conn):
    conn.rows = [{**_ROW, "last_failure": None, "p50_seconds": None, "p95_seconds": None}]
    text = await platform_metrics.usage_summary()
    assert "last failure=never" in text and "p50=n/a" in text


@pytest.mark.parametrize(("agent", "scope"), [("", "all agents"), ("review", "agent 'review'")])
async def test_no_rows_message(conn, agent, scope):
    conn.rows = []
    text = await platform_metrics.usage_summary(7, agent)
    assert text.startswith("No platform_metrics rows") and scope in text


async def test_no_database_message(monkeypatch):
    monkeypatch.setattr(settings, "database_url", None)
    assert "No database configured" in await platform_metrics.usage_summary()


async def test_database_error_is_a_message(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql://stub/db")

    async def boom():
        raise ConnectionError("db down")

    monkeypatch.setattr(platform_metrics.vector_db, "_get_pool", boom)
    text = await platform_metrics.usage_summary()
    assert "Could not query platform_metrics" in text and "db down" in text


async def test_mcp_tool_is_registered_async_and_delegates(conn):
    import asyncio

    from pydantic_agents import mcp_server

    tool = mcp_server.mcp._tool_manager.get_tool("agent_usage_summary")
    assert tool is not None and tool.is_async
    assert "review: runs=4" in await mcp_server.agent_usage_summary(7, "")
    assert asyncio.iscoroutinefunction(mcp_server.agent_usage_summary)
