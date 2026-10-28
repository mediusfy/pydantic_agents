"""platform_metrics: best-effort recording of per-call agent performance/quality signals."""

import pytest

from pydantic_agents.core import platform_metrics
from pydantic_agents.core.settings import settings


class _Conn:
    def __init__(self, db):
        self.db = db

    async def execute(self, sql, *args):
        if "INSERT INTO platform_metrics" in sql:
            row_id, agent, duration, success, input_tokens, output_tokens, cost_usd, verdict = args
            self.db[row_id] = {
                "id": row_id,
                "agent": agent,
                "duration": duration,
                "success": success,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "verdict": verdict,
                "created_at": None,
            }

    async def fetch(self, sql, *args):
        rows = sorted(self.db.values(), key=lambda r: r["id"], reverse=True)
        if "WHERE agent" in sql:
            agent, limit = args
            rows = [r for r in rows if r["agent"] == agent]
        else:
            (limit,) = args
        return rows[:limit]


class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_exc):
        return False


class _Pool:
    def __init__(self):
        self.db = {}

    def acquire(self):
        return _Ctx(_Conn(self.db))


@pytest.fixture
def pool(monkeypatch):
    p = _Pool()
    monkeypatch.setattr(settings, "database_url", "postgresql://stub/db")
    monkeypatch.setattr(platform_metrics, "_table_ready", False)

    async def get_pool():
        return p

    monkeypatch.setattr(platform_metrics.vector_db, "_get_pool", get_pool)
    return p


async def test_record_call_persists_a_row(pool):
    await platform_metrics.record_call("review", duration=1.5, success=True, verdict="passed")
    rows = await platform_metrics.recent_calls()
    assert len(rows) == 1
    assert rows[0]["agent"] == "review"
    assert rows[0]["success"] is True
    assert rows[0]["verdict"] == "passed"


async def test_recent_calls_filters_by_agent(pool):
    await platform_metrics.record_call("review")
    await platform_metrics.record_call("security")
    rows = await platform_metrics.recent_calls(agent="security")
    assert len(rows) == 1
    assert rows[0]["agent"] == "security"


async def test_disabled_without_a_database(monkeypatch):
    monkeypatch.setattr(settings, "database_url", None)
    await platform_metrics.record_call("review")  # no-op, must not raise
    assert await platform_metrics.recent_calls() == []


async def test_unreachable_database_is_best_effort(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql://stub/db")

    async def boom():
        raise ConnectionError("db down")

    monkeypatch.setattr(platform_metrics.vector_db, "_get_pool", boom)
    await platform_metrics.record_call("review")  # must not raise
    assert await platform_metrics.recent_calls() == []
