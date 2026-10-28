"""observer_agent's own tools, called directly (no LLM): recent_metrics/open_findings read, scan_for_findings
writes. AP-80 acceptance: a fixture of fake metrics rows produces at least one finding."""

import json
from unittest.mock import MagicMock

import pytest

from pydantic_agents.agents.observer import open_findings, recent_metrics, scan_for_findings
from pydantic_agents.core import platform_findings, platform_metrics
from tests.unit.db_stub import fetch_findings, insert_finding, stub_database

CTX = MagicMock()


class _Conn:
    """Answers both platform_metrics and platform_findings SQL over one shared ``db`` dict (keyed by row id),
    matching production where both modules share the one real database pool via ``vector_db._get_pool()``.
    """

    def __init__(self, db):
        self.db = db

    async def execute(self, sql, *args):
        if "INSERT INTO platform_metrics" in sql:
            row_id, agent, duration, success, *_rest = args
            self.db[row_id] = {"id": row_id, "agent": agent, "duration": duration, "success": success, "_t": "m"}
        elif "INSERT INTO platform_findings" in sql:
            insert_finding(self.db, args)

    async def fetch(self, sql, *args):
        if "platform_metrics" in sql:
            rows = sorted((r for r in self.db.values() if r["_t"] == "m"), key=lambda r: r["id"], reverse=True)
            (limit,) = args
            return rows[:limit]
        return fetch_findings(self.db, sql, args)


@pytest.fixture
def pool(monkeypatch):
    stub_database(monkeypatch, platform_metrics, _Conn)
    return stub_database(monkeypatch, platform_findings, _Conn)


async def test_scan_for_findings_persists_a_regression_from_a_metrics_fixture(pool):
    for _ in range(3):
        await platform_metrics.record_call("backend", duration=1.0, success=False)

    result = await scan_for_findings(CTX, limit=200)

    assert "regression" in result
    assert "backend" in result
    saved = await platform_findings.list_findings()
    assert len(saved) == 1
    assert saved[0]["kind"] == "regression"
    assert saved[0]["agent"] == "backend"


async def test_scan_for_findings_does_not_duplicate_an_open_finding(pool):
    for _ in range(3):
        await platform_metrics.record_call("backend", duration=1.0, success=False)

    await scan_for_findings(CTX, limit=200)
    second = await scan_for_findings(CTX, limit=200)

    assert "already open" in second
    assert len(await platform_findings.list_findings()) == 1


async def test_scan_for_findings_reports_nothing_when_all_calls_pass(pool):
    await platform_metrics.record_call("review", duration=1.0, success=True)
    assert await scan_for_findings(CTX, limit=200) == "No findings."
    assert await platform_findings.list_findings() == []


async def test_recent_metrics_returns_json(pool):
    await platform_metrics.record_call("review", duration=1.0, success=True)
    rows = json.loads(await recent_metrics(CTX, limit=10))
    assert rows[0]["agent"] == "review"


async def test_open_findings_returns_json(pool):
    await platform_findings.save_finding("health", "low", "db pool exhausted")
    rows = json.loads(await open_findings(CTX, limit=10))
    assert rows[0]["summary"] == "db pool exhausted"
