"""platform_findings: storage layer for observer/sre_autopilot findings (save/list, best-effort)."""

import pytest

from pydantic_agents.core import platform_findings
from pydantic_agents.core.settings import settings
from tests.unit.db_stub import stub_database


class _Conn:
    def __init__(self, db):
        self.db = db

    async def execute(self, sql, *args):
        if "INSERT INTO platform_findings" in sql:
            finding_id, kind, severity, agent, summary, evidence = args
            self.db[finding_id] = {
                "id": finding_id,
                "kind": kind,
                "severity": severity,
                "agent": agent,
                "summary": summary,
                "evidence": evidence,
                "status": "open",
                "created_at": None,
                "resolved_at": None,
            }

    async def fetch(self, sql, *args):
        rows = sorted(self.db.values(), key=lambda r: r["id"], reverse=True)
        *filters, limit = args
        if "status =" in sql and "kind =" in sql:
            status, kind = filters
            rows = [r for r in rows if r["status"] == status and r["kind"] == kind]
        elif "status =" in sql:
            (status,) = filters
            rows = [r for r in rows if r["status"] == status]
        elif "kind =" in sql:
            (kind,) = filters
            rows = [r for r in rows if r["kind"] == kind]
        return rows[:limit]


@pytest.fixture
def pool(monkeypatch):
    return stub_database(monkeypatch, platform_findings, _Conn)


async def test_save_and_list_a_finding(pool):
    finding_id = await platform_findings.save_finding(
        "regression", "high", "latency doubled", agent="review", evidence={"p95_before": 1.0, "p95_after": 2.1}
    )
    assert finding_id is not None
    rows = await platform_findings.list_findings()
    assert len(rows) == 1
    assert rows[0]["id"] == finding_id
    assert rows[0]["kind"] == "regression"
    assert rows[0]["status"] == "open"
    assert rows[0]["evidence"] == {"p95_before": 1.0, "p95_after": 2.1}


async def test_list_filters_by_status_and_kind(pool):
    await platform_findings.save_finding("flaky", "low", "intermittent timeout", agent="sre")
    await platform_findings.save_finding("health", "medium", "db pool exhausted")
    assert len(await platform_findings.list_findings(kind="flaky")) == 1
    assert len(await platform_findings.list_findings(status="open", kind="health")) == 1
    assert await platform_findings.list_findings(status="resolved") == []


async def test_invalid_kind_is_rejected():
    with pytest.raises(ValueError, match="Invalid finding kind"):
        await platform_findings.save_finding("not-a-kind", "low", "x")


async def test_disabled_without_a_database(monkeypatch):
    monkeypatch.setattr(settings, "database_url", None)
    assert await platform_findings.save_finding("health", "low", "x") is None
    assert await platform_findings.list_findings() == []


async def test_unreachable_database_is_best_effort(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql://stub/db")

    async def boom():
        raise ConnectionError("db down")

    monkeypatch.setattr(platform_findings.vector_db, "_get_pool", boom)
    assert await platform_findings.save_finding("health", "low", "x") is None
    assert await platform_findings.list_findings() == []
