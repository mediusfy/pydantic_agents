"""sre_autopilot_agent's own tools, called directly (no LLM). AP-83 acceptance: an allowlisted issue (an
interrupted pipeline run) gets auto-remediated; a non-allowlisted issue (a startup-check error) escalates and
is NOT acted on."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.agents.sre_autopilot import health_snapshot, scan_and_remediate
from pydantic_agents.core import platform_findings
from tests.unit.db_stub import stub_database

CTX = MagicMock()


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
            }

    async def fetch(self, sql, *args):
        rows = sorted(self.db.values(), key=lambda r: r["id"], reverse=True)
        *_filters, limit = args
        return rows[:limit]


@pytest.fixture
def pool(monkeypatch):
    return stub_database(monkeypatch, platform_findings, _Conn)


async def test_scan_and_remediate_auto_resumes_an_interrupted_run(pool):
    interrupted = {"pipe_abc123": {"project": "demo", "status": "interrupted"}}
    with (
        patch("pydantic_agents.agents.sre_autopilot.healthcheck.run_startup_checks", return_value=[]),
        patch("pydantic_agents.agents.sre_autopilot.pipeline_jobs.list_interrupted_runs", return_value=interrupted),
        patch(
            "pydantic_agents.agents.sre_autopilot.pipeline_jobs.resume_pipeline_run", new_callable=AsyncMock
        ) as resume,
    ):
        result = await scan_and_remediate(CTX)

    resume.assert_awaited_once_with("pipe_abc123")
    assert "[auto-remediated]" in result
    assert "pipe_abc123" in result
    saved = await platform_findings.list_findings()
    assert len(saved) == 1
    assert saved[0]["kind"] == "health"


async def test_scan_and_remediate_escalates_a_startup_check_error_without_acting(pool):
    with (
        patch(
            "pydantic_agents.agents.sre_autopilot.healthcheck.run_startup_checks",
            return_value=["DATABASE_URL not set"],
        ),
        patch("pydantic_agents.agents.sre_autopilot.pipeline_jobs.list_interrupted_runs", return_value={}),
        patch(
            "pydantic_agents.agents.sre_autopilot.pipeline_jobs.resume_pipeline_run", new_callable=AsyncMock
        ) as resume,
    ):
        result = await scan_and_remediate(CTX)

    resume.assert_not_awaited()
    assert "[escalated]" in result
    assert "DATABASE_URL not set" in result
    saved = await platform_findings.list_findings()
    assert len(saved) == 1
    assert saved[0]["severity"] == "high"


async def test_scan_and_remediate_reports_nothing_when_healthy(pool):
    with (
        patch("pydantic_agents.agents.sre_autopilot.healthcheck.run_startup_checks", return_value=[]),
        patch("pydantic_agents.agents.sre_autopilot.pipeline_jobs.list_interrupted_runs", return_value={}),
    ):
        assert await scan_and_remediate(CTX) == "No issues."
    assert await platform_findings.list_findings() == []


async def test_health_snapshot_returns_json():
    with (
        patch(
            "pydantic_agents.agents.sre_autopilot.healthcheck.run_startup_checks",
            return_value=["bad key"],
        ),
        patch("pydantic_agents.agents.sre_autopilot.pipeline_jobs.list_interrupted_runs", return_value={}),
    ):
        payload = json.loads(await health_snapshot(CTX))
    assert payload["startup_check_errors"] == ["bad key"]
    assert payload["interrupted_runs"] == {}
