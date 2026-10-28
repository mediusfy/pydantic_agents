"""pipeline_jobs.list_interrupted_runs: the stored runs sre_autopilot may auto-resume (AP-83)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from pydantic_agents.core import pipeline_jobs, run_store
from tests.unit.db_stub import stub_database

BEFORE_START = pipeline_jobs.PROCESS_STARTED_AT - timedelta(minutes=5)
AFTER_START = datetime.now(UTC) + timedelta(minutes=5)


class _Conn:
    def __init__(self, db):
        self.db = db

    execute = AsyncMock(return_value=None)  # CREATE TABLE IF NOT EXISTS

    async def fetch(self, sql, statuses, updated_before, limit):
        assert "FROM pipeline_runs" in sql
        rows = [r for r in self.db.values() if r["status"] in statuses and r["updated_at"] < updated_before]
        return sorted(rows, key=lambda r: r["updated_at"])[:limit]


def _row(run_id, status, updated_at=BEFORE_START):
    return {
        "run_id": run_id,
        "status": status,
        "project": "demo",
        "requirement": "r",
        "result": None,
        "error": None,
        "started_at": None,
        "completed_at": None,
        "elapsed_seconds": None,
        "updated_at": updated_at,
    }


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", {})
    return stub_database(monkeypatch, run_store, _Conn).db


async def test_in_flight_rows_from_before_this_process_are_interrupted(db):
    for run_id, status in (("pipe_a", "running"), ("pipe_b", "pending"), ("pipe_c", pipeline_jobs.STATUS_WAITING_CTO)):
        db[run_id] = _row(run_id, status)

    found = await pipeline_jobs.list_interrupted_runs()

    assert set(found) == {"pipe_a", "pipe_b", "pipe_c"}
    assert {run["status"] for run in found.values()} == {"interrupted"}
    assert found["pipe_a"]["project"] == "demo"


async def test_finished_rows_are_excluded(db):
    for run_id, status in (("pipe_d", "completed"), ("pipe_f", "failed"), ("pipe_x", "cancelled")):
        db[run_id] = _row(run_id, status)
    assert await pipeline_jobs.list_interrupted_runs() == {}


async def test_rows_written_since_this_process_started_are_excluded(db):
    db["pipe_new"] = _row("pipe_new", "running", updated_at=AFTER_START)
    assert await pipeline_jobs.list_interrupted_runs() == {}


async def test_runs_this_process_knows_are_excluded(db, monkeypatch):
    db["pipe_mine"] = _row("pipe_mine", "running")
    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", {"pipe_mine": {"status": "running"}})
    assert await pipeline_jobs.list_interrupted_runs() == {}


async def test_listed_runs_are_ones_resume_accepts(db, monkeypatch):
    """The point of the listing: resume_pipeline_run must not refuse what it returns."""
    db["pipe_abc123def456"] = _row("pipe_abc123def456", "running")
    monkeypatch.setattr(
        run_store, "load_run", AsyncMock(side_effect=lambda run_id: {k: db[run_id][k] for k in run_store._COLUMNS})
    )
    submit = AsyncMock(side_effect=lambda *_args, **kwargs: kwargs["run_id"])
    monkeypatch.setattr(pipeline_jobs, "submit_pipeline_run", submit)

    for run_id in await pipeline_jobs.list_interrupted_runs():
        assert await pipeline_jobs.resume_pipeline_run(run_id) == run_id
    assert [call.kwargs["run_id"] for call in submit.await_args_list] == ["pipe_abc123def456"]


async def test_no_database_yields_no_runs(monkeypatch):
    monkeypatch.setattr(run_store.settings, "database_url", "")
    assert await pipeline_jobs.list_interrupted_runs() == {}
