"""Durable pipeline runs: the pipeline_runs store, interrupted detection, resume and the CTO approval tool."""

import asyncio
from datetime import UTC, datetime

import pytest

from pydantic_agents.core import pipeline_jobs, run_store
from pydantic_agents.core.cto_gate import write_approval
from pydantic_agents.core.settings import settings
from pydantic_agents.core.utils import PathTraversalError


class _Conn:
    def __init__(self, db):
        self.db = db

    async def execute(self, sql, *args):
        if "INSERT INTO pipeline_runs" in sql:
            run_id, status, project, requirement, result, error, started, completed, elapsed = args
            self.db[run_id] = {
                "status": status,
                "project": project,
                "requirement": requirement,
                "result": result,
                "error": error,
                "started_at": started,
                "completed_at": completed,
                "elapsed_seconds": elapsed,
            }

    async def fetchrow(self, _sql, run_id):
        return self.db.get(run_id)


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
    monkeypatch.setattr(run_store, "_table_ready", False)

    async def get_pool():
        return p

    monkeypatch.setattr(run_store.vector_db, "_get_pool", get_pool)
    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", {})
    return p


def _run(status, **extra):
    return {
        "status": status,
        "project": "svc",
        "requirement": "build it",
        "result": None,
        "error": None,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "elapsed_seconds": None,
        **extra,
    }


async def test_run_survives_a_restart_and_is_marked_interrupted(pool):
    await run_store.save_run("pipe_1", _run("running"))
    # a new process: nothing in PIPELINE_RUNS, only the database
    got = await pipeline_jobs.get_run("pipe_1")
    assert got["status"] == "interrupted" and "restarted" in got["error"]


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
async def test_finished_runs_keep_their_status(pool, status):
    await run_store.save_run("pipe_2", _run(status))
    assert (await pipeline_jobs.get_run("pipe_2"))["status"] == status


async def test_unknown_run_and_no_database(monkeypatch):
    monkeypatch.setattr(settings, "database_url", None)
    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", {})
    assert await pipeline_jobs.get_run("nope") is None
    await run_store.save_run("x", _run("running"))  # no-op, must not raise


async def test_unreachable_database_is_best_effort(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql://stub/db")

    async def boom():
        raise ConnectionError("db down")

    monkeypatch.setattr(run_store.vector_db, "_get_pool", boom)
    await run_store.save_run("x", _run("running"))
    assert await run_store.load_run("x") is None


@pytest.mark.parametrize(
    ("status", "resumes"),
    [("interrupted", True), ("running", True), ("failed", True), ("cancelled", True), ("completed", False)],
)
async def test_resume_only_for_unfinished_runs(pool, monkeypatch, status, resumes):
    seen = {}

    async def fake_submit(project, requirement, *, run_id=None, resume=False):
        seen.update(project=project, requirement=requirement, run_id=run_id, resume=resume)
        return run_id

    monkeypatch.setattr(pipeline_jobs, "submit_pipeline_run", fake_submit)
    await run_store.save_run("pipe_3", _run(status))
    out = await pipeline_jobs.resume_pipeline_run("pipe_3")
    if resumes:
        assert out == "pipe_3"
        assert seen == {"project": "svc", "requirement": "build it", "run_id": "pipe_3", "resume": True}
    else:
        assert out is None and not seen


async def test_resume_unknown_run(pool):
    assert await pipeline_jobs.resume_pipeline_run("missing") is None


async def test_submit_persists_each_transition(pool, monkeypatch):
    class _P:
        async def execute(self):
            return None

    monkeypatch.setattr(pipeline_jobs, "build_default_pipeline", lambda _c: _P())
    run_id = await pipeline_jobs.submit_pipeline_run("svc", "req")
    await pipeline_jobs.PIPELINE_RUNS[run_id]["_task"]
    await asyncio.sleep(0)
    assert pool.db[run_id]["status"] == "completed" and pool.db[run_id]["completed_at"] is not None


@pytest.mark.parametrize(
    ("project", "approved", "content", "raises"),
    [("svc", True, "true\n", False), ("svc", False, "false\n", False), ("../escape", True, None, True)],
)
def test_write_approval(monkeypatch, tmp_path, project, approved, content, raises):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    if raises:
        with pytest.raises(PathTraversalError):
            write_approval(project, approved)
        return
    assert write_approval(project, approved).read_text() == content
