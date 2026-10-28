"""pipeline_jobs.list_interrupted_runs: used by sre_autopilot to find restart-cut-off runs (AP-83)."""

import asyncio

from pydantic_agents.core import pipeline_jobs


def _run(status, task=None):
    return {"project": "demo", "status": status, "_task": task}


def test_pending_or_running_run_with_no_live_task_is_interrupted(monkeypatch):
    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", {"pipe_a": _run("running"), "pipe_b": _run("pending")})
    found = pipeline_jobs.list_interrupted_runs()
    assert set(found) == {"pipe_a", "pipe_b"}


def test_waiting_on_cto_with_no_live_task_is_interrupted(monkeypatch):
    monkeypatch.setattr(
        pipeline_jobs, "PIPELINE_RUNS", {"pipe_a": _run(pipeline_jobs.STATUS_WAITING_CTO)}
    )
    assert set(pipeline_jobs.list_interrupted_runs()) == {"pipe_a"}


def test_finished_runs_are_excluded(monkeypatch):
    monkeypatch.setattr(
        pipeline_jobs,
        "PIPELINE_RUNS",
        {"pipe_done": _run("completed"), "pipe_failed": _run("failed"), "pipe_cancelled": _run("cancelled")},
    )
    assert pipeline_jobs.list_interrupted_runs() == {}


async def test_a_run_with_a_live_unfinished_task_is_excluded(monkeypatch):
    async def _never():
        await asyncio.sleep(3600)

    task = asyncio.create_task(_never())
    try:
        monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", {"pipe_live": _run("running", task=task)})
        assert pipeline_jobs.list_interrupted_runs() == {}
    finally:
        task.cancel()


def test_empty_registry_yields_no_runs(monkeypatch):
    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", {})
    assert pipeline_jobs.list_interrupted_runs() == {}
