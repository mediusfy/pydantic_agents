from types import SimpleNamespace

import pytest

pytest.importorskip("turso")

from pydantic_agents.core import existing_code_handler as handler  # noqa: E402
from pydantic_agents.core.adapters.stage_events import (  # noqa: E402
    NullStageEventRecorder,
    TursoStageEventRecorder,
    build_stage_event_recorder,
)
from pydantic_agents.core.models.core import Artifact  # noqa: E402


@pytest.fixture
def turso_backend(tmp_path, monkeypatch):
    monkeypatch.setattr("pydantic_agents.core.adapters.stage_events.settings.checkpoint_dir", str(tmp_path / "ckpt"))
    monkeypatch.setattr("pydantic_agents.core.adapters.stage_events.settings.state_backend", "turso")
    monkeypatch.setattr(handler, "_sync_memory_after_pipeline", lambda: None)

    async def fake_stage(stage_name, agent_name, prompt, span_attrs=None):  # noqa: ARG001
        return SimpleNamespace(output=Artifact(agent_name="se", files={"main.go": "package main"})), True

    monkeypatch.setattr(handler, "_run_stage", fake_stage)
    code = tmp_path / "proj"
    code.mkdir()
    return str(code)


def _gates(monkeypatch, qg_sequence):
    """Make the quality gate return successive (passed, score, feedback) values; review always approves."""
    calls = iter(qg_sequence)

    async def fake_qg(_artifacts):
        return next(calls)

    async def fake_review(_artifacts):
        return True, "low", []

    monkeypatch.setattr(handler, "_run_quality_gate", fake_qg)
    monkeypatch.setattr(handler, "_run_code_review", fake_review)


@pytest.mark.parametrize(
    "qg_sequence, expected, passed",
    [
        (
            [(True, 10, [])],
            [
                ("service_engineer", 0, "ok"),
                ("quality_gate", 0, "passed"),
                ("review", 0, "approved"),
                ("pipeline", 0, "passed"),
            ],
            True,
        ),
        (
            [(False, 90, ["too complex"]), (True, 10, [])],
            [
                ("service_engineer", 0, "ok"),
                ("quality_gate", 0, "failed"),
                ("review", 0, "approved"),
                ("service_engineer", 1, "ok"),
                ("quality_gate", 1, "passed"),
                ("review", 1, "approved"),
                ("pipeline", 1, "passed"),
            ],
            True,
        ),
        (
            [(False, 90, ["too complex"])] * (handler.MAX_RETRIES + 1),
            [
                event
                for attempt in range(handler.MAX_RETRIES + 1)
                for event in (
                    ("service_engineer", attempt, "ok"),
                    ("quality_gate", attempt, "failed"),
                    ("review", attempt, "approved"),
                )
            ]
            + [("pipeline", handler.MAX_RETRIES, "failed")],
            False,
        ),
    ],
)
async def test_pipeline_records_stage_events(turso_backend, monkeypatch, qg_sequence, expected, passed):
    _gates(monkeypatch, qg_sequence)

    result = await handler.run_existing_code_pipeline("change", turso_backend, run_tests=False)

    assert result.passed is passed
    events = TursoStageEventRecorder("proj").list_events(result.run_id)
    assert [(e.stage, e.attempt, e.verdict) for e in events] == expected
    failed_gate = [e for e in events if e.stage == "quality_gate" and e.verdict == "failed"]
    assert all(e.detail == "too complex" for e in failed_gate)


@pytest.mark.parametrize(
    "backend, expected_type",
    [("disk", NullStageEventRecorder), ("turso", TursoStageEventRecorder)],
)
def test_build_stage_event_recorder_selects_backend(tmp_path, monkeypatch, backend, expected_type):
    monkeypatch.setattr("pydantic_agents.core.adapters.stage_events.settings.checkpoint_dir", str(tmp_path))
    monkeypatch.setattr("pydantic_agents.core.adapters.stage_events.settings.state_backend", backend)
    assert isinstance(build_stage_event_recorder("proj"), expected_type)


def test_record_failure_never_raises(tmp_path):
    recorder = TursoStageEventRecorder("proj", db_path=str(tmp_path / "e.db"))
    recorder.close()
    recorder.record("run", "stage", 0, "ok", 1)  # closed connection: must be swallowed
