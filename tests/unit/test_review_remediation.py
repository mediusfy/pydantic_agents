"""Regression tests for the full-repository review of 2026-10-30 (see the CHANGELOG entries under Unreleased)."""

import asyncio
import inspect
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_agents.core import cto_gate, limits, pipeline_jobs, platform_metrics, sre_remediation
from pydantic_agents.core.models.core import Artifact, PipelineState, UserStory
from pydantic_agents.core.pipeline.config import STAGES, PipelineConfig
from pydantic_agents.core.session_recorder import SessionRecorder
from pydantic_agents.core.settings import settings
from pydantic_agents.core.utils import kill_and_reap, rotate_log

# --- CTO gate: a marker answers one gate -------------------------------------------------------------------------


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.delenv("ORCHESTRATION_CTO_OVERRIDE", raising=False)
    poll = cto_gate._poll_for_file_approval
    monkeypatch.setattr(
        cto_gate, "_poll_for_file_approval", lambda path, timeout_seconds: poll(path, 0.01, timeout_seconds)
    )
    return tmp_path


async def test_an_approval_marker_is_consumed_by_the_gate_it_approved(out_dir):
    cto_gate.write_approval("svc", True)

    first = await cto_gate.apply(PipelineState(run_id="a", stage="advisory"), project="svc", timeout_seconds=0.05)
    second = await cto_gate.apply(PipelineState(run_id="b", stage="advisory"), project="svc", timeout_seconds=0.05)

    assert first is True
    assert second is False  # the next run's CTO said no and nobody approved it: it must not inherit the old approval
    assert not (out_dir / "svc" / ".cto_approved").exists()


async def test_a_rejection_marker_stops_one_run_not_the_project_for_good(out_dir):
    cto_gate.write_approval("svc", False)

    first = await cto_gate.apply(PipelineState(run_id="a", stage="advisory"), project="svc", timeout_seconds=0.05)
    cto_gate.write_approval("svc", True)
    second = await cto_gate.apply(PipelineState(run_id="b", stage="advisory"), project="svc", timeout_seconds=0.05)

    assert (first, second) == (False, True)


@pytest.mark.parametrize(
    ("value", "bypasses"),
    [("true", True), ("1", True), ("YES", True), ("false", False), ("0", False), ("", False), ("no", False)],
)
async def test_the_override_variable_needs_a_true_value(out_dir, monkeypatch, value, bypasses):
    monkeypatch.setenv("ORCHESTRATION_CTO_OVERRIDE", value)

    result = await cto_gate.apply(PipelineState(run_id="a", stage="advisory"), project="svc", timeout_seconds=0.05)

    assert result is bypasses


async def test_each_project_has_its_own_marker(out_dir):
    cto_gate.write_approval("project-a", True)

    other = await cto_gate.apply(PipelineState(run_id="b", stage="advisory"), project="project-b", timeout_seconds=0.05)

    assert other is False
    assert (out_dir / "project-a" / ".cto_approved").exists()  # project-b's gate neither used nor consumed it


async def test_an_operator_approving_while_the_gate_polls_is_honoured_once(out_dir):
    async def approve_soon():
        await asyncio.sleep(0.05)
        (out_dir / "svc").mkdir(exist_ok=True)
        (out_dir / "svc" / ".cto_approved").write_text("")  # `touch`: an empty marker counts as approval

    approver = asyncio.create_task(approve_soon())
    first = await cto_gate.apply(PipelineState(run_id="a", stage="advisory"), project="svc", timeout_seconds=2)
    await approver
    second = await cto_gate.apply(PipelineState(run_id="b", stage="advisory"), project="svc", timeout_seconds=0.05)

    assert (first, second) == (True, False)


# --- blueprints ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("blueprint", "registered"),
    [
        ("new_feature", {"ba", "advisory", "implementation", "quality_gate"}),
        ("quick_backend", {"implementation", "quality_gate"}),
        ("docs_update", {"implementation", "quality_gate"}),
        ("security_review", {"quality_gate"}),
    ],
)
def test_a_legacy_blueprint_registers_only_its_own_stages(blueprint, registered):
    from pydantic_agents.core.pipeline.factory import build_default_pipeline

    pipeline = build_default_pipeline(PipelineConfig(blueprint=blueprint))

    assert {str(s.value) for s in pipeline.stages} == registered


def test_an_unknown_blueprint_fails_closed_instead_of_running_everything():
    from pydantic_agents.core.pipeline.factory import build_default_pipeline

    config = PipelineConfig(blueprint="new_feature")
    config.blueprint = "typo_blueprint"

    with pytest.raises(ValueError, match="typo_blueprint"):
        build_default_pipeline(config)


async def test_a_blueprint_without_ba_implements_the_requirement_as_one_story(monkeypatch):
    from pydantic_agents.core.pipeline.factory import build_default_pipeline

    pipeline = build_default_pipeline(PipelineConfig(blueprint="quick_backend", requirement="add a /health route"))
    seen = {}

    async def implementation(state, _config):
        seen["stories"] = list(state.user_stories)
        return state

    pipeline.stages[STAGES.IMPLEMENTATION].run = implementation
    pipeline.stages[STAGES.QUALITY_GATE].run = AsyncMock(side_effect=lambda state, _c: state)
    monkeypatch.setattr(pipeline.artifact_writer, "write_artifacts", lambda _a: None)

    await pipeline.execute()

    assert [s.gherkin for s in seen["stories"]] == ["add a /health route"]


async def test_a_gate_only_blueprint_actually_runs_the_gate(monkeypatch):
    from pydantic_agents.core.pipeline.factory import build_default_pipeline

    pipeline = build_default_pipeline(PipelineConfig(blueprint="security_review", requirement="audit"))
    gate = AsyncMock(side_effect=lambda state, _c: state)
    pipeline.stages[STAGES.QUALITY_GATE].run = gate
    monkeypatch.setattr(pipeline.artifact_writer, "write_artifacts", lambda _a: None)

    await pipeline.execute()

    gate.assert_awaited_once()


# --- crew and group chat status -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("crew", "errors", "cto_approved", "failure", "approved"),
    [
        ("ResearchCrew", [], False, False, True),
        ("ResearchCrew", ["crewai x: bad"], False, False, False),
        ("AuditCrew", [], False, False, True),
        ("AuditCrew", ["findings rejected"], False, False, False),
        ("ArchitectureReviewCrew", [], True, False, True),
        ("ArchitectureReviewCrew", [], False, False, False),
        ("ResearchCrew", [], False, True, False),
        ("SdlcCrew", [], True, False, False),  # decided by its review verdict, not here
    ],
)
def test_crews_without_a_verdict_are_approved_when_they_finish_cleanly(crew, errors, cto_approved, failure, approved):
    from pydantic_agents.core.models.core import FailureReport
    from pydantic_agents.core.pipeline.crewai.approval import finalize_approval as _finalize_approval

    state = PipelineState(run_id="r", stage="implementation", errors=errors, cto_approved=cto_approved)
    if failure:
        state.failure_report = FailureReport(source="crewai", error_type="runtime_error", trace="x")

    _finalize_approval(state, crew)

    assert bool(state.approved) is approved


# --- session recorder ----------------------------------------------------------------------------------------------


def _read(path):
    import json

    return [json.loads(line) for line in path.read_text().splitlines()]


async def test_a_stage_that_catches_its_own_failure_is_recorded_as_failed(tmp_path):
    rec = SessionRecorder(str(tmp_path / "t.jsonl"))

    async with rec.record_span("r", "ba", "ba", "p") as span:
        span.abort_run("boom")  # what the stage nodes do in their except branch

    (record,) = _read(tmp_path / "t.jsonl")
    assert (record["success"], record["error"]) == (False, "boom")


async def test_a_cancelled_run_is_recorded_as_failed(tmp_path):
    rec = SessionRecorder(str(tmp_path / "t.jsonl"))

    async def run():
        async with rec.record_span("r", "backend", "implementation", "p"):
            await asyncio.sleep(10)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (record,) = _read(tmp_path / "t.jsonl")
    assert record["success"] is False
    assert record["error"]


async def test_a_clean_span_is_still_recorded_once_as_a_success(tmp_path):
    rec = SessionRecorder(str(tmp_path / "t.jsonl"))

    async with rec.record_span("r", "ba", "ba", "p") as span:
        span.set_output("done")

    (record,) = _read(tmp_path / "t.jsonl")
    assert (record["success"], record["error"]) == (True, None)


# --- worker threads keep their permit until they end ---------------------------------------------------------------


async def test_a_timed_out_thread_keeps_its_llm_permit_until_it_finishes(monkeypatch):
    monkeypatch.setattr(settings, "max_concurrent_llm_calls", 1)
    limits._semaphores.clear()
    import threading

    release = threading.Event()

    with pytest.raises(TimeoutError):
        await limits.run_blocking_with_permit(lambda: release.wait(5), timeout_seconds=0.05)

    assert limits.llm_semaphore().locked()  # the thread is still calling models: the permit is not free

    release.set()
    for _ in range(100):
        if not limits.llm_semaphore().locked():
            break
        await asyncio.sleep(0.01)
    assert not limits.llm_semaphore().locked()


async def test_a_blocking_call_returns_its_value_and_frees_the_permit(monkeypatch):
    monkeypatch.setattr(settings, "max_concurrent_llm_calls", 1)
    limits._semaphores.clear()

    assert await limits.run_blocking_with_permit(lambda: 42, timeout_seconds=1) == 42
    assert not limits.llm_semaphore().locked()


# --- autopilot ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attempts", "auto"),
    [(0, True), (sre_remediation.MAX_AUTO_RESUMES - 1, True), (sre_remediation.MAX_AUTO_RESUMES, False), (9, False)],
)
def test_a_run_that_keeps_being_interrupted_stops_being_resumed(attempts, auto):
    plan = sre_remediation.plan_for_interrupted_run("pipe_x", {"project": "p", "resume_attempts": attempts})

    assert plan.auto_remediable is auto
    assert (plan.run_id is not None) is auto
    assert plan.evidence["run_id"] == "pipe_x"


# --- quality gate ---------------------------------------------------------------------------------------------------


async def test_a_run_with_no_artifacts_is_rejected_not_approved(monkeypatch):
    from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

    provider = AsyncMock()
    state = PipelineState(run_id="r", stage="implementation")

    out = await QualityGateStage(agent_provider=provider).run(state, PipelineConfig(project="p", requirement="r"))

    assert out.approved is False
    assert out.failure_report.error_type == "empty"
    provider.run_agent.assert_not_awaited()  # no model is asked to approve nothing


async def test_the_sre_artifact_is_replaced_on_a_retry_not_stacked():
    from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

    stage = QualityGateStage(agent_provider=AsyncMock())
    stage._run_agent = AsyncMock(return_value=Artifact(agent_name="sre", files={"chart.yaml": "x"}))
    stage._retrieved = AsyncMock(return_value="")
    state = PipelineState(run_id="r", stage="implementation", user_stories=[UserStory(id="U", title="t", gherkin="g")])
    artifacts = [Artifact(agent_name="backend", files={"main.go": "package main"})]
    config = PipelineConfig(project="p", requirement="r")

    await stage._run_sre(state, config, artifacts)
    await stage._run_sre(state, config, artifacts)

    assert [a.agent_name for a in artifacts] == ["backend", "infrastructure"]


# --- subprocesses and logs -----------------------------------------------------------------------------------------


async def test_killing_a_timed_out_command_also_kills_its_children(tmp_path):
    marker = tmp_path / "child.pid"
    proc = await asyncio.create_subprocess_exec(
        "sh", "-c", f"sleep 60 & echo $! > {marker}; wait", start_new_session=True
    )
    for _ in range(100):
        if marker.exists() and marker.read_text().strip():
            break
        await asyncio.sleep(0.02)
    child = int(marker.read_text().strip())

    await kill_and_reap(proc)

    assert proc.returncode is not None
    for _ in range(100):  # the child is gone, not orphaned
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.02)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


async def test_kill_and_reap_never_calls_killpg_on_a_magicmock_or_pid_1(monkeypatch):
    killpg = MagicMock()
    monkeypatch.setattr(os, "killpg", killpg)
    mock_proc = MagicMock(kill=MagicMock(), wait=AsyncMock())  # mock_proc.pid is a MagicMock -> __index__() == 1
    await kill_and_reap(mock_proc)
    killpg.assert_not_called()
    mock_proc.kill.assert_called_once()
    mock_proc.wait.assert_awaited_once()



def test_rotating_a_log_keeps_only_the_newest_archives(tmp_path, monkeypatch):
    log = tmp_path / "telemetry.jsonl"
    for stamp in range(1, 8):
        (tmp_path / f"telemetry.jsonl.{stamp}.archive").write_text("old")
    log.write_text("x" * 100)

    assert rotate_log(str(log), max_bytes=10, keep=3) is True

    archives = sorted(p.name for p in tmp_path.glob("telemetry.jsonl.*.archive"))
    assert len(archives) == 3
    assert not log.exists()
    assert "telemetry.jsonl.1.archive" not in archives


def test_a_small_log_is_not_rotated(tmp_path):
    log = tmp_path / "t.jsonl"
    log.write_text("x")

    assert rotate_log(str(log), max_bytes=10) is False
    assert log.exists()


# --- run bookkeeping -----------------------------------------------------------------------------------------------


def test_a_run_without_a_start_time_is_evicted_first(monkeypatch):
    done = MagicMock()
    done.done.return_value = True
    runs = {"pipe_a": {"started_at": None, "_task": done}, "pipe_b": {"started_at": "2026-10-01", "_task": done}}
    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", runs)
    monkeypatch.setattr(pipeline_jobs, "MAX_FINISHED_RUNS_IN_MEMORY", 1)

    pipeline_jobs._evict_finished_runs()

    assert list(runs) == ["pipe_b"]


def test_finished_runs_beyond_the_cap_are_dropped_oldest_first(monkeypatch):
    done = MagicMock()
    done.done.return_value = True
    live = MagicMock()
    live.done.return_value = False
    runs = {f"pipe_{i:012d}": {"started_at": f"2026-10-{i + 1:02d}", "_task": done} for i in range(5)}
    runs["pipe_live"] = {"started_at": "2020-01-01", "_task": live}
    monkeypatch.setattr(pipeline_jobs, "PIPELINE_RUNS", runs)
    monkeypatch.setattr(pipeline_jobs, "MAX_FINISHED_RUNS_IN_MEMORY", 2)

    pipeline_jobs._evict_finished_runs()

    assert sorted(runs) == ["pipe_000000000003", "pipe_000000000004", "pipe_live"]


# --- metrics -------------------------------------------------------------------------------------------------------


def test_token_counts_and_verdicts_come_from_the_run_result():
    result = MagicMock()
    result.usage.return_value = MagicMock(input_tokens=120, output_tokens=30)
    result.output = MagicMock(passed=False, approved=None)

    assert platform_metrics.token_counts(result) == (120, 30)
    assert platform_metrics.verdict_of(result) == "failed"
    assert platform_metrics.token_counts(object()) == (None, None)


async def test_the_provider_records_one_metric_row_per_agent_run(monkeypatch):
    from pydantic_agents.core.adapters import agents as provider_module

    record = AsyncMock()
    monkeypatch.setattr(provider_module.platform_metrics, "record_call", record)
    run_result = MagicMock()
    run_result.usage.return_value = MagicMock(input_tokens=5, output_tokens=2)
    run_result.output = MagicMock(approved=True, passed=None)
    agent = MagicMock()
    agent.run = AsyncMock(return_value=run_result)

    await provider_module.PydanticAIAgentProvider().run_agent(asyncio.Semaphore(1), "review", agent, "p")

    record.assert_awaited_once()
    assert record.await_args.args == ("review",)
    kwargs = record.await_args.kwargs
    assert (kwargs["success"], kwargs["input_tokens"], kwargs["output_tokens"], kwargs["verdict"]) == (
        True,
        5,
        2,
        "passed",
    )


async def test_a_failed_agent_run_is_recorded_as_a_failure(monkeypatch):
    from pydantic_agents.core.adapters import agents as provider_module

    record = AsyncMock()
    monkeypatch.setattr(provider_module.platform_metrics, "record_call", record)
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=ValueError("bad output"))

    provider = provider_module.PydanticAIAgentProvider()
    semaphore = asyncio.Semaphore(1)
    with pytest.raises(ValueError, match="bad output"):
        await provider.run_agent(semaphore, "review", agent, "p")

    assert record.await_args.kwargs["success"] is False


# --- MCP tools that touch the vector store are async ---------------------------------------------------------------


@pytest.mark.parametrize(
    "tool", ["search_adrs", "save_adr", "query_context", "seed_project_context", "sync_memory", "memory_bridge_status"]
)
async def test_blocking_mcp_tools_do_not_run_on_the_event_loop(tool):
    from pydantic_agents import mcp_server

    registered = mcp_server.mcp._tool_manager.get_tool(tool)

    assert registered is not None
    assert inspect.iscoroutinefunction(registered.fn)
    assert not inspect.iscoroutinefunction(getattr(mcp_server, tool))  # still a plain function for scripts and tests


# --- existing-code pipeline ----------------------------------------------------------------------------------------


async def test_a_retry_that_changes_nothing_is_not_a_pass(monkeypatch):
    from pydantic_agents.core import existing_code_handler as ech

    result = ech.ExistingCodeResult(run_id="r", errors=["review: missing error handling"])
    empty = MagicMock(files={})
    monkeypatch.setattr(ech, "_run_engineer", AsyncMock(return_value=(empty, 0.0)))
    options = ech._PipelineOptions("change", "", True, True, True)

    done = await ech._run_attempt(result, MagicMock(), 1, options)

    assert done is False
    assert result.passed is False
    assert any("no changed files on attempt 1" in e for e in result.errors)
