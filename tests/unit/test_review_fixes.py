"""Regression tests for the findings of the 2026-10-27 architecture review.

One test class per finding, each pinned at the smallest unit that showed the bug: the single-slot session
recorder, the bypassed run guards, delegation budgets, the review verdict, context scoping, the LangGraph thread
id, the plan data flow, project-name validation and the Langfuse OTLP export.
"""

import asyncio
import json
import shutil
from contextvars import copy_context
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core import context_engine, session_recorder
from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
from pydantic_agents.core.exceptions import AgentNotFoundError
from pydantic_agents.core.limits import run_id_var
from pydantic_agents.core.models.core import ReviewResult, RiskLevel
from pydantic_agents.core.registry import AgentRegistry
from pydantic_agents.core.settings import settings

# --- SessionRecorder: one record per span, also under concurrency ----------------------------------------------


class TestSessionRecorderConcurrency:
    @pytest.fixture
    def recorder(self, tmp_path):
        return session_recorder.SessionRecorder(filepath=str(tmp_path / "trace.jsonl"))

    async def test_concurrent_spans_each_write_their_own_record(self, recorder, tmp_path):
        async def span(agent: str, delay: float):
            async with recorder.record_span(run_id="r", agent_name=agent, pipeline_stage="advisory", prompt="p") as rec:
                await asyncio.sleep(delay)
                rec.set_output({"agent": agent})

        await asyncio.gather(span("cto", 0.02), span("ux", 0.01))

        records = session_recorder.extract_traces(str(tmp_path / "trace.jsonl"))
        by_agent = {r["agent_name"]: r for r in records}
        assert set(by_agent) == {"cto", "ux"}
        assert all(r["success"] for r in records)
        assert by_agent["cto"]["output_json"] == {"agent": "cto"}
        assert by_agent["ux"]["output_json"] == {"agent": "ux"}

    async def test_a_raising_span_writes_a_failure_and_the_other_span_is_untouched(self, recorder, tmp_path):
        async def good():
            async with recorder.record_span(run_id="r", agent_name="good", pipeline_stage="s", prompt="p") as rec:
                await asyncio.sleep(0.02)
                rec.set_output("fine")

        async def bad():
            async with recorder.record_span(run_id="r", agent_name="bad", pipeline_stage="s", prompt="p"):
                await asyncio.sleep(0.01)
                raise RuntimeError("model down")

        results = await asyncio.gather(good(), bad(), return_exceptions=True)
        assert isinstance(results[1], RuntimeError)

        by_agent = {r["agent_name"]: r for r in session_recorder.extract_traces(str(tmp_path / "trace.jsonl"))}
        assert by_agent["good"]["success"] is True
        assert by_agent["good"]["output"] == "fine"
        assert by_agent["bad"]["success"] is False
        assert by_agent["bad"]["error"] == "model down"

    async def test_tool_calls_are_recorded_on_the_innermost_span_of_the_task(self, recorder, tmp_path):
        async with recorder.record_span(run_id="r", agent_name="backend", pipeline_stage="s", prompt="p"):
            recorder.record_tool_call("run_go_build", {"files": 2}, "SUCCESS", 1.5)
        recorder.record_tool_call("outside", {}, "ignored", 0.1)  # no span: dropped

        [record] = session_recorder.extract_traces(str(tmp_path / "trace.jsonl"))
        assert [t["tool_name"] for t in record["tool_calls"]] == ["run_go_build"]


# --- registry.run: the one guarded way to run an agent --------------------------------------------------------


def _agent(output="ok"):
    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output=output))
    return agent


class TestRegistryRun:
    async def test_run_goes_through_the_provider_with_deps_from_the_factory(self):
        registry = AgentRegistry()
        agent = _agent()
        registry.register("weather", agent, ["weather"], lambda: {"city": "Oslo"})

        result = await registry.run("weather", "forecast?")

        assert result.output == "ok"
        kwargs = agent.run.await_args.kwargs
        assert kwargs["deps"] == {"city": "Oslo"}
        assert kwargs["usage_limits"] is not None
        assert kwargs["model_settings"]["max_tokens"] == settings.agent_max_output_tokens

    async def test_explicit_deps_win_over_the_factory(self):
        registry = AgentRegistry()
        agent = _agent()
        registry.register("weather", agent, ["weather"], lambda: {"city": "Oslo"})

        await registry.run("weather", "forecast?", deps={"city": "Bergen"})

        assert agent.run.await_args.kwargs["deps"] == {"city": "Bergen"}

    async def test_an_unknown_agent_raises_before_any_call(self):
        empty = AgentRegistry()
        with pytest.raises(AgentNotFoundError):
            await empty.run("nope", "x")

    async def test_run_honours_the_timeout(self, monkeypatch):
        registry = AgentRegistry()
        agent = MagicMock()

        async def slow(*_a, **_k):
            await asyncio.sleep(1)

        agent.run = slow
        registry.register("slow", agent, ["slow"])
        with pytest.raises(TimeoutError):
            await registry.run("slow", "x", timeout_seconds=0.01)

    async def test_a_shared_usage_object_is_passed_through_to_the_callee(self):
        registry = AgentRegistry()
        agent = _agent()
        registry.register("backend", agent, ["backend"])
        usage = MagicMock(name="usage")

        await registry.query("backend", "build", usage=usage)

        assert agent.run.await_args.kwargs["usage"] is usage

    async def test_the_provider_opens_an_agent_run_span_tagged_with_the_session(self, monkeypatch):
        spans = []

        class Span:
            def __init__(self, name, **attrs):
                spans.append((name, attrs))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("pydantic_agents.core.adapters.agents.logfire.span", Span)
        token = run_id_var.set("pipe_x")
        try:
            await PydanticAIAgentProvider().run_agent(asyncio.Semaphore(1), "ba", _agent(), "hi")
        finally:
            run_id_var.reset(token)
        [(name, attrs)] = spans
        assert name == "agent.run"
        assert attrs["langfuse.session.id"] == "pipe_x"
        assert attrs["agent_name"] == "ba"


# --- delegation shares the caller's budget --------------------------------------------------------------------


async def test_delegate_to_agent_passes_the_caller_usage(monkeypatch):
    from pydantic_agents.core.tools import a2a_tools

    seen = {}

    async def fake_query(**kwargs):
        seen.update(kwargs)
        return "done"

    monkeypatch.setattr(a2a_tools, "instrumented_query", fake_query)
    usage = MagicMock(name="usage")
    ctx = MagicMock(usage=usage)

    assert await a2a_tools.delegate_to_agent(ctx, "backend", "do it", "ctx") == "done"
    assert seen["usage"] is usage
    assert seen["delegation_depth"] == 1


# --- ReviewResult cannot approve what it rates as risky -------------------------------------------------------


@pytest.mark.parametrize(
    ("approved", "risk", "score", "want"),
    [
        (True, RiskLevel.LOW, 85, True),
        (True, RiskLevel.MEDIUM, 70, True),
        (True, RiskLevel.HIGH, 90, False),
        (True, RiskLevel.CRITICAL, 100, False),
        (True, RiskLevel.LOW, 69, False),
        (False, RiskLevel.LOW, 100, False),
    ],
)
def test_review_verdict_is_consistent_with_risk_and_score(approved, risk, score, want):
    result = ReviewResult(approved=approved, feedback=["x"], security_score=score, risk_level=risk)
    assert result.approved is want
    if approved and not want:
        assert any("Approval withdrawn" in f for f in result.feedback)


# --- context retrieval: async, project-scoped, no store notices in prompts -----------------------------------


class TestRetrieveContext:
    async def test_the_project_and_a_capped_query_reach_the_store(self, monkeypatch):
        calls = []

        async def search(domain, query, n_results=3, project=None, filters=None):
            calls.append((domain, len(query), n_results, project))
            return [f"{domain} snippet"]

        monkeypatch.setattr(context_engine.vector_db, "async_search", search)

        text = await context_engine.retrieve_context("database", "x" * 5_000, top_k=2, project="rufkis")

        assert calls == [("adrs", 1_000, 2, "rufkis"), ("architecture", 1_000, 2, "rufkis")]
        assert "adrs snippet" in text
        assert "architecture snippet" in text

    async def test_the_store_s_unavailability_notice_never_becomes_context(self, monkeypatch):
        async def search(*_a, **_k):
            return ["Vector DB not available — install heavy dependencies", "", "real"]

        monkeypatch.setattr(context_engine.vector_db, "async_search", search)

        assert (await context_engine.retrieve_context("cto", "q")).strip() == "real"

    async def test_a_failing_domain_is_skipped(self, monkeypatch):
        async def search(domain, *_a, **_k):
            if domain == "adrs":
                raise ConnectionError("down")
            return ["ok"]

        monkeypatch.setattr(context_engine.vector_db, "async_search", search)

        assert "ok" in await context_engine.retrieve_context("cto", "q")


# --- LangGraph: the run id is the thread id, so a checkpointer accepts the run --------------------------------


async def test_the_graph_wrapper_passes_the_run_id_as_thread_id():
    from pydantic_agents.core.pipeline.config import PipelineConfig
    from pydantic_agents.core.pipeline.factory import _GraphPipelineWrapper

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=lambda state, config=None: state)  # noqa: ARG005
    wrapper = _GraphPipelineWrapper(graph, PipelineConfig(blueprint="langgraph_full", framework="langgraph"))

    token = run_id_var.set("pipe_thread")
    try:
        state = await wrapper.execute()
    finally:
        run_id_var.reset(token)

    assert state.run_id == "pipe_thread"
    assert graph.ainvoke.await_args.kwargs["config"] == {"configurable": {"thread_id": "pipe_thread"}}


# --- plan_and_execute: dependencies carry results, failures skip dependants ----------------------------------


async def test_plan_tasks_receive_their_dependencies_results_and_failed_dependencies_skip(monkeypatch):
    from pydantic_agents import mcp_server
    from pydantic_agents.core import registry as registry_module
    from pydantic_agents.core.planning import Plan, Task

    plan = Plan(
        goal="g",
        tasks=[
            Task(id="t1", capability="ba", prompt="stories"),
            Task(id="t2", capability="backend", prompt="build", dependencies=["t1"]),
            Task(id="t3", capability="review", prompt="review", dependencies=["t2"]),
        ],
    )
    monkeypatch.setattr(mcp_server.registry, "run", AsyncMock(return_value=MagicMock(output=plan)))
    prompts = {}

    async def fake_query(capability, prompt, **_kwargs):
        prompts[capability] = prompt
        if capability == "backend":
            raise RuntimeError("compile error")
        return f"{capability} result"

    monkeypatch.setattr(registry_module, "instrumented_query", fake_query)

    summary = await mcp_server.plan_and_execute("ship it")

    assert "Results of the tasks this depends on:\n[t1]\nba result" in prompts["backend"]
    assert "review" not in prompts  # its dependency failed, so it never ran
    assert "✗ t2" in summary
    assert "✗ t3" in summary
    assert "✓ t1" in summary
    assert "skipped, dependency failed: t2" in summary


# --- project names are plain names ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ok"),
    [("svc", True), ("my.service-1_x", True), ("../etc", False), ("a/b", False), ("", False), (".hidden", False)],
)
def test_project_names_are_validated_at_the_mcp_boundary_and_in_the_config(name, ok):
    from pydantic_agents.core.pipeline.config import PipelineConfig
    from pydantic_agents.mcp_server import _check_project_name

    assert (_check_project_name(name) is None) is ok
    if ok:
        PipelineConfig(project=name)
    else:
        with pytest.raises(ValueError):
            PipelineConfig(project=name)


async def test_run_orchestration_pipeline_rejects_a_bad_project_name():
    from pydantic_agents.mcp_server import run_orchestration_pipeline

    assert (await run_orchestration_pipeline("../escape", "req")).startswith("ERROR: Invalid project_name")
    assert (await run_orchestration_pipeline("svc", "req", blueprint="nope")).startswith("ERROR: Unknown blueprint")


# --- Checkpoint files ------------------------------------------------------------------------------------------


class TestCheckpointFiles:
    LEGACY_ID = "8d4c0d6e-1d6a-4b5e-9d1a-2f3e4c5b6a70"

    def test_legacy_single_underscore_checkpoints_are_migrated_and_resumable(self, tmp_path, monkeypatch, caplog):
        """``load`` builds "{run_id}__{stage}.json"; a run checkpointed by an older version is renamed on open."""
        from pydantic_agents.core.adapters.state_repository import LocalDiskStateRepository
        from pydantic_agents.core.models.core import PipelineState

        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / f"{self.LEGACY_ID}_ba.json").write_text(
            PipelineState(run_id=self.LEGACY_ID, stage="ba").model_dump_json()
        )
        (project_dir / f"{self.LEGACY_ID}_advisory_stage.json").write_text(
            PipelineState(run_id=self.LEGACY_ID, stage="advisory").model_dump_json()
        )
        (project_dir / "notes_for_humans.json").write_text("{}")  # not a UUID prefix: left alone, reported

        repo = LocalDiskStateRepository("proj")
        assert (project_dir / f"{self.LEGACY_ID}_ba.json").exists()  # construction does no I/O beyond mkdir
        with caplog.at_level("INFO", logger="pydantic_agents.core.adapters.state_repository"):
            assert repo.find_latest_run_id() == self.LEGACY_ID

        assert sorted(f.name for f in project_dir.iterdir()) == [
            f"{self.LEGACY_ID}__advisory_stage.json",
            f"{self.LEGACY_ID}__ba.json",
            "notes_for_humans.json",
        ]
        assert "Migrated 2 legacy checkpoint file(s)" in caplog.text
        assert "1 .json file(s) in" in caplog.text
        assert "notes_for_humans.json" in caplog.text
        assert repo.load(self.LEGACY_ID, "advisory_stage") is not None  # underscores in the stage round-trip
        assert repo.exists(self.LEGACY_ID, "ba")

    def test_a_second_instance_sees_the_rename_without_redoing_it(self, tmp_path, monkeypatch, caplog):
        from pydantic_agents.core.adapters.state_repository import LocalDiskStateRepository

        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / f"{self.LEGACY_ID}_ba.json").write_text('{"run_id": "x", "stage": "ba"}')
        first = LocalDiskStateRepository("proj")
        assert first.find_latest_run_id() == self.LEGACY_ID

        second = LocalDiskStateRepository("proj")
        with caplog.at_level("INFO", logger="pydantic_agents.core.adapters.state_repository"):
            assert second.find_latest_run_id() == self.LEGACY_ID
            assert second.exists(self.LEGACY_ID, "ba")

        assert "Migrated" not in caplog.text
        assert sorted(f.name for f in project_dir.iterdir()) == [f"{self.LEGACY_ID}__ba.json"]

    def test_migration_failure_is_warned_once_and_retried(self, tmp_path, monkeypatch, caplog):
        from pydantic_agents.core.adapters import state_repository
        from pydantic_agents.core.models.core import PipelineState

        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        repo = state_repository.LocalDiskStateRepository("proj")
        repo.save("run-new", "ba", PipelineState(run_id="run-new", stage="ba"))
        legacy = tmp_path / "proj" / f"{self.LEGACY_ID}_ba.json"
        legacy.write_text(PipelineState(run_id=self.LEGACY_ID, stage="ba").model_dump_json())

        with (
            patch.object(state_repository.os, "replace", side_effect=PermissionError("read-only")),
            caplog.at_level("WARNING", logger="pydantic_agents.core.adapters.state_repository"),
        ):
            assert repo.find_latest_run_id() == "run-new"  # reads proceed on the files that are there
            assert repo.load("run-new", "ba") is not None  # a second read does not warn again
        assert caplog.text.count("Could not migrate legacy checkpoint files") == 1
        assert legacy.exists()

        # The failure was transient: the next read migrates (the legacy file is the newest) and loads the run.
        assert repo.find_latest_run_id() == self.LEGACY_ID
        assert not legacy.exists()
        assert repo.load(self.LEGACY_ID, "ba") is not None

    def test_a_checkpoint_that_is_not_a_pipeline_state_is_treated_as_corrupt(self, tmp_path, monkeypatch, caplog):
        from pydantic_agents.core.adapters.state_repository import LocalDiskStateRepository

        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        repo = LocalDiskStateRepository("proj")
        (tmp_path / "proj" / "run-x__ba.json").write_text("{}")  # valid JSON, not a PipelineState

        with caplog.at_level("WARNING", logger="pydantic_agents.core.adapters.state_repository"):
            assert repo.load("run-x", "ba") is None

        assert "Corrupted checkpoint for stage 'ba'" in caplog.text

    def test_any_value_error_from_validation_is_treated_as_corrupt(self, tmp_path, monkeypatch, caplog):
        """The resume path must never raise: a ValueError from a validator, not only a JSON error, is logged."""
        from pydantic_agents.core.adapters import state_repository
        from pydantic_agents.core.models.core import PipelineState

        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        repo = state_repository.LocalDiskStateRepository("proj")
        repo.save("run-x", "ba", PipelineState(run_id="run-x", stage="ba"))

        with (
            patch.object(state_repository.PipelineState, "model_validate", side_effect=ValueError("validator bug")),
            caplog.at_level("WARNING", logger="pydantic_agents.core.adapters.state_repository"),
        ):
            assert repo.load("run-x", "ba") is None

        assert "Corrupted checkpoint for stage 'ba': validator bug" in caplog.text

    @staticmethod
    def _legacy_dir_with_two_runs(tmp_path, monkeypatch):
        """A project dir with one legacy file plus an unrelated current-format run; returns (repo, legacy path)."""
        from pydantic_agents.core.adapters import state_repository
        from pydantic_agents.core.models.core import PipelineState

        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        repo = state_repository.LocalDiskStateRepository("proj")
        repo.save("run-other", "ba", PipelineState(run_id="run-other", stage="ba"))
        legacy = tmp_path / "proj" / f"{TestCheckpointFiles.LEGACY_ID}_ba.json"
        legacy.write_text(PipelineState(run_id=TestCheckpointFiles.LEGACY_ID, stage="ba").model_dump_json())
        return repo, legacy

    def test_legacy_migration_missing_source_propagates(self, tmp_path, monkeypatch):
        """FileNotFoundError is only "someone else renamed it" when the target exists; otherwise it propagates."""
        from pydantic_agents.core.adapters import state_repository

        repo, legacy = self._legacy_dir_with_two_runs(tmp_path, monkeypatch)

        def vanish(src, dst):
            legacy.unlink()  # the source disappears under us and nothing appears at the target
            raise FileNotFoundError(src)

        with patch.object(state_repository.os, "replace", side_effect=vanish), pytest.raises(FileNotFoundError):
            repo._migrate_legacy_files()

        # The unrelated run was left untouched by the failed pass.
        assert (tmp_path / "proj" / "run-other__ba.json").exists()
        assert repo.load("run-other", "ba") is not None

    def test_legacy_migration_failure_is_retried_on_the_next_read(self, tmp_path, monkeypatch, caplog):
        from pydantic_agents.core.adapters import state_repository

        repo, legacy = self._legacy_dir_with_two_runs(tmp_path, monkeypatch)
        with (
            patch.object(state_repository.os, "replace", side_effect=FileNotFoundError("gone")),
            caplog.at_level("WARNING", logger="pydantic_agents.core.adapters.state_repository"),
        ):
            # The pass fails (no target appeared), is logged, and the read answers from the files that are there.
            assert repo.find_latest_run_id() == "run-other"
        assert "Could not migrate legacy checkpoint files" in caplog.text
        assert legacy.exists()

        # os.replace works again: the next read migrates the legacy file (the newest) and the run loads.
        assert repo.find_latest_run_id() == self.LEGACY_ID
        assert not legacy.exists()
        assert repo.load(self.LEGACY_ID, "ba") is not None

    def test_rename_done_by_another_instance_counts_as_migrated(self, tmp_path, monkeypatch, caplog):
        """Two instances on one directory: the loser of the rename race must not fail or warn."""
        from pydantic_agents.core.adapters import state_repository

        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / f"{self.LEGACY_ID}_ba.json").write_text('{"run_id": "x", "stage": "ba"}')
        real_replace = state_repository.os.replace

        def other_instance_wins(src, dst):
            real_replace(src, dst)  # the other instance renamed it between our listdir and replace
            raise FileNotFoundError(src)

        repo = state_repository.LocalDiskStateRepository("proj")
        with (
            patch.object(state_repository.os, "replace", side_effect=other_instance_wins),
            caplog.at_level("INFO", logger="pydantic_agents.core.adapters.state_repository"),
        ):
            assert repo.find_latest_run_id() == self.LEGACY_ID

        assert "Could not migrate" not in caplog.text
        assert "Migrated 1 legacy checkpoint file(s)" in caplog.text
        assert repo.exists(self.LEGACY_ID, "ba")

    def test_migration_never_overwrites_a_current_format_file(self, tmp_path, monkeypatch):
        from pydantic_agents.core.adapters.state_repository import LocalDiskStateRepository

        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / f"{self.LEGACY_ID}_ba.json").write_text('{"old": true}')
        (project_dir / f"{self.LEGACY_ID}__ba.json").write_text('{"new": true}')

        LocalDiskStateRepository("proj")

        assert (project_dir / f"{self.LEGACY_ID}__ba.json").read_text() == '{"new": true}'
        assert (project_dir / f"{self.LEGACY_ID}_ba.json").exists()

    def test_clear_removes_every_file_of_that_run(self, tmp_path, monkeypatch):
        from pydantic_agents.core.adapters.state_repository import LocalDiskStateRepository
        from pydantic_agents.core.models.core import PipelineState

        monkeypatch.setattr("pydantic_agents.core.adapters.state_repository.settings.checkpoint_dir", str(tmp_path))
        repo = LocalDiskStateRepository("proj")
        for stage in ("ba", "advisory"):
            (tmp_path / "proj" / f"{self.LEGACY_ID}_{stage}.json").write_text("{}")
        repo.save("run-new", "ba", PipelineState(run_id="run-new", stage="ba"))

        repo.clear(self.LEGACY_ID)

        assert sorted(f.name for f in (tmp_path / "proj").iterdir()) == ["run-new__ba.json"]


# --- Langfuse OTLP export -------------------------------------------------------------------------------------


class TestLangfuseExport:
    @pytest.fixture(autouse=True)
    def reset(self, monkeypatch):
        from pydantic_agents.core import observability

        monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
        monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
        monkeypatch.setattr("os.path.exists", lambda _p: False)
        monkeypatch.setattr(observability.settings, "logfire_token", "")
        observability._initialized = False
        observability.LOGFIRE_ENABLED = False
        observability.LANGFUSE_ENABLED = False
        observability.LOGFIRE_CONFIGURE_ERROR = None
        yield
        observability._initialized = False

    def test_langfuse_keys_alone_turn_the_export_on(self, monkeypatch):
        from pydantic_agents.core import observability

        monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example/")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        fake_logfire = MagicMock()
        exporter = MagicMock()
        with (
            patch("pydantic_agents.core.observability.logfire", fake_logfire),
            patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", exporter),
        ):
            observability.configure_observability()

        assert observability.LANGFUSE_ENABLED is True
        assert observability.LOGFIRE_ENABLED is False
        endpoint = exporter.call_args.kwargs["endpoint"]
        assert endpoint == "https://langfuse.example/api/public/otel/v1/traces"
        assert exporter.call_args.kwargs["headers"]["Authorization"].startswith("Basic ")
        configure = fake_logfire.configure.call_args.kwargs
        assert configure["send_to_logfire"] is False
        assert len(configure["additional_span_processors"]) == 1
        fake_logfire.instrument_pydantic_ai.assert_called_once()
        fake_logfire.instrument_httpx.assert_called_once()  # traceparent reaches LiteLLM

    def test_without_any_backend_telemetry_stays_off(self):
        from pydantic_agents.core import observability

        observability.configure_observability()

        assert observability.LANGFUSE_ENABLED is False
        assert "Langfuse" in (observability.LOGFIRE_CONFIGURE_ERROR or "")


# --- the per-call run id in run_agent_by_name ------------------------------------------------------------------


async def test_a_single_agent_call_gets_its_own_run_id_and_restores_the_context():
    from pydantic_agents import mcp_server

    seen = {}

    async def run(name, prompt, **_k):
        seen["run_id"] = run_id_var.get()
        return MagicMock(output="x")

    with (
        patch("pydantic_agents.mcp_server.registry") as reg,
        patch("pydantic_agents.mcp_server.retrieve_context", AsyncMock(return_value="")),
    ):
        reg.is_registered.return_value = True
        reg.run = run
        token = run_id_var.set(None)
        try:
            await copy_context().run(lambda: asyncio.ensure_future(mcp_server.run_agent_by_name("ba", "q")))
            await asyncio.sleep(0)
            assert run_id_var.get() is None
        finally:
            run_id_var.reset(token)
    assert seen["run_id"].startswith("call_")


# --- the verifier runs the real toolchain -----------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("go") is None, reason="needs the Go toolchain")
class TestVerifier:
    async def test_compiling_code_with_a_passing_test_passes(self):
        from pydantic_agents.core.models.core import Artifact
        from pydantic_agents.core.verifier import verify_artifacts

        art = Artifact(
            agent_name="backend",
            files={
                "go.mod": "module tempmodule\n\ngo 1.22\n",
                "sum.go": "package tempmodule\n\nfunc Sum(a, b int) int { return a + b }\n",
                "sum_test.go": (
                    'package tempmodule\n\nimport "testing"\n\n'
                    'func TestSum(t *testing.T) { if Sum(1, 2) != 3 { t.Fatal("bad") } }\n'
                ),
            },
        )
        report = await verify_artifacts([art])
        assert report.passed, report.failures()
        assert [c.name for c in report.checks] == ["go build", "go vet", "go test"]
        assert report.as_prompt().startswith("VERIFICATION: passed")

    async def test_a_compile_error_fails_and_names_the_check(self):
        from pydantic_agents.core.models.core import Artifact
        from pydantic_agents.core.verifier import verify_artifacts

        art = Artifact(
            agent_name="backend", files={"go.mod": "module tempmodule\n\ngo 1.22\n", "x.go": "package x\nfunc (\n"}
        )
        report = await verify_artifacts([art])
        assert report.passed is False
        assert report.checks[0].name == "go build"
        assert not report.checks[0].passed
        assert "VERIFICATION: FAILED" in report.as_prompt()

    async def test_no_go_files_is_a_skip_not_a_pass(self):
        from pydantic_agents.core.models.core import Artifact
        from pydantic_agents.core.verifier import verify_artifacts

        report = await verify_artifacts([Artifact(agent_name="docs", files={"README.md": "x"})])
        assert report.skipped
        assert not report.passed


async def test_run_test_suite_reports_setup_errors_as_failures(monkeypatch):
    from pydantic_agents.core.tools import test_tools

    async def broken(_dir):
        return "'go mod init' timed out."

    monkeypatch.setattr(test_tools, "init_and_tidy", broken)
    result = await test_tools.run_test_suite({"svc.go": "package svc"}, {"svc_test.go": "package svc"})
    assert result.passed is False
    assert result.error == "'go mod init' timed out."
    assert result.trace().startswith("TEST EXECUTION FAILED")


def test_json_output_of_trace_records_survives_a_roundtrip(tmp_path):
    rec = session_recorder.SessionRecorder(filepath=str(tmp_path / "t.jsonl"))
    record = rec._new_record("r", "a", "s", "p")
    rec._finish(record, success=True, output={"k": 1})
    [line] = (tmp_path / "t.jsonl").read_text().splitlines()
    assert json.loads(line)["output_json"] == {"k": 1}
