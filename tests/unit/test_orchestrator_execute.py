"""OrgitrationPipeline.execute: stage order, the implementation/quality-gate retry loop, resume, merge."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from pydantic_agents.core.limits import run_id_var
from pydantic_agents.core.models.core import FailureReport
from pydantic_agents.core.pipeline import orchestrator
from pydantic_agents.core.pipeline.config import STAGES, PipelineConfig
from pydantic_agents.core.settings import settings


class FakeStage:
    """Records every run in ``calls`` and lets the test script what each run does to the state."""

    def __init__(self, name: str, calls: list[str], on_run=None):
        self.name = name
        self.calls = calls
        self.on_run = on_run
        self.agent_provider = None

    async def run(self, state, config):
        self.calls.append(self.name)
        if self.on_run:
            self.on_run(state, len(self.calls))
        return state


def _config(step: str = "all", resume: bool = False) -> PipelineConfig:
    config = PipelineConfig(step=step, resume=resume, requirement="build it")
    return config


@pytest.fixture
def make(monkeypatch):
    """Build a pipeline whose repository, checkpoints and artifact writer are fakes."""
    monkeypatch.setattr(
        orchestrator, "build_state_repository", lambda _project: MagicMock(find_latest_run_id=lambda: None)
    )

    def build(config: PipelineConfig, stages: dict[str, FakeStage]):
        pipe = orchestrator.OrgitrationPipeline(config)
        pipe.checkpoint_mgr = MagicMock()
        pipe.checkpoint_mgr.stage_exists.return_value = False
        pipe.artifact_writer = MagicMock()
        for name, stage in stages.items():
            pipe.register_stage(name, stage)
        return pipe

    return build


def _all_stages(calls, **hooks) -> dict[str, FakeStage]:
    return {
        name: FakeStage(name, calls, hooks.get(name)) for name in ("ba", "advisory", "implementation", "quality_gate")
    }


def _approve(state, _n):
    state.approved = True


# --- routing ----------------------------------------------------------------------------------------------------


async def test_an_unknown_step_does_nothing_and_says_so(make, caplog):
    calls: list[str] = []
    pipe = make(_config(step="nonsense"), _all_stages(calls))

    with caplog.at_level(logging.ERROR, logger=orchestrator.logger.name):
        await pipe.execute()

    assert calls == []
    assert any("Unknown step: nonsense" in r.getMessage() for r in caplog.records)
    pipe.artifact_writer.write_artifacts.assert_not_called()


async def test_a_full_run_goes_through_the_stages_in_order_with_the_gate_inside_the_loop(make):
    calls: list[str] = []
    pipe = make(_config(), _all_stages(calls, quality_gate=_approve))

    await pipe.execute()

    assert calls == ["ba", "advisory", "implementation", "quality_gate"]
    saved = [c.args[0] for c in pipe.checkpoint_mgr.save.call_args_list]
    assert saved == [STAGES.BA, STAGES.ADVISORY, STAGES.IMPLEMENTATION]  # the gate has no checkpoint of its own
    pipe.artifact_writer.write_artifacts.assert_called_once_with(pipe.state.artifacts)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        ("ba", ["ba"]),
        ("advisory", ["advisory"]),
        ("implementation", ["implementation", "quality_gate"]),
        ("backend", ["implementation", "quality_gate"]),
        ("docs", ["implementation", "quality_gate"]),
        ("review", ["quality_gate"]),
        ("security", ["quality_gate"]),
        ("sre", ["quality_gate"]),
    ],
)
async def test_a_single_step_runs_only_its_stage(make, step, expected):
    calls: list[str] = []
    pipe = make(_config(step=step), _all_stages(calls))

    await pipe.execute()

    assert calls == expected


async def test_an_unregistered_stage_is_skipped_with_a_warning(make, caplog):
    calls: list[str] = []
    stages = _all_stages(calls, quality_gate=_approve)
    del stages["advisory"]
    pipe = make(_config(), stages)

    with caplog.at_level(logging.WARNING, logger=orchestrator.logger.name):
        await pipe.execute()

    assert calls == ["ba", "implementation", "quality_gate"]
    # The stage name is a str-Enum, which formats as "advisory" on 3.11 and "StageName.ADVISORY" on 3.12+.
    assert any(
        "advisory" in r.getMessage().lower() and "not registered. Skipping." in r.getMessage() for r in caplog.records
    )


async def test_without_a_quality_gate_the_loop_runs_to_max_retries(make, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 2)
    calls: list[str] = []
    stages = _all_stages(calls)
    del stages["quality_gate"]
    pipe = make(_config(), stages)

    await pipe.execute()

    assert calls == ["ba", "advisory", "implementation", "implementation"]  # nothing can approve, so it retries


# --- the implementation retry loop ------------------------------------------------------------------------------


def _fail_tests(state, _n):
    state.failure_report = FailureReport(source="test_suite", error_type="tests", trace="boom")


async def test_a_test_failure_retries_implementation_without_running_the_gate(make, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 3)
    calls: list[str] = []

    def impl(state, n):
        if n <= 2:  # impl is the third call overall only when ba/advisory ran first; count impl runs instead
            pass
        impl_runs = calls.count("implementation")
        if impl_runs < 3:
            _fail_tests(state, impl_runs)
        else:
            state.failure_report = None

    pipe = make(_config(), _all_stages(calls, implementation=impl, quality_gate=_approve))

    await pipe.execute()

    assert calls.count("implementation") == 3
    assert calls.count("quality_gate") == 1  # only after the clean attempt
    assert pipe.state.approved is True


async def test_the_loop_stops_at_max_retries_when_tests_keep_failing(make, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 2)
    calls: list[str] = []
    pipe = make(_config(), _all_stages(calls, implementation=_fail_tests))

    await pipe.execute()

    assert calls.count("implementation") == 2
    assert "quality_gate" not in calls


async def test_the_loop_retries_when_the_gate_does_not_approve(make, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 3)
    calls: list[str] = []

    def gate(state, _n):
        state.approved = calls.count("quality_gate") == 2  # approves on its second run

    pipe = make(_config(), _all_stages(calls, quality_gate=gate))

    await pipe.execute()

    assert calls.count("implementation") == 2
    assert calls.count("quality_gate") == 2


async def test_a_single_step_gets_one_attempt_whatever_max_retries_says(make, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 5)
    calls: list[str] = []
    pipe = make(_config(step="backend"), _all_stages(calls))

    await pipe.execute()

    assert calls == ["implementation", "quality_gate"]


# --- resume -----------------------------------------------------------------------------------------------------


async def test_a_checkpointed_stage_is_loaded_instead_of_run(make):
    calls: list[str] = []
    pipe = make(_config(resume=True), _all_stages(calls, quality_gate=_approve))
    loaded = pipe.state.model_copy(update={"stage": "ba"})
    pipe.checkpoint_mgr.stage_exists.side_effect = lambda name: name == STAGES.BA
    pipe.checkpoint_mgr.load.return_value = loaded

    await pipe.execute()

    assert calls == ["advisory", "implementation", "quality_gate"]
    pipe.checkpoint_mgr.load.assert_called_once_with(STAGES.BA)
    assert STAGES.BA not in [c.args[0] for c in pipe.checkpoint_mgr.save.call_args_list]


async def test_checkpoints_are_ignored_without_resume(make):
    calls: list[str] = []
    pipe = make(_config(resume=False), _all_stages(calls))
    pipe.checkpoint_mgr.stage_exists.return_value = True

    await pipe.execute()

    assert calls[0] == "ba"
    pipe.checkpoint_mgr.load.assert_not_called()


# --- failures and merge -----------------------------------------------------------------------------------------


async def test_a_crashing_stage_is_reraised_without_a_checkpoint_or_flush(make, caplog):
    calls: list[str] = []

    def crash(_state, _n):
        raise RuntimeError("stage blew up")

    pipe = make(_config(), _all_stages(calls, advisory=crash))

    with caplog.at_level(logging.ERROR, logger=orchestrator.logger.name), pytest.raises(RuntimeError, match="blew up"):
        await pipe.execute()

    assert calls == ["ba", "advisory"]
    assert [c.args[0] for c in pipe.checkpoint_mgr.save.call_args_list] == [STAGES.BA]
    pipe.artifact_writer.write_artifacts.assert_not_called()
    assert any("Pipeline crashed during stage" in r.getMessage() for r in caplog.records)


async def test_an_approved_full_run_ends_with_the_merge_stage(make):
    calls: list[str] = []
    pipe = make(_config(), _all_stages(calls, quality_gate=_approve))

    class FakeMerge:
        agent_provider = None

        async def run(self, state, config):
            calls.append("merge")
            return state

    with patch("pydantic_agents.core.stages.merge_stage.MergeStage", FakeMerge):
        await pipe.execute()

    assert calls[-1] == "merge"


@pytest.mark.parametrize(("step", "approve"), [("all", False), ("backend", True)], ids=["not-approved", "single-step"])
async def test_merge_only_runs_for_an_approved_full_run(make, step, approve):
    calls: list[str] = []
    hooks = {"quality_gate": _approve} if approve else {}
    pipe = make(_config(step=step), _all_stages(calls, **hooks))

    class FakeMerge:
        agent_provider = None

        async def run(self, state, config):
            calls.append("merge")
            return state

    with patch("pydantic_agents.core.stages.merge_stage.MergeStage", FakeMerge):
        await pipe.execute()

    assert "merge" not in calls


# --- bookkeeping ------------------------------------------------------------------------------------------------


async def test_the_run_id_is_published_for_tracing_once(make):
    token = run_id_var.set(None)
    try:
        pipe = make(_config(step="ba"), _all_stages([]))
        await pipe.execute()
        assert run_id_var.get() == pipe.state.run_id
        run_id_var.set("already-set")
        await pipe.execute()
        assert run_id_var.get() == "already-set"
    finally:
        run_id_var.reset(token)


def test_registering_a_stage_gives_it_the_pipeline_provider_unless_it_has_one(make):
    pipe = make(_config(), {})
    bare, own = FakeStage("a", []), FakeStage("b", [])
    own.agent_provider = "its own"

    pipe.register_stage("a", bare)
    pipe.register_stage("b", own)

    assert bare.agent_provider is pipe.agent_provider
    assert own.agent_provider == "its own"
