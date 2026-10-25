"""run_existing_code_pipeline: the engineer -> tests -> parallel gates loop with autonomous remediation.

The agents, gates, event recorder and memory sync are replaced with scripted fakes, so these tests pin the control
flow (what runs, in which order, what is recorded, when the loop retries or stops) and not any model behaviour.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_agents.core import existing_code_handler as ech
from pydantic_agents.core.models.core import Artifact

MAX = ech.MAX_RETRIES


def _artifact(agent: str = "service_engineer", **files: str) -> Artifact:
    return Artifact(
        agent_name=agent, files={name.replace("__", "/").replace("_dot_", "."): body for name, body in files.items()}
    )


class Script:
    """Hands out scripted outcomes per agent; an exception outcome is raised."""

    def __init__(self, **queues):
        self.queues = {agent: list(outcomes) for agent, outcomes in queues.items()}
        self.calls: list[tuple[str, str, str, dict | None]] = []

    async def run_stage(self, stage_name, agent_name, prompt, span_attrs=None):
        self.calls.append((stage_name, agent_name, prompt, span_attrs))
        outcome = self.queues[agent_name].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(output=outcome), True

    def agents(self) -> list[str]:
        return [c[1] for c in self.calls]


class Recorder:
    def __init__(self):
        self.events: list[tuple[str, int, str, str]] = []

    def record(self, run_id, stage, attempt, verdict, duration_ms, detail=""):
        assert run_id  # every event belongs to the run
        self.events.append((stage, attempt, verdict, detail))

    def verdicts(self, stage: str) -> list[tuple[int, str]]:
        return [(a, v) for s, a, v, _ in self.events if s == stage]


@pytest.fixture
def env(monkeypatch):
    """Patch the module's collaborators; returns a namespace with the fakes."""
    recorder = Recorder()
    sync = MagicMock()
    quality = AsyncMock(return_value=(True, 90, []))
    review = AsyncMock(return_value=(True, "low", []))
    monkeypatch.setattr(ech, "build_stage_event_recorder", lambda _project: recorder)
    monkeypatch.setattr(ech, "_sync_memory_after_pipeline", sync)
    monkeypatch.setattr(ech, "_run_quality_gate", quality)
    monkeypatch.setattr(ech, "_run_code_review", review)
    monkeypatch.delenv("ORCHESTRATION_CODEBASE_ROOT", raising=False)
    holder = SimpleNamespace(recorder=recorder, sync=sync, quality=quality, review=review, script=None)

    def use(script: Script) -> SimpleNamespace:
        holder.script = script
        monkeypatch.setattr(ech, "_run_stage", script.run_stage)
        return holder

    holder.use = use
    return holder


async def _run(env, script, **kwargs):
    env.use(script)
    return await ech.run_existing_code_pipeline("add a health endpoint", "/srv/app", **kwargs)


# --- the first attempt succeeds ---------------------------------------------------------------------------------


async def test_a_clean_first_attempt_passes(env):
    script = Script(
        service_engineer=[_artifact(main_dot_go="x")], test_eng=[_artifact("test_eng", main_test_dot_go="t")]
    )

    result = await _run(env, script)

    assert result.passed is True
    assert result.retries_used == 0
    assert result.changed_files == ["main.go"]
    assert result.test_files == ["main_test.go"]
    assert [a.agent_name for a in result.artifacts] == ["service_engineer", "test_eng"]
    assert result.quality_gate_score == 90
    assert result.review_risk == "low"
    assert result.errors == []
    assert script.agents() == ["service_engineer", "test_eng"]
    env.sync.assert_called_once_with()


async def test_every_stage_is_recorded_in_order(env):
    script = Script(service_engineer=[_artifact(main_dot_go="x")], test_eng=[_artifact("test_eng", t_dot_go="t")])

    await _run(env, script)

    assert [(s, a, v) for s, a, v, _ in env.recorder.events] == [
        ("service_engineer", 0, "ok"),
        ("test_engineer", 0, "ok"),
        ("quality_gate", 0, "passed"),
        ("review", 0, "approved"),
        ("pipeline", 0, "passed"),
    ]
    assert env.recorder.events[0][3] == "1 files"
    assert env.recorder.events[3][3] == "risk=low"


async def test_the_engineer_prompt_names_the_change_and_the_codebase(env):
    script = Script(service_engineer=[_artifact(a_dot_go="x")], test_eng=[_artifact("test_eng", t_dot_go="t")])

    await _run(env, script)

    stage_name, agent, prompt, attrs = script.calls[0]
    assert (stage_name, agent) == ("service_engineer.attempt_0", "service_engineer")
    assert "/srv/app" in prompt
    assert "add a health endpoint" in prompt
    assert attrs == {"codebase_path": "/srv/app", "attempt": 0}


async def test_the_test_engineer_prompt_includes_the_changed_files(env):
    script = Script(
        service_engineer=[_artifact(main_dot_go="package main")], test_eng=[_artifact("test_eng", t_dot_go="t")]
    )

    await _run(env, script)

    stage_name, agent, prompt, attrs = script.calls[1]
    assert (stage_name, agent) == ("test_engineer.attempt_0", "test_eng")
    assert "main.go" in prompt
    assert "package main" in prompt
    assert attrs == {"changed_files_count": 1}


async def test_the_gates_see_the_engineer_and_test_artifacts(env):
    script = Script(service_engineer=[_artifact(main_dot_go="x")], test_eng=[_artifact("test_eng", t_dot_go="t")])

    await _run(env, script)

    [seen] = [c.args[0] for c in env.quality.call_args_list]
    assert [a.agent_name for a in seen] == ["service_engineer", "test_eng"]
    assert [c.args[0] for c in env.review.call_args_list] == [seen]


# --- options ----------------------------------------------------------------------------------------------------


async def test_tests_can_be_skipped(env):
    script = Script(service_engineer=[_artifact(main_dot_go="x")])

    result = await _run(env, script, run_tests=False)

    assert result.passed is True
    assert result.test_files == []
    assert script.agents() == ["service_engineer"]
    assert env.recorder.verdicts("test_engineer") == []


async def test_with_both_gates_off_the_run_passes_after_the_engineer_and_tests(env):
    script = Script(service_engineer=[_artifact(main_dot_go="x")], test_eng=[_artifact("test_eng", t_dot_go="t")])

    result = await _run(env, script, run_quality_gate=False, run_review=False)

    assert result.passed is True
    env.quality.assert_not_called()
    env.review.assert_not_called()
    assert env.recorder.events[-1] == ("pipeline", 0, "passed", "gates disabled")
    env.sync.assert_not_called()  # this early exit does not sync memory


async def test_a_disabled_review_counts_as_approved_with_low_risk(env):
    script = Script(service_engineer=[_artifact(main_dot_go="x")], test_eng=[_artifact("test_eng", t_dot_go="t")])

    result = await _run(env, script, run_review=False)

    assert result.passed is True
    env.review.assert_not_called()
    assert result.review_risk == "low"
    assert env.recorder.verdicts("review") == [(0, "approved")]


async def test_a_disabled_quality_gate_counts_as_passed_with_score_zero(env):
    script = Script(service_engineer=[_artifact(main_dot_go="x")], test_eng=[_artifact("test_eng", t_dot_go="t")])

    result = await _run(env, script, run_quality_gate=False)

    assert result.passed is True
    env.quality.assert_not_called()
    assert result.quality_gate_score == 0
    assert env.recorder.verdicts("quality_gate") == [(0, "passed")]


# --- early exits ------------------------------------------------------------------------------------------------


async def test_no_changed_files_passes_immediately(env):
    script = Script(service_engineer=[_artifact()])

    result = await _run(env, script)

    assert result.passed is True
    assert result.changed_files == []
    assert script.agents() == ["service_engineer"]
    assert env.recorder.events[-1] == ("pipeline", 0, "passed", "no changes")
    env.quality.assert_not_called()


@pytest.mark.parametrize(
    ("codebase", "expect_set"),
    [("/srv/app", True), ("", False)],
)
async def test_the_codebase_root_environment_variable_follows_the_path(env, monkeypatch, codebase, expect_set):
    monkeypatch.setenv("ORCHESTRATION_CODEBASE_ROOT", "stale")
    env.use(Script(service_engineer=[_artifact()]))

    await ech.run_existing_code_pipeline("change", codebase)

    if expect_set:
        assert os.environ["ORCHESTRATION_CODEBASE_ROOT"] == os.path.realpath(codebase)
    else:
        assert "ORCHESTRATION_CODEBASE_ROOT" not in os.environ


# --- retries ----------------------------------------------------------------------------------------------------


async def test_an_engineer_failure_is_retried_with_the_error_in_the_prompt(env):
    script = Script(
        service_engineer=[RuntimeError("model down"), _artifact(main_dot_go="x")],
        test_eng=[_artifact("test_eng", t_dot_go="t")],
    )

    result = await _run(env, script)

    assert result.passed is True
    assert result.retries_used == 1
    assert result.errors == ["Service Engineer failed on attempt 0: model down"]
    retry_prompt = script.calls[1][2]
    assert script.calls[1][0] == "service_engineer.attempt_1"
    assert f"ATTEMPT 2/{MAX}" in retry_prompt
    assert "Service Engineer failed on attempt 0: model down" in retry_prompt
    assert env.recorder.verdicts("service_engineer") == [(0, "error"), (1, "ok")]
    assert env.recorder.events[0][3] == "model down"


async def test_a_failing_test_engineer_forces_a_retry_even_if_the_gates_pass(env):
    script = Script(
        service_engineer=[_artifact(main_dot_go="x"), _artifact(main_dot_go="y")],
        test_eng=[RuntimeError("no tests"), _artifact("test_eng", t_dot_go="t")],
    )

    result = await _run(env, script)

    assert result.passed is True
    assert result.retries_used == 1
    assert result.errors == ["Test Engineer failed on attempt 0: no tests"]
    assert env.recorder.verdicts("test_engineer") == [(0, "error"), (1, "ok")]
    assert env.recorder.verdicts("pipeline") == [(1, "passed")]


async def test_a_test_result_without_files_is_not_recorded_as_test_files(env):
    script = Script(service_engineer=[_artifact(main_dot_go="x")], test_eng=["just prose"])

    result = await _run(env, script)

    assert result.passed is True
    assert result.test_files == []
    assert [a.agent_name for a in result.artifacts] == ["service_engineer"]
    assert env.recorder.verdicts("test_engineer") == [(0, "ok")]


async def test_a_failed_quality_gate_is_retried_with_its_feedback(env):
    script = Script(
        service_engineer=[_artifact(main_dot_go="x"), _artifact(main_dot_go="y")],
        test_eng=[_artifact("test_eng", t_dot_go="t"), _artifact("test_eng", t_dot_go="t")],
    )
    env.quality.side_effect = [(False, 40, ["too complex"]), (True, 85, [])]

    result = await _run(env, script)

    assert result.passed is True
    assert result.retries_used == 1
    assert result.errors == ["too complex"]
    assert "too complex" in script.calls[2][2]  # the second engineer prompt (calls: eng, test, eng, test)
    assert env.recorder.verdicts("quality_gate") == [(0, "failed"), (1, "passed")]
    assert result.quality_gate_score == 85
    assert result.quality_gate_feedback == []


async def test_a_rejected_review_is_retried_and_its_feedback_is_kept(env):
    script = Script(
        service_engineer=[_artifact(main_dot_go="x"), _artifact(main_dot_go="y")],
        test_eng=[_artifact("test_eng", t_dot_go="t"), _artifact("test_eng", t_dot_go="t")],
    )
    env.review.side_effect = [(False, "high", ["sql injection"]), (True, "low", [])]

    result = await _run(env, script)

    assert result.passed is True
    assert result.retries_used == 1
    assert result.errors == ["sql injection"]
    assert env.recorder.verdicts("review") == [(0, "rejected"), (1, "approved")]
    assert result.review_risk == "low"


async def test_review_feedback_is_kept_even_when_the_review_approves(env):
    script = Script(service_engineer=[_artifact(main_dot_go="x")], test_eng=[_artifact("test_eng", t_dot_go="t")])
    env.review.return_value = (True, "low", ["consider a comment"])

    result = await _run(env, script)

    assert result.passed is True
    assert result.errors == ["consider a comment"]


async def test_a_gate_that_raises_fails_the_attempt_and_is_recorded_as_an_error(env):
    script = Script(
        service_engineer=[_artifact(main_dot_go="x"), _artifact(main_dot_go="y")],
        test_eng=[_artifact("test_eng", t_dot_go="t"), _artifact("test_eng", t_dot_go="t")],
    )
    env.quality.side_effect = [RuntimeError("gate crashed"), (True, 80, [])]
    env.review.side_effect = [RuntimeError("review crashed"), (True, "low", [])]

    result = await _run(env, script)

    assert result.passed is True
    assert result.retries_used == 1
    assert result.errors == ["Quality Gate exception: gate crashed", "Code Review exception: review crashed"]
    assert env.recorder.verdicts("quality_gate") == [(0, "error"), (1, "passed")]
    assert env.recorder.verdicts("review") == [(0, "error"), (1, "approved")]


# --- giving up --------------------------------------------------------------------------------------------------


async def test_exhausting_retries_fails_with_all_errors_and_syncs_memory(env):
    attempts = MAX + 1
    script = Script(
        service_engineer=[_artifact(main_dot_go="x")] * attempts,
        test_eng=[_artifact("test_eng", t_dot_go="t")] * attempts,
    )
    env.quality.return_value = (False, 10, ["same problem"])

    result = await _run(env, script)

    assert result.passed is False
    assert result.retries_used == MAX
    assert result.errors == ["same problem"]  # the same feedback on every attempt is recorded once
    assert script.agents().count("service_engineer") == attempts
    assert env.recorder.verdicts("quality_gate") == [(a, "failed") for a in range(attempts)]
    assert env.recorder.events[-1] == ("pipeline", MAX, "failed", "1 errors")
    env.sync.assert_called_once_with()


async def test_an_engineer_that_never_succeeds_ends_the_run_failed(env):
    attempts = MAX + 1
    script = Script(service_engineer=[RuntimeError(f"boom {i}") for i in range(attempts)])

    result = await _run(env, script)

    assert result.passed is False
    assert result.retries_used == MAX
    assert result.errors == [f"Service Engineer failed on attempt {i}: boom {i}" for i in range(attempts)]
    assert env.recorder.events[-1] == ("pipeline", MAX, "failed", f"{attempts} errors")
    env.quality.assert_not_called()
    env.sync.assert_called_once_with()


async def test_each_attempt_starts_from_the_new_engineer_artifact(env):
    script = Script(
        service_engineer=[_artifact(old_dot_go="1"), _artifact(new_dot_go="2")],
        test_eng=[_artifact("test_eng", t1_dot_go="t"), _artifact("test_eng", t2_dot_go="t")],
    )
    env.quality.side_effect = [(False, 1, ["x"]), (True, 90, [])]

    result = await _run(env, script)

    assert result.changed_files == ["new.go"]
    assert result.test_files == ["t2.go"]
    assert [list(a.files) for a in result.artifacts] == [["new.go"], ["t2.go"]]
