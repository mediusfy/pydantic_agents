"""The quality gate and CTO advisory must fail closed and compute verdicts in code."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from pydantic_agents.core.models.core import (
    AgentAdvice,
    Artifact,
    PipelineState,
    QualityGateResult,
    SecurityReport,
    UserStory,
)
from pydantic_agents.core.pipeline.config import PipelineConfig


def test_agent_advice_defaults_to_not_approved():
    assert AgentAdvice().approved_to_proceed is False


@pytest.mark.parametrize(
    "claimed_passed, score, expected",
    [
        (True, 25, True),
        (True, 60, True),
        (True, 61, False),
        (False, 10, True),
        (False, 75, False),
    ],
)
def test_quality_gate_passed_is_derived_from_score(claimed_passed, score, expected):
    result = QualityGateResult(passed=claimed_passed, complexity_score=score, feedback=[])
    assert result.passed is expected


def test_quality_gate_passed_ignores_omitted_field():
    assert QualityGateResult(complexity_score=90, feedback=[]).passed is False


@pytest.mark.parametrize("score", [-1, 101])
def test_security_score_is_bounded(score):
    with pytest.raises(ValidationError):
        SecurityReport(passed=True, score=score)


def _runner(outputs):
    provider = AsyncMock()

    async def run_agent(sem, name, agent, prompt, deps=None):
        out = outputs[name]
        if isinstance(out, Exception):
            raise out
        return MagicMock(output=out)

    provider.run_agent.side_effect = run_agent
    return provider


async def test_security_failure_sends_feedback_instead_of_crashing():
    from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

    outputs = {
        "sre": Artifact(agent_name="sre", files={"a.yaml": "x"}),
        "security": SecurityReport(passed=False, score=20, feedback=["SQL injection in handler"]),
    }
    stage = QualityGateStage(agent_provider=_runner(outputs))
    state = PipelineState(run_id="sec-fail", stage="implementation", artifacts=[])
    config = PipelineConfig(project="p", requirement="r")

    with (
        patch("pydantic_agents.core.stages.quality_gate_stage.registry.get", return_value=MagicMock()),
        patch(
            "pydantic_agents.core.stages.quality_gate_stage.CodeValidator.run_all_async", AsyncMock(return_value=True)
        ),
        patch("pydantic_agents.core.context_engine.retrieve_context", return_value=""),
    ):
        result = await stage.run(state, config)

    assert result.approved is False
    assert "SQL injection in handler" in result.feedback
    assert result.failure_report.error_type == "security"


@pytest.mark.parametrize(
    "cto_output, gate_result, expected",
    [
        (RuntimeError("CTO API error"), False, False),
        (RuntimeError("CTO API error"), True, True),  # explicit human override still works
        (AgentAdvice(), False, False),  # omitted approved_to_proceed
        (AgentAdvice(approved_to_proceed=True), True, True),
    ],
)
async def test_advisory_fails_closed(cto_output, gate_result, expected):
    from pydantic_agents.core.stages.advisory_stage import AdvisoryStage

    outputs = {"cto": cto_output, "ux": AgentAdvice(recommendations=["x"])}
    stage = AdvisoryStage(agent_provider=_runner(outputs))
    state = PipelineState(
        run_id="adv", stage="ba", user_stories=[UserStory(id="US-1", title="t", gherkin="Given When Then")]
    )
    config = PipelineConfig(project="p", requirement="r")

    async def gate(state, project):
        # Mirror the real gate: an unapproved state is blocked unless overridden.
        return gate_result

    with (
        patch("pydantic_agents.core.cto_gate.apply", side_effect=gate),
        patch("pydantic_agents.core.stages.advisory_stage.registry.get", return_value=MagicMock()),
    ):
        result = await stage.run(state, config)

    assert result.cto_approved is expected


@pytest.mark.parametrize("score, expected", [(0, True), (100, False)])
def test_quality_gate_score_boundaries(score, expected):
    assert QualityGateResult(complexity_score=score, feedback=[]).passed is expected


def test_quality_gate_prompt_uses_the_single_threshold():
    from pydantic_agents.agents.quality_gate import quality_gate_prompt
    from pydantic_agents.core.models.core import QUALITY_GATE_MAX_COMPLEXITY

    assert f"<= {QUALITY_GATE_MAX_COMPLEXITY}" in quality_gate_prompt
    assert "__MAX_COMPLEXITY__" not in quality_gate_prompt


async def test_security_failure_without_feedback_still_gives_the_retry_loop_something():
    from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

    outputs = {
        "sre": Artifact(agent_name="sre", files={"a.yaml": "x"}),
        "security": SecurityReport(passed=False, score=10),
    }
    state = PipelineState(run_id="sec-empty", stage="implementation", artifacts=[])
    with (
        patch("pydantic_agents.core.stages.quality_gate_stage.registry.get", return_value=MagicMock()),
        patch(
            "pydantic_agents.core.stages.quality_gate_stage.CodeValidator.run_all_async", AsyncMock(return_value=True)
        ),
        patch("pydantic_agents.core.context_engine.retrieve_context", return_value=""),
    ):
        result = await QualityGateStage(agent_provider=_runner(outputs)).run(
            state, PipelineConfig(project="p", requirement="r")
        )
    assert "no details" in result.feedback


async def test_omitted_cto_verdict_is_logged(caplog):
    from pydantic_agents.core.stages.advisory_stage import AdvisoryStage

    outputs = {"cto": AgentAdvice(recommendations=["x"]), "ux": AgentAdvice()}
    state = PipelineState(run_id="adv", stage="ba", user_stories=[UserStory(id="U", title="t", gherkin="g")])
    with (
        patch("pydantic_agents.core.cto_gate.apply", AsyncMock(return_value=False)),
        patch("pydantic_agents.core.stages.advisory_stage.registry.get", return_value=MagicMock()),
        caplog.at_level("WARNING"),
    ):
        await AdvisoryStage(agent_provider=_runner(outputs)).run(state, PipelineConfig(project="p", requirement="r"))
    assert "omitted approved_to_proceed" in caplog.text


# The real gate (no mocks): a blocked state is released only by the explicit human overrides.


@pytest.fixture
def blocked_state():
    return PipelineState(run_id="gate", stage="advisory", cto_approved=False)


async def test_real_cto_gate_blocks_without_an_override(blocked_state, tmp_path, monkeypatch):
    from pydantic_agents.core import cto_gate

    monkeypatch.delenv("ORCHESTRATION_CTO_OVERRIDE", raising=False)
    monkeypatch.setattr("pydantic_agents.core.settings.settings.output_dir", tmp_path)

    async def no_wait(filepath, **_kwargs):
        return False

    monkeypatch.setattr(cto_gate, "_poll_for_file_approval", no_wait)
    assert await cto_gate.apply(blocked_state, project="p") is False


async def test_real_cto_gate_env_override(blocked_state, tmp_path, monkeypatch):
    from pydantic_agents.core import cto_gate

    monkeypatch.setenv("ORCHESTRATION_CTO_OVERRIDE", "1")
    monkeypatch.setattr("pydantic_agents.core.settings.settings.output_dir", tmp_path)
    assert await cto_gate.apply(blocked_state, project="p") is True


async def test_real_cto_gate_marker_file_override(blocked_state, tmp_path, monkeypatch):
    from pydantic_agents.core import cto_gate

    monkeypatch.delenv("ORCHESTRATION_CTO_OVERRIDE", raising=False)
    monkeypatch.setattr("pydantic_agents.core.settings.settings.output_dir", tmp_path)
    cto_gate._approval_file_path("p").write_text("true")
    assert await cto_gate.apply(blocked_state, project="p") is True
