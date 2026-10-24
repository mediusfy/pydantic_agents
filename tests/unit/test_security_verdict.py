"""SecurityReport.passed is derived from the score, so the verdict can never contradict it."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_agents.core.models.core import (
    SECURITY_CRITICAL_MAX_SCORE,
    SECURITY_HIGH_MAX_SCORE,
    SECURITY_PASS_SCORE,
    Artifact,
    PipelineState,
    SecurityReport,
)
from pydantic_agents.core.pipeline.config import PipelineConfig


@pytest.mark.parametrize(
    "claimed, score, expected",
    [
        (True, 100, True),
        (True, SECURITY_PASS_SCORE, True),
        (True, SECURITY_PASS_SCORE - 1, False),  # the model claims a pass the score does not support
        (False, SECURITY_PASS_SCORE, True),
        (False, 55, False),  # the case seen in `make eval`: failed verdict with "no critical/high findings"
        (False, 0, False),
        (True, 0, False),
    ],
)
def test_passed_is_derived_from_score(claimed, score, expected):
    assert SecurityReport(passed=claimed, score=score).passed is expected


def test_passed_may_be_omitted():
    assert SecurityReport(score=SECURITY_PASS_SCORE).passed is True
    assert SecurityReport(score=SECURITY_PASS_SCORE - 1).passed is False


def test_agent_json_output_is_validated_through_the_derivation():
    report = SecurityReport.model_validate_json('{"passed": true, "score": 30, "feedback": ["SQLi"]}')
    assert report.passed is False
    assert report.feedback == ["SQLi"]


def test_bands_are_ordered():
    assert SECURITY_CRITICAL_MAX_SCORE < SECURITY_HIGH_MAX_SCORE < SECURITY_PASS_SCORE


def test_prompt_states_the_bands_and_has_no_placeholders():
    from pydantic_agents.agents.security import security_prompt

    assert str(SECURITY_PASS_SCORE) in security_prompt
    assert str(SECURITY_HIGH_MAX_SCORE) in security_prompt
    assert str(SECURITY_CRITICAL_MAX_SCORE) in security_prompt
    assert "__" not in security_prompt


async def test_quality_gate_blocks_on_the_derived_verdict_not_the_claimed_one():
    from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

    outputs = {
        "sre": Artifact(agent_name="sre", files={"a.yaml": "x"}),
        "security": SecurityReport(passed=True, score=SECURITY_PASS_SCORE - 1, feedback=["weak hash"]),
    }
    provider = AsyncMock()

    async def run_agent(sem, name, agent, prompt, deps=None):  # noqa: ARG001
        return MagicMock(output=outputs[name])

    provider.run_agent.side_effect = run_agent
    with (
        patch("pydantic_agents.core.stages.quality_gate_stage.registry.get", return_value=MagicMock()),
        patch(
            "pydantic_agents.core.stages.quality_gate_stage.CodeValidator.run_all_async", AsyncMock(return_value=True)
        ),
        patch("pydantic_agents.core.context_engine.retrieve_context", return_value=""),
    ):
        result = await QualityGateStage(agent_provider=provider).run(
            PipelineState(run_id="r", stage="implementation"), PipelineConfig(project="p", requirement="r")
        )

    assert result.approved is False
    assert "weak hash" in result.feedback
    assert result.failure_report.error_type == "security"
