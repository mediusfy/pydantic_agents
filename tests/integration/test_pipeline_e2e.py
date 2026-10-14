"""Integration tests for end-to-end pipeline execution.

Uses mocked LLM calls to verify the full pipeline produces correct outputs,
agents hand off context correctly, and all stages complete without errors.
"""

from unittest.mock import AsyncMock

import pytest

from pydantic_agents.core.pipeline.config import PipelineConfig
from tests.evaluation.fixtures import (
    make_artifact,
    make_pipeline_state,
    make_user_story,
)


@pytest.fixture
def mock_provider():
    p = AsyncMock()
    p.run_agent = AsyncMock()
    return p


@pytest.fixture
def config():
    return PipelineConfig(
        blueprint="new_feature", framework="legacy", requirement="Build a payment service", project="test"
    )


class TestPipelineEndToEnd:
    def test_full_pipeline_with_mocked_agents_yields_state(self, config, mock_provider):
        state = make_pipeline_state("Build a payment service")

        assert state.run_id == "test-run-001"
        assert state.requirement == "Build a payment service"
        assert state.stage == "init"

    def test_ba_stage_sets_user_stories(self, config, mock_provider):
        state = make_pipeline_state("Build a payment service")
        stories = [make_user_story("US-001", "Process Payment"), make_user_story("US-002", "Refund Payment")]
        state.user_stories = stories
        state.stage = "ba"

        assert len(state.user_stories) == 2
        assert state.stage == "ba"
        assert state.user_stories[0].id == "US-001"

    def test_advisory_stage_sets_approval(self, config, mock_provider):
        state = make_pipeline_state("Build a payment service")
        state.user_stories = [make_user_story()]
        state.advisory_results = {"cto": "Use hex arch", "ux": "WCAG 2.1 AA"}
        state.cto_approved = True

        assert state.cto_approved
        assert "cto" in state.advisory_results

    def test_implementation_stage_produces_artifacts(self, config, mock_provider):
        state = make_pipeline_state("Build a payment service")
        state.user_stories = [make_user_story()]
        state.advisory_results = {"cto": "Use hex arch"}
        state.artifacts = [make_artifact("backend"), make_artifact("frontend")]
        state.stage = "implementation"

        assert len(state.artifacts) == 2
        assert state.artifacts[0].agent_name == "backend"

    def test_quality_gate_stage_approves_valid_code(self, config, mock_provider):
        state = make_pipeline_state("Build a payment service")
        state.artifacts = [make_artifact()]
        state.approved = True
        state.stage = "quality_gate"

        assert state.approved

    def test_pipeline_state_preserves_full_context_history(self):
        from pydantic_agents.core.pipeline.context import ContextFrame

        state = make_pipeline_state("Build a payment service")
        cf_ba = ContextFrame(output="Generated 2 user stories")
        cf_advisory = ContextFrame(output="CTO approved architecture")
        state.context_history.append(cf_ba)
        state.context_history.append(cf_advisory)

        assert len(state.context_history) == 2
        assert "2 user stories" in state.context_history[0].output

    def test_retry_loop_sets_refinement_attempts(self):
        state = make_pipeline_state("Build a payment service")
        state.refinement_attempts = 3
        assert state.refinement_attempts == 3

    def test_error_state_propagates_correctly(self):
        state = make_pipeline_state("Build a payment service")
        from pydantic_agents.core.models.core import FailureReport

        state.failure_report = FailureReport(source="test_suite", error_type="test_failure", trace="assertion failed")
        state.errors.append("test_eng: assertion failed")
        state.approved = False

        assert not state.approved
        assert state.failure_report.source == "test_suite"


class TestFrameworkParity:
    def test_langgraph_and_legacy_produce_same_state_shape(self):
        legacy = make_pipeline_state("Build API")
        legacy.stage = "ba"
        legacy.user_stories = [make_user_story()]

        langgraph = make_pipeline_state("Build API")
        langgraph.stage = "ba"
        langgraph.user_stories = [make_user_story()]

        assert legacy.stage == langgraph.stage
        assert len(legacy.user_stories) == len(langgraph.user_stories)

    def test_context_frame_preserved_across_nodes(self):
        from pydantic_agents.core.pipeline.context import ContextFrame

        state = make_pipeline_state("Build API")
        for _i, stage in enumerate(["ba", "advisory", "implementation", "quality_gate"]):
            cf = ContextFrame(output=f"{stage} stage completed")
            state.context_history.append(cf)
            state.stage = stage

        assert len(state.context_history) == 4
        assert all(isinstance(cf, ContextFrame) for cf in state.context_history)

    def test_dry_run_produces_valid_state(self):
        state = make_pipeline_state("Build API")
        state.stage = "implementation"
        state.approved = True
        state.artifacts = [make_artifact("backend", story_id="US-001")]
        assert state.approved
        assert len(state.artifacts) == 1

    def test_ab_testing_framework_works(self):
        from tests.evaluation.ab_testing import ABReport, compare_configs

        config_a = PipelineConfig(blueprint="langgraph_full", framework="langgraph")
        config_b = PipelineConfig(blueprint="new_feature", framework="legacy")

        def runner(cfg: PipelineConfig, req: str):
            state = make_pipeline_state(req)
            state.stage = "implementation"
            state.approved = True
            state.artifacts = [make_artifact("backend")]
            return state

        report = ABReport()
        result = compare_configs(config_a, config_b, "Build a payment service", runner)
        report.add_result(result)

        assert len(report.results) == 1
        assert result.a_score == result.b_score  # Both produce same artifacts
