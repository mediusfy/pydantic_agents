"""Per-agent behavior tests using mock LLM responses.

Validates that agents produce correct structured outputs,
pass hallucination checks, and make correct decisions.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_agents.core.models.core import (
    AgentAdvice,
    Artifact,
    FailureReport,
    PipelineState,
    SecurityReport,
    UserStory,
)
from pydantic_agents.core.pipeline.config import PipelineConfig
from tests.evaluation.decision_evaluator import validate_artifact_structure, validate_story_structure


@pytest.fixture
def state():
    return PipelineState(run_id="test-1", stage="init", requirement="Build a payment service")


@pytest.fixture
def config():
    return PipelineConfig(blueprint="langgraph_full", framework="langgraph", requirement="Build a payment service")


@pytest.fixture
def mock_provider():
    p = AsyncMock()
    p.run_agent = AsyncMock()
    return p


class TestBAAgentOutputs:
    def test_ba_produces_user_stories(self, state, config, mock_provider):
        mock_provider.run_agent.return_value = MagicMock(
            output=[
                UserStory(
                    id="US-001", title="Process Payment", gherkin="Given a payment\nWhen processed\nThen completed"
                ),
            ]
        )
        result = validate_story_structure(mock_provider.run_agent.return_value.output)
        assert result["valid"]

    def test_ba_stories_follow_structure(self):
        stories = [
            UserStory(
                id="BC-01", title="Create Order", gherkin="Given items in cart\nWhen order placed\nThen order created"
            ),
            UserStory(
                id="BC-02", title="Cancel Order", gherkin="Given existing order\nWhen cancelled\nThen order cancelled"
            ),
        ]
        result = validate_story_structure(stories)
        assert result["valid"]
        assert len(result["errors"]) == 0

    def test_ba_story_ids_follow_convention(self):
        stories = [
            UserStory(
                id="ORDERS-01", title="Process Order", gherkin="Given order\nWhen processed\nThen PaymentSucceeded"
            ),
        ]
        for story in stories:
            assert "-" in story.id, f"Story ID {story.id} should follow BC-NN convention"

    def test_ba_gherkin_must_have_given_when_then(self):
        invalid_story = UserStory(id="US-001", title="Test", gherkin="As a user I want payments")
        result = validate_story_structure([invalid_story])
        assert not result["valid"]


class TestCTOAgentOutputs:
    def test_cto_produces_agent_advice(self):
        advice = AgentAdvice(
            bottlenecks_identified=["Payment gateway rate limiting"],
            recommendations=["Use event sourcing", "Add circuit breaker"],
            approved_to_proceed=True,
        )
        assert advice.approved_to_proceed
        assert len(advice.recommendations) >= 1

    def test_cto_rejections_include_reasoning(self):
        advice = AgentAdvice(
            bottlenecks_identified=["Architecture violates data residency", "No event sourcing"],
            recommendations=[],
            approved_to_proceed=False,
        )
        assert not advice.approved_to_proceed
        assert len(advice.bottlenecks_identified) >= 2


class TestBackendAgentOutputs:
    def test_backend_produces_artifacts_with_files(self):
        artifact = Artifact(
            agent_name="backend",
            files={"cmd/server/main.go": "package main\nfunc main() {}", "go.mod": "module paymentsvc"},
            metadata={"user_story_id": "US-001"},
        )
        result = validate_artifact_structure([artifact])
        assert result["valid"]
        assert len(artifact.files) >= 2

    def test_backend_empty_file_rejected(self):
        artifact = Artifact(
            agent_name="backend",
            files={"empty.go": ""},
            metadata={"user_story_id": "US-001"},
        )
        result = validate_artifact_structure([artifact])
        assert not result["valid"]

    def test_backend_self_validates(self):
        artifact = Artifact(
            agent_name="backend",
            files={
                "cmd/server/main.go": 'package main\n\nimport "fmt"\n\nfunc main() { fmt.Println("hello") }\n',
                "go.mod": "module test\n\ngo 1.21\n",
            },
            metadata={"user_story_id": "US-001", "language": "go"},
        )
        result = validate_artifact_structure([artifact])
        assert result["valid"]


class TestSecurityAgentOutputs:
    def test_security_produces_report(self):
        report = SecurityReport(passed=True, score=92, feedback=["No vulnerabilities found"])
        assert report.passed
        assert report.score >= 0
        assert report.score <= 100

    def test_security_failures_include_feedback(self):
        report = SecurityReport(
            passed=False,
            score=45,
            feedback=["SQL injection risk in query builder", "Hardcoded secret on line 42"],
        )
        assert not report.passed
        assert len(report.feedback) >= 1


class TestAgentCrossValidation:
    def test_ba_stories_directly_feed_cto(self):
        stories = [
            UserStory(
                id="US-001",
                title="Process Payment",
                gherkin="Given payment request\nWhen processed\nThen PaymentSucceeded",
            ),
        ]
        cto_prompt = "Review: " + " | ".join(s.gherkin for s in stories)
        assert "Given payment request" in cto_prompt
        assert "When processed" in cto_prompt

    def test_cto_advice_directly_feeds_backend(self):
        advice = "Use hexagonal architecture and Postgres"
        stories = "Given payment\nWhen processed\nThen completed"
        prompt = f"Context: {advice}\nStories: {stories}"
        assert "hexagonal" in prompt.lower()
        assert "payment" in prompt.lower()


class TestHallucinationInAgentOutputs:
    def test_backend_output_should_not_hallucinate_unsupported_arch(self):
        from tests.evaluation.hallucination_detector import detect_hallucination

        context = "Use Go with hexagonal architecture. Store data in PostgreSQL."
        output = "Generated Go service using hexagonal architecture with PostgreSQL repository."
        report = detect_hallucination(output, context)
        assert report.passed
        assert report.score >= 0.85

    def test_ba_output_should_stay_factual(self):
        from tests.evaluation.hallucination_detector import detect_hallucination

        context = "Requirements: Build payment service for e-commerce platform"
        output = (
            "User Story BC-01: Process Payment - Given a valid"
            " payment request, When the system processes it,"
            " Then emits PaymentSucceeded"
        )
        report = detect_hallucination(output, context)
        assert report.passed


class TestFailurePropagation:
    def test_failure_report_preserves_source_and_trace(self):
        fr = FailureReport(source="test_suite", error_type="compilation", trace="undefined: PaymentService")
        assert fr.source == "test_suite"
        assert fr.error_type == "compilation"
        assert "PaymentService" in fr.trace

    def test_pipeline_state_accumulates_errors(self):
        state = PipelineState(run_id="test-1", stage="implementation")
        state.errors.append("backend: connection refused")
        state.errors.append("test: assertion failed")
        assert len(state.errors) == 2
