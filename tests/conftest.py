from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_agents.core.models.core import (
    AgentAdvice,
    Artifact,
    EventTrigger,
    PipelineState,
    QualityGateResult,
    ReviewResult,
    RiskLevel,
    SecurityReport,
    UserStory,
)


@pytest.fixture
def sample_user_story() -> UserStory:
    return UserStory(
        id="US-001",
        title="Process Payment",
        gherkin="Given a valid payment request\nWhen the system processes it\nThen it emits PaymentSucceeded",
        event_triggers=[
            EventTrigger(id="evt-1", name="PaymentSucceeded", description="Payment processed successfully"),
        ],
    )


@pytest.fixture
def sample_user_stories(sample_user_story) -> list[UserStory]:
    return [sample_user_story]


@pytest.fixture
def sample_artifact() -> Artifact:
    return Artifact(
        agent_name="backend",
        files={"cmd/server/main.go": "package main\nfunc main() {}\n"},
        metadata={"user_story_id": "US-001", "language": "go", "service_name": "payment", "framework": "net/http"},
    )


@pytest.fixture
def sample_artifacts(sample_artifact) -> list[Artifact]:
    return [sample_artifact]


@pytest.fixture
def sample_pipeline_state() -> PipelineState:
    return PipelineState(
        run_id="test-run-001",
        stage="init",
        attempt=1,
    )


@pytest.fixture
def sample_review_result_approved() -> ReviewResult:
    return ReviewResult(
        approved=True,
        feedback=[],
        security_score=85,
        risk_level=RiskLevel.LOW,
    )


@pytest.fixture
def sample_review_result_rejected() -> ReviewResult:
    return ReviewResult(
        approved=False,
        feedback=["Missing error handling", "No health check endpoint"],
        security_score=45,
        risk_level=RiskLevel.HIGH,
    )


@pytest.fixture
def sample_quality_gate_passed() -> QualityGateResult:
    return QualityGateResult(
        passed=True,
        complexity_score=25,
        feedback=[],
    )


@pytest.fixture
def sample_quality_gate_failed() -> QualityGateResult:
    return QualityGateResult(
        passed=False,
        complexity_score=75,
        feedback=["Over-engineered: unnecessary abstraction layer"],
    )


@pytest.fixture
def sample_security_report_passed() -> SecurityReport:
    return SecurityReport(
        passed=True,
        score=90,
        feedback=[],
    )


@pytest.fixture
def sample_agent_advice() -> AgentAdvice:
    return AgentAdvice(
        bottlenecks_identified=["Single point of failure in event bus"],
        recommendations=["Use NATS with JetStream for persistence"],
        approved_to_proceed=True,
    )


@pytest.fixture
def mock_agent():
    """Returns a configured MagicMock that mimics a pydantic_ai Agent."""
    agent = MagicMock()
    agent.run = AsyncMock()
    return agent


@pytest.fixture
def mock_agent_result(mock_agent, sample_artifact):
    """Configures mock_agent.run to return sample_artifact."""
    mock_result = MagicMock()
    mock_result.output = sample_artifact
    mock_agent.run.return_value = mock_result
    return mock_agent


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    output_dir = tmp_path / "generated-app"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_checkpoint_dir(tmp_path, monkeypatch) -> Path:
    ckpt_dir = tmp_path / ".claw-checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("pydantic_agents.core.checkpoint.settings.checkpoint_dir", ckpt_dir)
    return ckpt_dir


@pytest.fixture
def temp_telemetry_file(tmp_path) -> str:
    telemetry_file = tmp_path / "telemetry.jsonl"
    return str(telemetry_file)
