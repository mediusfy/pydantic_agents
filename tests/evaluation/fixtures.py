"""Shared test fixtures and mock LLM responses for deterministic testing."""

from dataclasses import dataclass
from typing import Any

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


@dataclass
class MockLLMResponse:
    data: Any
    output: Any | None = None

    def __post_init__(self):
        if self.output is None:
            self.output = self.data


def make_pipeline_state(requirement: str = "Build a payment service", **overrides: Any) -> PipelineState:
    return PipelineState(
        run_id="test-run-001",
        stage="init",
        requirement=requirement,
        **overrides,
    )


def make_user_story(
    story_id: str = "US-001",
    title: str = "Process Payment",
    gherkin: str = "Given a valid payment\nWhen processed\nThen PaymentSucceeded",
) -> UserStory:
    return UserStory(
        id=story_id,
        title=title,
        gherkin=gherkin,
        event_triggers=[
            EventTrigger(id="evt-1", name="PaymentSucceeded", description="Payment processed successfully"),
        ],
    )


def make_artifact(
    agent_name: str = "backend", files: dict[str, str] | None = None, story_id: str = "US-001"
) -> Artifact:
    if files is None:
        files = {
            "cmd/server/main.go": "package main\n\nfunc main() {}\n",
            "internal/domain/types.go": "package domain\n\ntype Payment struct {}\n",
        }
    return Artifact(
        agent_name=agent_name,
        files=files,
        metadata={"user_story_id": story_id, "language": "go", "framework": "chi"},
    )


def make_agent_advice(
    approved: bool = True, recs: list[str] | None = None, bottlenecks: list[str] | None = None
) -> AgentAdvice:
    return AgentAdvice(
        bottlenecks_identified=bottlenecks or [],
        recommendations=recs or ["Use hexagonal architecture", "Add OpenTelemetry tracing"],
        approved_to_proceed=approved,
    )


def make_review_result(approved: bool = True, score: int = 85, feedback: list[str] | None = None) -> ReviewResult:
    return ReviewResult(
        approved=approved,
        feedback=feedback or ["Code looks clean", "Minor: add more test coverage"],
        security_score=score,
        risk_level=RiskLevel.LOW if score >= 80 else RiskLevel.MEDIUM,
    )


def make_quality_gate_result(
    passed: bool = True, complexity: int = 30, feedback: list[str] | None = None
) -> QualityGateResult:
    return QualityGateResult(
        passed=passed,
        complexity_score=complexity,
        feedback=feedback or ["Simple idiomatic code", "No over-engineering detected"],
    )


def make_security_report(passed: bool = True, score: int = 90, feedback: list[str] | None = None) -> SecurityReport:
    return SecurityReport(
        passed=passed,
        score=score,
        feedback=feedback or ["No OWASP vulnerabilities found"],
    )


MOCK_BA_OUTPUT = MockLLMResponse(
    data=[
        make_user_story(
            "US-001",
            "Process Payment",
            "Given a payment request\nWhen processed\nThen PaymentSucceeded",
        ),
        make_user_story(
            "US-002",
            "Refund Payment",
            "Given a completed payment\nWhen refund requested\nThen payment reversed",
        ),
    ],
)

MOCK_ADVISORY_OUTPUT = MockLLMResponse(
    data=make_agent_advice(
        approved=True,
        recs=["Use event sourcing for payment history", "Implement idempotency for payment requests"],
        bottlenecks=["Payment gateway rate limiting needs circuit breaker"],
    ),
)

MOCK_BACKEND_OUTPUT = MockLLMResponse(
    data=make_artifact(
        "backend",
        {
            "cmd/server/main.go": "package main\n\nfunc main() {}\n",
            "internal/domain/types.go": "package domain\n\ntype Payment struct {\n  ID string\n}\n",
            "internal/domain/events.go": "package domain\n\ntype PaymentCreated struct{}\n",
            "internal/ports/repository.go": "package ports\n\ntype PaymentRepo interface{}\n",
            "internal/adapters/postgres_repo.go": "package postgres\n\ntype PgRepo struct{}\n",
        },
    ),
)

MOCK_QUALITY_GATE_OUTPUT = MockLLMResponse(
    data=make_quality_gate_result(passed=True, complexity=25),
)
