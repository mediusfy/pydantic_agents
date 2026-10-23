from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from pydantic_agents.core.pipeline.context import ContextFrame


class EventTrigger(BaseModel):
    id: str
    name: str = Field(description="Name of the event")
    description: str


class UserStory(BaseModel):
    id: str
    title: str
    gherkin: str = Field(description="Given/When/Then acceptance criteria")
    event_triggers: list[EventTrigger] = Field(default_factory=list)


class Artifact(BaseModel):
    agent_name: str
    files: dict[str, str] = Field(description="Map of file paths to their contents")
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "user_story_id": "",
            "service_name": "",
            "language": "",
            "framework": "",
        }
    )


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ReviewResult(BaseModel):
    approved: bool
    feedback: list[str]
    security_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel


QUALITY_GATE_MAX_COMPLEXITY = 60


class QualityGateResult(BaseModel):
    passed: bool = Field(
        default=False,
        description=f"Derived in code from complexity_score <= {QUALITY_GATE_MAX_COMPLEXITY}; supplied value is ignored",
    )
    complexity_score: int = Field(ge=0, le=100, description="0 is simple, 100 is overly complex/over-engineered")
    feedback: list[str]

    @model_validator(mode="after")
    def _derive_passed(self) -> "QualityGateResult":
        # The verdict must not depend on the model keeping two fields consistent.
        self.passed = self.complexity_score <= QUALITY_GATE_MAX_COMPLEXITY
        return self


class AgentAdvice(BaseModel):
    bottlenecks_identified: list[str] = Field(
        default_factory=list, description="Potential issues or bottlenecks spotted"
    )
    recommendations: list[str] = Field(default_factory=list, description="Actionable advice for downstream agents")
    approved_to_proceed: bool = Field(
        default=False, description="Must be set explicitly; an omitted value blocks the pipeline"
    )


class FailureReport(BaseModel):
    source: str = Field(description="The stage or agent that produced the failure")
    error_type: str = Field(description="The category of failure (e.g., 'compilation', 'security', 'test')")
    trace: str = Field(description="The detailed error trace or feedback")
    relevant_files: list[str] = Field(default_factory=list)


class ActivityLog(BaseModel):
    pass_num: int = 0
    agent_name: str
    action: str
    summary: str
    timestamp: str
    artifact_ids: list[str] = Field(default_factory=list)


class SecurityReport(BaseModel):
    passed: bool
    score: int = Field(ge=0, le=100)
    feedback: list[str] = Field(default_factory=list)


class PipelineState(BaseModel):
    run_id: str
    stage: str
    requirement: str = ""
    attempt: int = 1
    refinement_attempts: int = 0
    user_stories: list[UserStory] = Field(default_factory=list)
    advisory_results: dict[str, str] = Field(default_factory=dict)
    cto_approved: bool = Field(default=True, description="CTO gate: true when approved_to_proceed")
    approved: bool = Field(default=False)
    artifacts: list[Artifact] = Field(default_factory=list)
    feedback: str | None = None
    failure_report: FailureReport | None = None
    errors: list[str] = Field(default_factory=list)
    active_context: ContextFrame | None = None
    context_history: list[ContextFrame] = Field(default_factory=list)
