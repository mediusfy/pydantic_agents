from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from pydantic_agents.core.pipeline.context import ContextFrame


class EventTrigger(BaseModel):
    id: str
    name: str = Field(description="Name of the event, past-tense and namespaced (e.g. orders.OrderPlaced)")
    description: str
    owner: str = Field(default="", description="Bounded context that owns and emits the event")
    consumers: list[str] = Field(default_factory=list, description="Contexts/services that consume the event")
    payload_schema: str = Field(default="", description="Sketch of the event payload (field names and types)")


class UserStory(BaseModel):
    id: str
    title: str
    statement: str = Field(default="", description="'As a ... I want ... so that ...' statement")
    bounded_context: str = Field(default="", description="Bounded context the story belongs to")
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
        description=f"Derived in code from complexity_score <= {QUALITY_GATE_MAX_COMPLEXITY}; supplied value ignored",
    )
    complexity_score: int = Field(ge=0, le=100, description="0 is simple, 100 is overly complex/over-engineered")
    feedback: list[str]

    @model_validator(mode="after")
    def _derive_passed(self) -> "QualityGateResult":
        # The verdict must not depend on the model keeping two fields consistent.
        self.passed = self.complexity_score <= QUALITY_GATE_MAX_COMPLEXITY
        return self


class ArchitecturalConstraint(BaseModel):
    description: str
    affected_agents: list[str] = Field(default_factory=list, description="Agents that must honour the constraint")


class AgentAdvice(BaseModel):
    bottlenecks_identified: list[str] = Field(
        default_factory=list, description="Potential issues or bottlenecks spotted"
    )
    recommendations: list[str] = Field(default_factory=list, description="Actionable advice for downstream agents")
    architectural_constraints: list[ArchitecturalConstraint] = Field(
        default_factory=list, description="Constraints that bind specific downstream agents"
    )
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


# Security score bands (100 = clean). The audit passes iff the score is at least SECURITY_PASS_SCORE; the agent's
# prompt asks it to put critical findings at or below SECURITY_CRITICAL_MAX_SCORE and high findings at or below
# SECURITY_HIGH_MAX_SCORE, so "no critical/high findings" and "score >= SECURITY_PASS_SCORE" mean the same thing.
SECURITY_PASS_SCORE = 70
SECURITY_HIGH_MAX_SCORE = SECURITY_PASS_SCORE - 1
# Fixed on purpose: critical findings sit well below the high band (40..SECURITY_HIGH_MAX_SCORE) whatever the pass
# score is, so moving SECURITY_PASS_SCORE widens or narrows only the high band and never leaves a gap.
SECURITY_CRITICAL_MAX_SCORE = 39


class SecurityReport(BaseModel):
    passed: bool = Field(
        default=False,
        description=f"Derived in code from score >= {SECURITY_PASS_SCORE}; supplied value ignored",
    )
    score: int = Field(ge=0, le=100, description="0-100, 100 is clean; see the security score bands")
    feedback: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_passed(self) -> "SecurityReport":
        """Overrides (does not validate) the supplied `passed`: the score is the single source of truth."""
        # The model reported both fields and they could disagree (passed=false with score 55 and no high findings).
        self.passed = self.score >= SECURITY_PASS_SCORE
        return self


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
