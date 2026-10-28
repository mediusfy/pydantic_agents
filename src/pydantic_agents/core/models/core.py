from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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


# A review cannot approve code it rates as a high or critical risk, or scores below the security pass band.
REVIEW_BLOCKING_RISKS = frozenset({RiskLevel.HIGH, RiskLevel.CRITICAL})
REVIEW_MIN_SECURITY_SCORE = 70


class ReviewResult(BaseModel):
    approved: bool
    feedback: list[str]
    security_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel

    @model_validator(mode="after")
    def _consistent_verdict(self) -> "ReviewResult":
        """Force a rejection when the risk level or the security score contradicts an approval."""
        if self.approved and (
            self.risk_level in REVIEW_BLOCKING_RISKS or self.security_score < REVIEW_MIN_SECURITY_SCORE
        ):
            self.approved = False
            self.feedback = [
                *self.feedback,
                f"Approval withdrawn: risk_level={self.risk_level.value}, security_score={self.security_score} "
                f"(approval needs risk below HIGH and a score of at least {REVIEW_MIN_SECURITY_SCORE}).",
            ]
        return self


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


class FindingSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# A CRITICAL finding blocks downstream approval (CTO review gate); no other severity does.
FINDING_BLOCKING_SEVERITIES = frozenset({FindingSeverity.CRITICAL})


class Finding(BaseModel):
    """A typed, versioned finding shared across the CTO Architecture Review and SecOps/FinOps Audit pipelines.

    `location` is required so a finding can be acted on without re-investigation; unreferenced findings
    (no file:line, no resource id) must be dropped by the orchestrator, not accepted here.
    """

    schema_version: Literal[1] = 1
    id: str
    source_agent: str = Field(description="Agent that raised the finding (e.g. security, devops, sre, finops)")
    severity: FindingSeverity
    description: str
    location: str = Field(min_length=1, description="file:line or a resource id the finding applies to")
    owner: str = Field(default="", description="Set by review when consolidating; empty until then")
    due_date: str = Field(default="", description="ISO date set by review when consolidating; empty until then")

    @field_validator("severity", mode="before")
    @classmethod
    def _upper_severity(cls, value: Any) -> Any:
        """Models write "critical" as often as "CRITICAL"; either must reach the CRITICAL gate."""
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("location", mode="before")
    @classmethod
    def _strip_location(cls, value: Any) -> Any:
        """A blank location is no location: it must fail min_length, not pass as "   "."""
        return value.strip() if isinstance(value, str) else value


class AgentAdvice(BaseModel):
    bottlenecks_identified: list[str] = Field(
        default_factory=list, description="Potential issues or bottlenecks spotted"
    )
    recommendations: list[str] = Field(default_factory=list, description="Actionable advice for downstream agents")
    architectural_constraints: list[ArchitecturalConstraint] = Field(
        default_factory=list, description="Constraints that bind specific downstream agents"
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="Typed findings merged from CTO review's members (ba/security/database/sre/finops)",
    )
    adr_id: str = Field(
        default="", description="The ADR this decision is recorded under; approval requires one (e.g. 0002)"
    )
    approved_to_proceed: bool = Field(
        default=False, description="Must be set explicitly; an omitted value blocks the pipeline"
    )

    @model_validator(mode="after")
    def _block_on_critical_finding(self) -> "AgentAdvice":
        """Overrides the supplied approval: no approval while a CRITICAL finding is open, or without an adr_id."""
        if self.approved_to_proceed and any(f.severity in FINDING_BLOCKING_SEVERITIES for f in self.findings):
            self.approved_to_proceed = False
        if self.approved_to_proceed and not self.adr_id.strip():
            self.approved_to_proceed = False
        return self


class CriterionResult(BaseModel):
    criterion_id: str = Field(description="Acceptance-criterion id the Business Analyst assigned")
    passed: bool
    detail: str = Field(default="", description="Failure detail, or any other note about the run")


class TestRunReport(BaseModel):
    """A versioned test-execution report produced by `qa_executor` for the QA Engineer pipeline."""

    __test__ = False  # pytest: not a test class despite the name

    schema_version: Literal[1] = 1
    criterion_results: list[CriterionResult] = Field(default_factory=list)
    coverage_percent: float = Field(ge=0, le=100, default=0.0)
    unexecuted_criterion_ids: list[str] = Field(
        default_factory=list, description="Criterion ids qa_executor could not execute; reason goes in feedback"
    )
    feedback: list[str] = Field(default_factory=list)
    all_passed: bool = Field(
        default=False,
        description="Derived in code: true iff every result passed and nothing is unexecuted; supplied value ignored",
    )

    @model_validator(mode="after")
    def _derive_verdict(self) -> "TestRunReport":
        """Overrides (does not validate) the supplied verdict: results and unexecuted ids are the source of truth."""
        self.all_passed = (
            bool(self.criterion_results)
            and all(r.passed for r in self.criterion_results)
            and not self.unexecuted_criterion_ids
        )
        return self


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


# Shared field description for the platform-intelligence agents' report narratives.
_SCAN_DIGEST = "Short human-readable digest of the scan"


class InnovationProposal(BaseModel):
    """One draft ADR the innovation_scout agent wrote; mirrors what it persisted via ``propose_adr``."""

    decision_id: str
    title: str
    summary: str


class InnovationScoutReport(BaseModel):
    """Structured output of the innovation_scout agent: what it drafted, in its own words. Drafts are proposals
    only; none are auto-adopted (see ADR-0001's guardrails) — a human/CTO approves via the existing CTO gate
    before any follow-on implementation ticket is created.
    """

    proposals: list[InnovationProposal] = Field(default_factory=list)
    narrative: str = Field(default="", description=_SCAN_DIGEST)


class ObserverFinding(BaseModel):
    """One cross-run signal the observer agent surfaced; mirrors a `platform_findings` row."""

    kind: str = Field(description="regression, flaky, pattern, or health")
    severity: str = Field(description="low, medium, high, or critical")
    agent: str | None = Field(default=None, description="Affected agent name, when the finding is agent-specific")
    summary: str


class ObserverReport(BaseModel):
    """Structured output of the observer agent: what it found, in its own words."""

    findings: list[ObserverFinding] = Field(default_factory=list)
    narrative: str = Field(default="", description=_SCAN_DIGEST)


class RemediationRecord(BaseModel):
    """One action the sre_autopilot agent took or escalated, from a single health/remediation scan."""

    action: str | None = Field(default=None, description="Allowlisted action name, or null when escalated")
    auto_remediated: bool
    summary: str
    finding_id: str | None = Field(default=None, description="platform_findings row id, when one was written")


class SreAutopilotReport(BaseModel):
    """Structured output of the sre_autopilot agent: what it checked, fixed, and escalated."""

    records: list[RemediationRecord] = Field(default_factory=list)
    narrative: str = Field(default="", description=_SCAN_DIGEST)


class ProductProposal(BaseModel):
    """One customer-facing product/capability idea the product_strategist agent surfaced."""

    title: str
    rationale: str
    evidence: list[str] = Field(default_factory=list, description="Requirement text or finding summaries cited")


class ProductStrategistReport(BaseModel):
    """Structured output of the product_strategist agent: a written proposal report, never code or tickets.
    A human/CTO reviews and decides what (if anything) to act on (ADR-0001's guardrails).
    """

    proposals: list[ProductProposal] = Field(default_factory=list)
    narrative: str = Field(default="", description="Short human-readable digest of the analysis")


class PipelineState(BaseModel):
    run_id: str
    stage: str
    requirement: str = ""
    attempt: int = 1
    refinement_attempts: int = 0
    user_stories: list[UserStory] = Field(default_factory=list)
    advisory_results: dict[str, str] = Field(default_factory=dict)
    cto_approved: bool = Field(
        default=False, description="CTO gate: true only after approved_to_proceed or an explicit human override"
    )
    approved: bool = Field(default=False)
    artifacts: list[Artifact] = Field(default_factory=list)
    feedback: str | None = None
    failure_report: FailureReport | None = None
    errors: list[str] = Field(default_factory=list)
    active_context: ContextFrame | None = None
    context_history: list[ContextFrame] = Field(default_factory=list)


class ClaimCheck(BaseModel):
    claim: str = Field(description="The claim, as the producing agent stated it")
    supported: bool = Field(description="True iff the supplied evidence backs the claim")
    evidence: str = Field(default="", description="The evidence that backs the claim, or why none does")


class CriticReport(BaseModel):
    claims: list[ClaimCheck] = Field(default_factory=list)
    unsupported_count: int = Field(
        default=0, description="Derived in code from claims (count with supported=false); supplied value ignored"
    )
    passed: bool = Field(
        default=False, description="Derived in code: true iff unsupported_count == 0; supplied value ignored"
    )
    feedback: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_verdict(self) -> "CriticReport":
        """Overrides (does not validate) the supplied counts: the claim list is the single source of truth."""
        self.unsupported_count = sum(1 for check in self.claims if not check.supported)
        self.passed = self.unsupported_count == 0
        return self


ContractDriftKind = Literal["missing_endpoint", "extra_endpoint", "type_mismatch", "breaking_change"]
# Drift kinds that fail the contract check; the other kinds are reported but do not block.
CONTRACT_BLOCKING_KINDS: frozenset[str] = frozenset({"breaking_change", "type_mismatch"})


class ContractDrift(BaseModel):
    kind: ContractDriftKind
    location: str = Field(description="file:line of the drift (spec, handler or frontend type)")
    detail: str = Field(description="What differs between the spec and the implementation")


class ContractReport(BaseModel):
    drifts: list[ContractDrift] = Field(default_factory=list)
    passed: bool = Field(
        default=False,
        description=f"Derived in code: true iff no drift has a kind in {sorted(CONTRACT_BLOCKING_KINDS)}; "
        "supplied value ignored",
    )
    feedback: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_passed(self) -> "ContractReport":
        """Overrides (does not validate) the supplied `passed`: the drift kinds are the single source of truth."""
        self.passed = not any(drift.kind in CONTRACT_BLOCKING_KINDS for drift in self.drifts)
        return self


class CostLine(BaseModel):
    item: str = Field(description="What was paid for (model group, key alias, cloud resource)")
    amount_usd: float = Field(ge=0, description="Spend for the item in USD, never negative")
    period: str = Field(description="The period the amount covers (e.g. '2026-10-20..2026-10-27', 'monthly')")
    source: str = Field(description="Where the number came from (tool output, invoice, estimate)")


class CostReport(BaseModel):
    lines: list[CostLine] = Field(default_factory=list)
    total_usd: float = Field(default=0.0, description="Derived in code as the sum of lines; supplied value ignored")
    budget_usd: float | None = Field(default=None, description="The budget the total is compared against, if any")
    over_budget: bool = Field(
        default=False,
        description="Derived in code: true iff a budget is set and total_usd exceeds it; supplied value ignored",
    )
    recommendations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_totals(self) -> "CostReport":
        """Overrides (does not validate) the supplied total and verdict: the lines are the single source of truth."""
        self.total_usd = sum(line.amount_usd for line in self.lines)
        self.over_budget = self.budget_usd is not None and self.total_usd > self.budget_usd
        return self
