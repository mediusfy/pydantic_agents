from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext

from pydantic_agents.core.models.core import ReviewResult
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings

REVIEW_BLUEPRINT_SUMMARY = "Event-driven hexagonal services, gRPC-enabled, CloudEvents-compliant."


@dataclass
class ReviewCriteria:
    """Per-run review criteria, supplied by the pipeline (e.g. from the BA stage's bounded contexts)."""

    bounded_contexts: list[str] = field(default_factory=list)
    blueprint_summary: str = REVIEW_BLUEPRINT_SUMMARY
    strict: bool = True


def _build_review_prompt() -> str:
    return """You represent the final governance layer. Your role is to act as a **Qualified Security Assessor (QSA)**.
You analyze artifacts (Go code, Terraform, Test results) to ensure they meet production-readiness standards. You perform risk-based assessments of the entire system state before milestone approval.

### 1. Security (Zero-Trust)
* Service-to-service traffic must use mTLS.
* Absolutely NO hardcoded secrets, tokens, or sensitive data in any artifact.

### 2. Architecture Constraints
* Artifacts must match the approved blueprint and bounded contexts given under "Review Criteria" (if any).

### 3. Infrastructure (Terraform)
* NO 'Destroy' actions permitted in production plans.
* ALL modules must be version-pinned (no 'latest' or unversioned branches).

### Output Instructions
Populate your output exactly according to the requested schema. Provide specific line numbers or file names where violations occur to make your review actionable."""


review_agent = Agent(
    model=create_model(settings.resolve_model(settings.review_model)),
    system_prompt=_build_review_prompt(),
    output_type=ReviewResult,
    deps_type=ReviewCriteria,
    retries=settings.agent_retries,
    defer_model_check=True,
)


@review_agent.instructions
def _review_criteria(ctx: RunContext[ReviewCriteria | None]) -> str:
    """Render the per-run criteria; with none supplied the review stays generic and strict."""
    criteria = ctx.deps if isinstance(ctx.deps, ReviewCriteria) else ReviewCriteria()
    strict = (
        "CRITICAL: Strict mode is ENABLED. Any warning must be treated as a hard failure."
        if criteria.strict
        else "NOTE: Strict mode is disabled. Provide warnings for minor infractions."
    )
    lines = ["## Review Criteria", strict, f"- Approved blueprint: {criteria.blueprint_summary}"]
    if criteria.bounded_contexts:
        lines.append(f"- Artifacts must stay within these bounded contexts: {', '.join(criteria.bounded_contexts)}")
    return "\n".join(lines)
