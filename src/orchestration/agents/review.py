from pydantic_ai import Agent

from orchestration.models.core import ReviewResult
from orchestration.models.factory import create_model
from orchestration.settings import settings

# Review-stage configuration.
REVIEW_BLUEPRINT_SUMMARY = "Event-driven hexagonal services, gRPC-enabled, CloudEvents-compliant."
REVIEW_BOUNDED_CONTEXTS = ["payments", "notifications", "users"]
REVIEW_STRICT_MODE = True


def _build_review_prompt() -> str:
    strict_clause = (
        "CRITICAL: Strict mode is ENABLED. Any warning must be treated as a hard failure."
        if REVIEW_STRICT_MODE
        else "NOTE: Strict mode is disabled. Provide warnings for minor infractions."
    )
    bounded = ", ".join(REVIEW_BOUNDED_CONTEXTS)
    return f"""You represent the final governance layer. Your role is to act as a **Qualified Security Assessor (QSA)**.
You analyze artifacts (Go code, Terraform, Test results) to ensure they meet production-readiness standards. You perform risk-based assessments of the entire system state before milestone approval.

{strict_clause}

### 1. Security (Zero-Trust)
* mTLS must be explicitly enabled everywhere.
* Absolutely NO hardcoded secrets, tokens, or sensitive data in any artifact.

### 2. Architecture Constraints
* Artifacts must perfectly match the approved Event-Driven Blueprint: {REVIEW_BLUEPRINT_SUMMARY}
* Artifacts must strictly adhere to these Bounded Contexts: {bounded}

### 3. Infrastructure (Terraform)
* NO 'Destroy' actions permitted in production plans.
* ALL modules must be version-pinned (no 'latest' or unversioned branches).

### Output Instructions
Populate your output exactly according to the requested schema. Provide specific line numbers or file names where violations occur to make your review actionable."""


review_agent = Agent(
    model=create_model(settings.resolve_model(settings.review_model)),
    system_prompt=_build_review_prompt(),
    output_type=ReviewResult,
    retries=5,
    defer_model_check=True,
)
