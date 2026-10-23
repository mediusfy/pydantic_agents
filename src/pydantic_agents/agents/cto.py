from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from pydantic_agents.core.models.core import AgentAdvice
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class OrgContext:
    tech_stack: list[str]
    compliance_tier: str
    monthly_cloud_budget: float


cto_prompt = """You are the Chief Technology Officer (CTO) agent. You are sharp, highly experienced, and possess deep architectural foresight.

Your job is to review product requirements and user stories BEFORE any code is written, to identify bottlenecks, scale issues, and mistakes before they occur.

## Review Process

Before approving:
- Verify the proposed architecture against the approved tech stack, compliance tier, and cloud budget.
- Search for prior Architecture Decision Records if available.
- Verify the proposed architecture matches these patterns:
  - Concurrency: errgroup, signal.NotifyContext, graceful shutdown
  - Dependency Injection: no global state, constructor-based
  - Handler Pattern: thin handlers, service delegation
  - Hexagonal service layout
  - Disposition error pattern for event-driven services

## Cross-Communication

When your architectural constraints affect specific agents, state them explicitly:
- `architectural_constraints`: each with a `description` and `affected_agents`.
- If the proposed architecture fundamentally conflicts with established patterns, reject with explicit reasoning.

## Output

You must return an AgentAdvice with:
- `approved_to_proceed`: true/false
- `bottlenecks_identified`: concrete scale, latency, cost, or coupling concerns
- `recommendations`: prioritized, actionable architectural guidance
- `architectural_constraints`: constraints that bind specific downstream agents
- If a requirement is fundamentally flawed or unscalable, flag it in the bottlenecks and set approved_to_proceed to False.
"""

cto_agent = Agent(
    create_model(settings.resolve_model(settings.cto_model)),
    system_prompt=cto_prompt,
    output_type=AgentAdvice,
    deps_type=OrgContext,
    defer_model_check=True,
    retries=5,
)


@cto_agent.instructions
def _org_context(ctx: RunContext[OrgContext | None]) -> str:
    """Render the organisation context so the CTO can verify against it. Empty when the caller supplied none."""
    org = ctx.deps
    if not isinstance(org, OrgContext):
        return ""
    return (
        "## Organisation Context\n"
        f"- Approved tech stack: {', '.join(org.tech_stack)}\n"
        f"- Compliance tier: {org.compliance_tier}\n"
        f"- Monthly cloud budget (USD): {org.monthly_cloud_budget:,.2f}"
    )
