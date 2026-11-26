from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext

from pydantic_agents.core.models.core import AgentAdvice
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings
from pydantic_agents.core.vector_db import vector_db


@dataclass
class OrgContext:
    tech_stack: list[str]
    compliance_tier: str
    monthly_cloud_budget: float


def default_org_context() -> OrgContext:
    """The OrgContext used when a caller passes none (``registry.run`` without deps: MCP, delegation, evals).

    Built from the ``ORG_TECH_STACK``, ``ORG_COMPLIANCE_TIER`` and ``ORG_MONTHLY_CLOUD_BUDGET`` settings. Their
    defaults are placeholders, not any organisation's real policy; the CTO approves or rejects against them, so set
    them per deployment.
    """
    return OrgContext(
        tech_stack=[s.strip() for s in settings.org_tech_stack.split(",") if s.strip()],
        compliance_tier=settings.org_compliance_tier,
        monthly_cloud_budget=settings.org_monthly_cloud_budget,
    )


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
- `adr_id`: identifier of the governing Architecture Decision Record when approving (e.g. `"ADR-0001"` from `search_existing_adrs`, or `"ADR-NEW"` when establishing a new baseline architecture decision). Required whenever `approved_to_proceed` is `true`; leave empty when rejecting.
- `findings`: typed architectural or risk findings (each with `id`, `source_agent="cto"`, `severity`, `description`, and `location`); any `CRITICAL` finding blocks approval.
- `bottlenecks_identified`: concrete scale, latency, cost, or coupling concerns
- `recommendations`: prioritized, actionable architectural guidance
- `architectural_constraints`: constraints that bind specific downstream agents
- If a requirement is fundamentally flawed or unscalable, flag it in `bottlenecks_identified` / `findings` and set `approved_to_proceed` to `false`.
"""

cto_agent = Agent(
    create_model(settings.resolve_model(settings.cto_model)),
    system_prompt=cto_prompt,
    output_type=AgentAdvice,
    deps_type=OrgContext,
    defer_model_check=True,
    retries=settings.agent_retries,
)


@cto_agent.output_validator
def _require_adr_id_on_approval(ctx: RunContext[OrgContext | None], advice: AgentAdvice) -> AgentAdvice:
    """Prompt a retry when the model attempted to approve without supplying adr_id."""
    if advice._approval_withdrawn_reason == "missing_adr_id":
        raise ModelRetry(
            "approved_to_proceed=true requires a non-empty adr_id citing the governing Architecture Decision Record "
            "(e.g. 'ADR-0001' from search_existing_adrs, or 'ADR-NEW' for a new baseline decision). "
            "Populate adr_id or set approved_to_proceed=false."
        )
    return advice


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


@cto_agent.tool
async def search_existing_adrs(ctx: RunContext[Any], query: str, n_results: int = 5) -> str:
    """Search past Architecture Decision Records (ADRs) for related context before approving an architecture."""
    try:
        results = await vector_db.async_search("adrs", query, n_results=max(1, min(n_results, 20)))
    except Exception as exc:  # best-effort, like the memory tools: an unreachable store must not abort the review
        return f"ADR search unavailable ({type(exc).__name__}); continue without past ADRs."
    if not results:
        return f"No ADRs found for query: {query}"
    return "Found past architectural context:\n" + "\n---\n".join(results)
