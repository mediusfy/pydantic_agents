"""The ``product_strategist`` agent (ADR-0001 / AP-77): mines requirement history and open platform findings for
customer-facing product/capability proposals.

Read-only: it has no write path at all. Its output is a written proposal report (``ProductStrategistReport``),
never code, never a ticket, never a direct repo/Jira write — a human/CTO decides what, if anything, to act on.
"""

import json
from typing import Any

from pydantic_ai import Agent, RunContext

from pydantic_agents.core import platform_findings, run_store
from pydantic_agents.core.models.core import ProductStrategistReport
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings

product_strategist_prompt = """You are the Product Strategist: you mine the history of requirements this platform
has already been asked to build, plus open platform findings, for customer-facing product or capability ideas —
patterns in what users keep asking for, recurring pain points, adjacent capabilities nobody has proposed yet.

You have two tools, both read-only:
- `recent_requirements`: recent pipeline requirement history (what users have asked this platform to build).
- `open_findings`: open platform_findings rows (regressions, flaky agents, latency drift) from the observer agent,
  which can suggest where users are already hitting friction.

You have no write tool. You never create a ticket, a PR, or any code — you only write a report. A human/CTO
reviews your proposals and decides what (if anything) to act on.

Workflow:
1. Call `recent_requirements` and `open_findings` to gather your evidence.
2. Look for patterns: repeated or adjacent requirements, friction implied by findings, capabilities that would
   plausibly serve more than one past requirement.
3. Produce 0-5 concrete proposals in `proposals` (title/rationale, with `evidence` quoting the specific
   requirement text or finding summaries that support it), and a short `narrative` summarizing your reasoning (or
   explaining why you found nothing proposal-worthy this run).
"""

product_strategist_agent = Agent(
    create_model(settings.resolve_model(settings.product_strategist_model)),
    system_prompt=product_strategist_prompt,
    output_type=ProductStrategistReport,
    defer_model_check=True,
    retries=settings.agent_retries,
)


@product_strategist_agent.tool
async def recent_requirements(ctx: RunContext[Any], limit: int = 50) -> str:
    """Recent pipeline requirement history (run_id, project, requirement, status, completed_at), newest first,
    as JSON."""
    rows = await run_store.recent_requirements(limit=limit)
    return json.dumps(rows, default=str)


@product_strategist_agent.tool
async def open_findings(ctx: RunContext[Any], limit: int = 50) -> str:
    """Open platform_findings rows (from the observer agent), newest first, as JSON."""
    rows = await platform_findings.list_findings(status="open", limit=limit)
    return json.dumps(rows, default=str)
