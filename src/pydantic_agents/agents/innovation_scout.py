"""The ``innovation_scout`` agent (ADR-0001 / AP-77): scans for adoptable techniques and technologies relevant to
this stack, and proposes platform improvements as draft ADRs.

Read-only with respect to the rest of the platform: its only write path is ``propose_adr``, which persists a
*draft* ADR to the ``adrs`` vector store via the same underlying write ``mcp_server.save_adr`` uses. Drafts are
never auto-adopted — any follow-on implementation work requires an explicit human/CTO approval through the
existing ``core/cto_gate.py`` (``approve_cto_gate``) pattern, the same gate blueprint pipelines already go through
before implementation. This agent never creates implementation tickets, never triggers a pipeline run, and never
writes the CTO approval marker itself; it only drafts the ADR for a human to review and decide on.
"""

from typing import Any

from pydantic_ai import Agent, RunContext

from pydantic_agents.core.models.core import InnovationProposal, InnovationScoutReport
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings
from pydantic_agents.core.vector_db import vector_db

innovation_scout_prompt = """You are the Innovation Scout: you look for techniques, frameworks, and tools adoptable
by this platform (this repo), and write up draft Architecture Decision Records for a human/CTO to review. You are
not an implementer: you never write or change code, never create implementation tickets, and never approve your
own proposals.

You have two tools:
- `search_existing_adrs`: search past ADRs for related context, so you do not duplicate or contradict a past
  decision. Call this first, for every proposal, before drafting anything new.
- `propose_adr`: persist a draft ADR (status: Proposed) to the ADR store. This is your only write path. A draft
  is NOT an approval — any follow-on implementation ticket requires an explicit human/CTO approval through the
  existing CTO approval gate (`core/cto_gate.py`, `approve_cto_gate`) before anyone (not you) acts on it.

Workflow:
1. Call `search_existing_adrs` with a few relevant queries about the area you are scouting, to ground yourself in
   what has already been decided.
2. Research the specific technique/technology you are proposing (you may be given web research results, or asked
   to reason from what you already know) and form a concrete, scoped recommendation.
3. Call `propose_adr` with a unique `decision_id` and a short ADR-style `content` (Status: Proposed, Context,
   Decision, Consequences, Alternatives considered).
4. Summarize what you proposed (or why you proposed nothing this run) in `narrative`, and mirror each drafted
   proposal in `proposals` (decision_id/title/summary) so a caller reading only your output still sees it.

You never bypass the CTO approval gate, and you never create or trigger any implementation work yourself.
"""

innovation_scout_agent = Agent(
    create_model(settings.resolve_model(settings.innovation_scout_model)),
    system_prompt=innovation_scout_prompt,
    output_type=InnovationScoutReport,
    defer_model_check=True,
    retries=settings.agent_retries,
)


@innovation_scout_agent.tool
async def search_existing_adrs(ctx: RunContext[Any], query: str, n_results: int = 5) -> str:
    """Search past ADRs for related context, so a new proposal does not duplicate or contradict one."""
    results = await vector_db.async_search("adrs", query, n_results=n_results)
    if not results:
        return f"No ADRs found for query: {query}"
    return "Found past architectural context:\n" + "\n---\n".join(results)


@innovation_scout_agent.tool
async def propose_adr(ctx: RunContext[Any], decision_id: str, content: str) -> str:
    """Persist a draft ADR (status: Proposed) to the ADR store. This is the only write path this agent has; the
    draft requires explicit human/CTO approval (the existing CTO approval gate) before any follow-on
    implementation work is created — this tool never creates that follow-on work itself.
    """
    await vector_db.async_add_documents(
        "adrs", [content], [{"id": decision_id, "status": "proposed"}], [decision_id]
    )
    return f"Saved draft ADR {decision_id} (status: proposed, pending human/CTO approval)."


def build_report(proposals: list[InnovationProposal], narrative: str = "") -> InnovationScoutReport:
    """Assemble an ``InnovationScoutReport`` from drafted proposals; used by tests to check the output shape
    without an LLM call.
    """
    return InnovationScoutReport(proposals=proposals, narrative=narrative)
