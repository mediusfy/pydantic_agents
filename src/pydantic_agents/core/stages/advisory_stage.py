import asyncio
import logging

from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.registry import registry
from pydantic_agents.core.session_recorder import recorder
from pydantic_agents.core.stages.base import StageRunner

logger = logging.getLogger(__name__)


class AdvisoryStage(StageRunner):
    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        logger.info("--- Running CTO & UX Advisory Stage ---")

        if config.dry_run:
            logger.info("DRY RUN: Skipping Advisory phase.")
            state.advisory_results = {}
            state.cto_approved = True
            state.stage = "advisory"
            return state

        user_stories = state.user_stories or []
        story_context = "\n".join(f"- {us.id}: {us.title}\n  {us.gherkin}" for us in user_stories)

        cto_agent = registry.get("cto")
        ux_agent = registry.get("ux")
        semaphore = asyncio.Semaphore(10)

        from pydantic_agents.core.adapters.workspace import WorkspaceContext

        workspace = WorkspaceContext()
        md_context = workspace.get_markdown_context()

        from pydantic_agents.agents.cto import OrgContext

        org_ctx = OrgContext(
            tech_stack=["Go", "Vue", "PostgreSQL", "Redis"], compliance_tier="Standard", monthly_cloud_budget=1000.0
        )

        from pydantic_agents.core.context_engine import retrieve_context

        cto_ctx = retrieve_context("cto", story_context)
        ux_ctx = retrieve_context("ux", story_context)

        cto_prompt = f"Review these user stories and provide architectural advice:\n{story_context}"
        if cto_ctx:
            cto_prompt += f"\n\nRetrieved Architectural Context/ADRs/Rules:\n{cto_ctx}"

        ux_prompt = f"Review these user stories and provide UX/UI advice:\n{story_context}"
        if ux_ctx:
            ux_prompt += f"\n\nRetrieved UX Context/Rules:\n{ux_ctx}"

        if md_context:
            cto_prompt += f"\n\nAdhere to the following local guidelines:\n{md_context}"
            ux_prompt += f"\n\nAdhere to the following local guidelines:\n{md_context}"

        async def _run_cto():
            async with recorder.record_span(
                run_id=state.run_id,
                agent_name="cto",
                pipeline_stage="advisory",
                prompt=cto_prompt,
                context_summary=f"{len(user_stories)} stories",
            ) as rec:
                result = await self.provider.run_agent(semaphore, "cto", cto_agent, cto_prompt, deps=org_ctx)
                rec.set_output(result)
                return result

        async def _run_ux():
            async with recorder.record_span(
                run_id=state.run_id,
                agent_name="ux",
                pipeline_stage="advisory",
                prompt=ux_prompt,
                context_summary=f"{len(user_stories)} stories",
            ) as rec:
                result = await self.provider.run_agent(semaphore, "ux", ux_agent, ux_prompt)
                rec.set_output(result)
                return result

        advisory_results = await asyncio.gather(_run_cto(), _run_ux(), return_exceptions=True)

        state.advisory_results = {}
        state.cto_approved = True

        if not isinstance(advisory_results[0], BaseException):
            cto_data = advisory_results[0].output
            recs = cto_data.recommendations or []
            for constraint in cto_data.architectural_constraints:
                agents = ", ".join(constraint.affected_agents) or "all agents"
                recs.append(f"[constraint: {agents}] {constraint.description}")
            state.advisory_results["cto"] = "CTO Guidelines:\n" + "\n".join(recs) if recs else "CTO Guidelines: (none)"
            bottlenecks = getattr(cto_data, "bottlenecks_identified", [])
            if bottlenecks:
                state.advisory_results["cto_blockers"] = "\n".join(bottlenecks)
            state.cto_approved = cto_data.approved_to_proceed
            logger.info(f"CTO recs={len(recs)}, approved={state.cto_approved}")
        else:
            logger.warning(f"CTO agent failed: {advisory_results[0]}")

        if not isinstance(advisory_results[1], BaseException):
            ux_data = advisory_results[1].output
            recs = ux_data.recommendations or []
            state.advisory_results["ux"] = "UX Guidelines:\n" + "\n".join(recs) if recs else "UX Guidelines: (none)"
            logger.info(f"UX recs={len(recs)}")
        else:
            logger.warning(f"UX agent failed: {advisory_results[1]}")
            state.advisory_results["ux"] = "UX Guidelines: (failed)"

        from pydantic_agents.core.cto_gate import apply as cto_gate

        if not await cto_gate(state, project=config.project):
            state.cto_approved = False

        state.stage = "advisory"
        return state
