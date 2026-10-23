import asyncio
import logging

from pydantic_agents.core.limits import llm_semaphore
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
        semaphore = llm_semaphore()

        from pydantic_agents.core.adapters.workspace import WorkspaceContext

        workspace = WorkspaceContext()
        md_context = workspace.get_markdown_context()

        from pydantic_agents.agents.cto import OrgContext

        org_ctx = OrgContext(
            tech_stack=["Go", "Vue", "PostgreSQL", "Redis"], compliance_tier="Standard", monthly_cloud_budget=1000.0
        )

        cto_prompt, ux_prompt = self._build_prompts(story_context, md_context)

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
        self._record_cto(state, advisory_results[0])
        self._record_ux(state, advisory_results[1])

        from pydantic_agents.core.cto_gate import apply as cto_gate

        # The gate returns True for an approved verdict or an explicit human override (env var / marker file).
        state.cto_approved = await cto_gate(state, project=config.project)

        state.stage = "advisory"
        return state

    @staticmethod
    def _build_prompts(story_context: str, md_context: str) -> tuple[str, str]:
        from pydantic_agents.core.context_engine import retrieve_context

        cto_prompt = f"Review these user stories and provide architectural advice:\n{story_context}"
        cto_ctx = retrieve_context("cto", story_context)
        if cto_ctx:
            cto_prompt += f"\n\nRetrieved Architectural Context/ADRs/Rules:\n{cto_ctx}"

        ux_prompt = f"Review these user stories and provide UX/UI advice:\n{story_context}"
        ux_ctx = retrieve_context("ux", story_context)
        if ux_ctx:
            ux_prompt += f"\n\nRetrieved UX Context/Rules:\n{ux_ctx}"

        if md_context:
            guidelines = f"\n\nAdhere to the following local guidelines:\n{md_context}"
            cto_prompt += guidelines
            ux_prompt += guidelines
        return cto_prompt, ux_prompt

    @staticmethod
    def _record_cto(state: PipelineState, result) -> None:
        """Fold the CTO result into state. Fails closed: no verdict means not approved."""
        if isinstance(result, BaseException):
            logger.warning(f"CTO agent failed: {result}")
            state.cto_approved = False
            return

        cto_data = result.output
        recs = list(cto_data.recommendations or [])
        for constraint in cto_data.architectural_constraints:
            agents = ", ".join(constraint.affected_agents) or "all agents"
            recs.append(f"[constraint: {agents}] {constraint.description}")
        state.advisory_results["cto"] = "CTO Guidelines:\n" + "\n".join(recs) if recs else "CTO Guidelines: (none)"
        if cto_data.bottlenecks_identified:
            state.advisory_results["cto_blockers"] = "\n".join(cto_data.bottlenecks_identified)
        state.cto_approved = cto_data.approved_to_proceed
        if "approved_to_proceed" not in cto_data.model_fields_set:
            # Distinguish "the CTO said no" from "the CTO never answered": the default blocks the pipeline.
            logger.warning("CTO output omitted approved_to_proceed; treating as not approved.")
        logger.info(f"CTO recs={len(recs)}, approved={state.cto_approved}")

    @staticmethod
    def _record_ux(state: PipelineState, result) -> None:
        if isinstance(result, BaseException):
            logger.warning(f"UX agent failed: {result}")
            state.advisory_results["ux"] = "UX Guidelines: (failed)"
            return

        recs = result.output.recommendations or []
        state.advisory_results["ux"] = "UX Guidelines:\n" + "\n".join(recs) if recs else "UX Guidelines: (none)"
        logger.info(f"UX recs={len(recs)}")
