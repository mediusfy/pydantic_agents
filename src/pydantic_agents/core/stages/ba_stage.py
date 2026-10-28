import logging

from pydantic_agents.core.limits import llm_semaphore
from pydantic_agents.core.models.core import EventTrigger, PipelineState, UserStory
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.registry import registry
from pydantic_agents.core.session_recorder import recorder
from pydantic_agents.core.stages.base import StageRunner

logger = logging.getLogger(__name__)


class BAStage(StageRunner):
    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        logger.info("--- Running BA Stage ---")

        if config.dry_run:
            logger.info("DRY RUN: Using mock BA data.")
            state.user_stories = [
                UserStory(
                    id="US-001",
                    title="Process Payment",
                    gherkin=(
                        "Given a valid payment request\nWhen the system processes it\nThen it emits PaymentSucceeded"
                    ),
                    event_triggers=[EventTrigger(id="evt-1", name="PaymentSucceeded", description="Payment processed")],
                ),
            ]
            state.stage = "ba"
            return state

        from pydantic_agents.core.adapters.workspace import WorkspaceContext

        workspace = WorkspaceContext()
        md_context = workspace.get_markdown_context()

        semaphore = llm_semaphore()
        raw_requirements = config.requirement or "Build a standard service."

        from pydantic_agents.core.context_engine import retrieve_context

        retrieved_ctx = await retrieve_context("ba", raw_requirements, project=config.project)

        prompt = f"Process these requirements: {raw_requirements}"
        if retrieved_ctx:
            prompt += f"\n\nRetrieved Context/ADRs/Rules:\n{retrieved_ctx}"
        if md_context:
            prompt += f"\n\nAdhere to the following local guidelines:\n{md_context}"

        async with recorder.record_span(
            run_id=state.run_id,
            agent_name="ba",
            pipeline_stage="ba",
            prompt=prompt,
            context_summary=raw_requirements,
        ):
            ba_agent = registry.get("ba")
            ba_result = await self.provider.run_agent(semaphore, "ba", ba_agent, prompt)
            output = getattr(ba_result, "output", [])
            state.user_stories = output if isinstance(output, list) else []

            if not state.user_stories:
                # The orchestrator stops the run on an empty story list; the stage still records that it ran.
                logger.warning("BA returned no user stories. Exiting.")
                state.stage = "ba"
                return state

            logger.info(f"BA generated {len(state.user_stories)} stories:")
            for us in state.user_stories:
                logger.info(f"  [{us.id}] {us.title}")

        state.stage = "ba"
        return state
