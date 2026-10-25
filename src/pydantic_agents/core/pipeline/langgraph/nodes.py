import logging

from pydantic_agents.core.adapters.agents import AgentProviderPort
from pydantic_agents.core.models.core import FailureReport, PipelineState
from pydantic_agents.core.pipeline.config import PipelineConfig
from pydantic_agents.core.pipeline.context import ContextFrame
from pydantic_agents.core.session_recorder import recorder
from pydantic_agents.core.stages.advisory_stage import AdvisoryStage
from pydantic_agents.core.stages.ba_stage import BAStage
from pydantic_agents.core.stages.implementation_stage import ImplementationStage
from pydantic_agents.core.stages.merge_stage import MergeStage
from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

logger = logging.getLogger(__name__)


def _make_error(state: PipelineState, agent_name: str, message: str) -> dict:
    return {
        "errors": [*state.errors, f"{agent_name}: {message}"],
        "failure_report": FailureReport(
            source=agent_name,
            error_type="runtime_error",
            trace=message,
        ),
    }


async def ba_node(
    state: PipelineState,
    config: PipelineConfig,
    provider: AgentProviderPort,
) -> dict:
    stage = BAStage()
    stage.agent_provider = provider
    async with recorder.record_span(
        run_id=state.run_id,
        agent_name="ba",
        pipeline_stage="ba",
        prompt=config.requirement or "",
    ) as rec:
        try:
            new_state = await stage.run(state, config)
            rec.set_output(new_state)
            if not new_state.user_stories:
                logger.warning("BA returned no stories")

            notes = [f"Defined requirements for project: {config.project}"]
            if new_state.user_stories:
                notes.extend(f"Story: {us.title}" for us in new_state.user_stories)
            cf = ContextFrame(
                output=f"Generated {len(new_state.user_stories)} user stories.",
                memory_notes=notes,
            )
            history = [*state.context_history, cf] if state.context_history else [cf]
            return {
                "user_stories": new_state.user_stories,
                "stage": "ba",
                "errors": new_state.errors,
                "active_context": cf,
                "context_history": history,
            }
        except Exception as e:
            logger.exception("BA stage failed: %s", e)
            return _make_error(state, "ba", str(e))


async def advisory_node(
    state: PipelineState,
    config: PipelineConfig,
    provider: AgentProviderPort,
) -> dict:
    stage = AdvisoryStage()
    stage.agent_provider = provider
    async with recorder.record_span(
        run_id=state.run_id,
        agent_name="advisory",
        pipeline_stage="advisory",
        prompt="\n".join(s.gherkin for s in (state.user_stories or [])),
    ) as rec:
        try:
            new_state = await stage.run(state, config)
            rec.set_output(new_state)

            notes = []
            for k, v in new_state.advisory_results.items():
                notes.append(f"Advisory ({k}): {v[:100]}...")
            cf = ContextFrame(
                output=f"CTO Advisory completed. Approved: {new_state.cto_approved}.",
                memory_notes=notes,
            )
            history = [*state.context_history, cf] if state.context_history else [cf]
            return {
                "advisory_results": new_state.advisory_results,
                "cto_approved": new_state.cto_approved,
                "stage": "advisory",
                "active_context": cf,
                "context_history": history,
            }
        except Exception as e:
            logger.exception("Advisory stage failed: %s", e)
            return _make_error(state, "advisory", str(e))


async def implementation_node(
    state: PipelineState,
    config: PipelineConfig,
    provider: AgentProviderPort,
) -> dict:
    stage = ImplementationStage()
    stage.agent_provider = provider
    async with recorder.record_span(
        run_id=state.run_id,
        agent_name="implementation",
        pipeline_stage="implementation",
        prompt=config.requirement or "",
    ) as rec:
        try:
            new_state = await stage.run(state, config)
            rec.set_output(new_state)

            files: list[str] = []
            for art in new_state.artifacts or []:
                if art.files:
                    files.extend(art.files.keys())
            cf = ContextFrame(
                output=f"Implementation stage generated {len(new_state.artifacts or [])} artifacts.",
                modified_files=files,
            )
            history = [*state.context_history, cf] if state.context_history else [cf]
            return {
                "artifacts": new_state.artifacts,
                "failure_report": new_state.failure_report,
                "feedback": new_state.feedback,
                "stage": "implementation",
                "refinement_attempts": state.refinement_attempts + 1,
                "active_context": cf,
                "context_history": history,
            }
        except Exception as e:
            logger.exception("Implementation stage failed: %s", e)
            return _make_error(state, "implementation", str(e))


async def quality_gate_node(
    state: PipelineState,
    config: PipelineConfig,
    provider: AgentProviderPort,
) -> dict:
    stage = QualityGateStage()
    stage.agent_provider = provider
    async with recorder.record_span(
        run_id=state.run_id,
        agent_name="quality_gate",
        pipeline_stage="quality_gate",
        prompt="Quality gate check",
    ) as rec:
        try:
            new_state = await stage.run(state, config)
            rec.set_output(new_state)

            cf = ContextFrame(
                output=f"Quality Gate validation complete. Approved: {new_state.approved}.",
                memory_notes=[f"Feedback: {new_state.feedback}"] if new_state.feedback else [],
            )
            history = [*state.context_history, cf] if state.context_history else [cf]
            return {
                "approved": new_state.approved,
                "failure_report": new_state.failure_report,
                "feedback": new_state.feedback,
                "stage": "quality_gate",
                "active_context": cf,
                "context_history": history,
            }
        except Exception as e:
            logger.exception("Quality gate failed: %s", e)
            return _make_error(state, "quality_gate", str(e))


async def merge_node(
    state: PipelineState,
    config: PipelineConfig,
) -> dict:
    stage = MergeStage()
    async with recorder.record_span(
        run_id=state.run_id,
        agent_name="merge",
        pipeline_stage="merge",
        prompt="Merge artifacts into project",
    ) as rec:
        try:
            new_state = await stage.run(state, config)
            rec.set_output(new_state)

            cf = ContextFrame(
                output=f"Merge stage complete. Approved: {new_state.approved}.",
            )
            history = [*state.context_history, cf] if state.context_history else [cf]
            return {
                "stage": "merge",
                "approved": new_state.approved,
                "active_context": cf,
                "context_history": history,
            }
        except Exception as e:
            logger.exception("Merge failed: %s", e)
            return _make_error(state, "merge", str(e))
