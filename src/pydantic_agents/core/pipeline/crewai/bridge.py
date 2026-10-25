import logging
from typing import Any

from pydantic_agents.core.models.core import Artifact, FailureReport, PipelineState
from pydantic_agents.core.pipeline.config import BLUEPRINTS, PipelineConfig
from pydantic_agents.core.pipeline.crewai.crews import ResearchCrew, SdlcCrew, SecurityReviewCrew
from pydantic_agents.core.session_recorder import recorder

logger = logging.getLogger(__name__)


_CREW_MAP: dict[str, Any] = {
    "SdlcCrew": SdlcCrew,
    "ResearchCrew": ResearchCrew,
    "SecurityReviewCrew": SecurityReviewCrew,
}


async def run_crewai_pipeline(
    state: PipelineState,
    config: PipelineConfig,
) -> PipelineState:
    bp = BLUEPRINTS.get(config.blueprint, {})
    crew_class_name = bp.get("crew_class", "SdlcCrew")

    if config.dry_run:
        logger.info("DRY RUN: CrewAI crew execution skipped.")
        state.stage = "complete"
        return state

    crew_cls = _CREW_MAP.get(crew_class_name)
    if crew_cls is None:
        msg = f"Unknown CrewAI crew class: {crew_class_name}"
        logger.error(msg)
        state.errors.append(msg)
        state.failure_report = FailureReport(source="crewai", error_type="config", trace=msg)
        return state

    async with recorder.record_span(
        run_id=state.run_id,
        agent_name=f"crewai_{crew_class_name}",
        pipeline_stage="implementation",
        prompt=config.requirement or "",
    ) as rec:
        try:
            from pydantic_agents.core.pipeline.context import ContextFrame

            logger.info("Running %s...", crew_class_name)
            crew = crew_cls().build(state)
            result = await _run_crew_async(crew)
            rec.set_output({"crew": crew_class_name, "result": str(result)[:500]})
            _apply_result(state, result)

            files: list[str] = []
            for art in state.artifacts or []:
                if art.files:
                    files.extend(art.files.keys())
            cf = ContextFrame(
                output=f"CrewAI execution for {crew_class_name} completed.",
                modified_files=files,
            )
            state.active_context = cf
            state.context_history.append(cf)

            state.stage = "implementation"
        except Exception as e:
            logger.exception("CrewAI pipeline failed: %s", e)
            state.errors.append(f"crewai: {e}")
            state.failure_report = FailureReport(source="crewai", error_type="runtime_error", trace=str(e))
            state.approved = False

    return state


async def _run_crew_async(crew: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(crew.kickoff)


def _apply_result(state: PipelineState, result: Any) -> None:
    tasks_output = getattr(result, "tasks_output", [])
    for task_out in tasks_output:
        agent_name = getattr(getattr(task_out, "agent", None), "name", "").lower()
        output = str(task_out.output) if hasattr(task_out, "output") else str(task_out)
        if "business analyst" in agent_name or agent_name == "ba":
            state.advisory_results["ba"] = output
        elif "cto" in agent_name:
            state.advisory_results["cto"] = output
        elif "backend" in agent_name:
            state.artifacts.append(_artifact_from_output("backend", task_out))
        elif "frontend" in agent_name:
            state.artifacts.append(_artifact_from_output("frontend", task_out))
        elif "qa" in agent_name or "test" in agent_name:
            state.artifacts.append(_artifact_from_output("test_eng", task_out))
        elif "review" in agent_name:
            state.feedback = output
            state.approved = "approved" in output.lower() and "rejected" not in output.lower()


def _artifact_from_output(agent_name: str, task_output: Any) -> Artifact:
    output = str(getattr(task_output, "output", task_output)) if hasattr(task_output, "output") else str(task_output)
    from pydantic_agents.core.utils import parse_code_blocks_to_files

    files = parse_code_blocks_to_files(output)
    return Artifact(
        agent_name=agent_name,
        files=files,
        metadata={"agent_name": agent_name},
    )
