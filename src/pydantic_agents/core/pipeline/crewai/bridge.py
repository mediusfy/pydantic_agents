import asyncio
import json
import logging
from typing import Any

from pydantic import ValidationError

from pydantic_agents.core.models.core import AgentAdvice, Artifact, FailureReport, PipelineState, ReviewResult
from pydantic_agents.core.pipeline.config import BLUEPRINTS, PipelineConfig
from pydantic_agents.core.pipeline.crewai.crews import (
    ArchitectureReviewCrew,
    AuditCrew,
    QaCrew,
    ResearchCrew,
    SdlcCrew,
    SecurityReviewCrew,
)
from pydantic_agents.core.session_recorder import recorder

logger = logging.getLogger(__name__)


_CREW_MAP: dict[str, Any] = {
    "SdlcCrew": SdlcCrew,
    "ResearchCrew": ResearchCrew,
    "SecurityReviewCrew": SecurityReviewCrew,
    "QaCrew": QaCrew,
    "ArchitectureReviewCrew": ArchitectureReviewCrew,
    "AuditCrew": AuditCrew,
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
    return await asyncio.to_thread(crew.kickoff)


# ── Mapping crew output back onto the pipeline state ──────────────────────


def _output_text(task_output: Any) -> str:
    raw = getattr(task_output, "raw", None)
    if isinstance(raw, str) and raw:
        return raw
    output = getattr(task_output, "output", None)
    return str(output) if output is not None else str(task_output)


def _role_of(task_output: Any) -> str:
    agent = getattr(task_output, "agent", None)
    if isinstance(agent, str):
        return agent
    return str(getattr(agent, "role", None) or getattr(agent, "name", None) or "")


def _owner_and_kind(task_output: Any) -> tuple[str, str] | None:
    """The ``<owner>/<kind>`` a crew task declared, or a guess from the agent role for tasks that did not."""
    name = getattr(task_output, "name", None)
    if isinstance(name, str) and "/" in name:
        owner, _, kind = name.partition("/")
        return owner, kind
    role = _role_of(task_output).lower()
    if not role:
        return None
    if "business analyst" in role or role == "ba":
        return "ba", "note"
    if "cto" in role:
        return "cto", "note"
    if "backend" in role:
        return "backend", "artifact"
    if "frontend" in role:
        return "frontend", "artifact"
    if "qa" in role or "test" in role:
        return "test_eng", "artifact"
    if "review" in role:
        return "review", "verdict"
    return role, "note"


def _apply_result(state: PipelineState, result: Any) -> None:
    for task_out in getattr(result, "tasks_output", None) or []:
        target = _owner_and_kind(task_out)
        if target is None:
            continue
        owner, kind = target
        output = _output_text(task_out)
        if kind == "artifact":
            _merge_artifact(state, _artifact_from_output(owner, output))
        elif kind == "verdict":
            _apply_verdict(state, output)
        elif kind == "advice":
            _apply_advice(state, owner, output)
        elif kind == "feedback":
            state.feedback = output
        else:
            state.advisory_results[owner] = output


def _first_json_object(text: str) -> str | None:
    """The first complete JSON object in ``text`` (models often wrap it in prose or a code fence)."""
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return text[idx:end]
    return None


def _parse_json_model(model: type[Any], text: str) -> tuple[Any | None, str | None]:
    """Parse the first JSON object of ``text`` as ``model``; returns (instance, None) or (None, reason)."""
    payload = _first_json_object(text)
    if payload is None:
        return None, f"no JSON object found in {model.__name__} output"
    try:
        return model.model_validate_json(payload), None
    except ValidationError as exc:
        return None, f"{model.__name__} output failed validation: {exc.errors()[0].get('msg', exc)}"


def _apply_verdict(state: PipelineState, output: str) -> None:
    verdict, reason = _parse_json_model(ReviewResult, output)
    if verdict is None:
        state.approved = False
        state.feedback = output
        state.errors.append(f"crewai review: {reason}")
        logger.warning("CrewAI review verdict rejected: %s", reason)
        return
    state.approved = verdict.approved
    state.feedback = "\n".join(verdict.feedback) if verdict.feedback else output


def _apply_advice(state: PipelineState, owner: str, output: str) -> None:
    state.advisory_results[owner] = output
    advice, reason = _parse_json_model(AgentAdvice, output)
    if advice is None:
        state.cto_approved = False
        state.errors.append(f"crewai {owner} advice: {reason}")
        logger.warning("CrewAI %s advice rejected: %s", owner, reason)
        return
    state.cto_approved = advice.approved_to_proceed


def _merge_artifact(state: PipelineState, artifact: Artifact) -> None:
    """Keep one artifact per owner: a later task of the same owner (QaCrew's fix task after its test task) updates
    the files it re-emits and adds the rest, instead of leaving two artifacts for consumers keyed by owner."""
    for idx, existing in enumerate(state.artifacts):
        if existing.agent_name == artifact.agent_name:
            # A new instance, not an in-place update: an Artifact may already be referenced elsewhere.
            state.artifacts[idx] = existing.model_copy(update={"files": {**existing.files, **artifact.files}})
            return
    state.artifacts.append(artifact)


def _artifact_from_output(agent_name: str, output: str) -> Artifact:
    from pydantic_agents.core.utils import parse_code_blocks_to_files

    return Artifact(
        agent_name=agent_name,
        files=parse_code_blocks_to_files(output),
        metadata={"agent_name": agent_name},
    )
