import asyncio
import logging
from typing import Any

from pydantic_agents.core.limits import llm_semaphore
from pydantic_agents.core.models.core import FailureReport, PipelineState
from pydantic_agents.core.pipeline.autogen.groups import DebugGroupChat, DesignGroupChat, SdlcGroupChat
from pydantic_agents.core.pipeline.config import BLUEPRINTS, PipelineConfig
from pydantic_agents.core.session_recorder import recorder
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)


_GROUP_MAP: dict[str, Any] = {
    "SdlcGroupChat": SdlcGroupChat,
    "DesignGroupChat": DesignGroupChat,
    "DebugGroupChat": DebugGroupChat,
}


async def run_autogen_pipeline(
    state: PipelineState,
    config: PipelineConfig,
) -> PipelineState:
    bp = BLUEPRINTS.get(config.blueprint, {})
    group_class_name = bp.get("group_class", "SdlcGroupChat")

    if config.dry_run:
        logger.info("DRY RUN: AutoGen group chat skipped.")
        state.stage = "complete"
        return state

    group_cls = _GROUP_MAP.get(group_class_name)
    if group_cls is None:
        msg = f"Unknown AutoGen group class: {group_class_name}"
        logger.error(msg)
        state.errors.append(msg)
        state.failure_report = FailureReport(source="autogen", error_type="config", trace=msg)
        return state

    task = config.requirement or "Design and implement a standard service."
    label = {"SdlcGroupChat": "complete", "DesignGroupChat": "advisory", "DebugGroupChat": "review"}.get(
        group_class_name, "complete"
    )

    async with recorder.record_span(
        run_id=state.run_id,
        agent_name=f"autogen_{group_class_name}",
        pipeline_stage=label,
        prompt=task,
    ) as rec:
        try:
            from pydantic_agents.core.pipeline.context import ContextFrame

            logger.info("Starting %s...", group_class_name)
            _, manager = group_cls().build(task)
            from autogen import UserProxyAgent

            user = UserProxyAgent(
                name="Orchestrator",
                human_input_mode="NEVER",
                code_execution_config=False,
            )
            async with llm_semaphore():
                result = await asyncio.wait_for(
                    asyncio.to_thread(user.initiate_chat, manager, message=task),
                    timeout=settings.agent_timeout_seconds,
                )
            rec.set_output({"group": group_class_name, "summary": str(result)[:500]})
            _apply_group_result(state, group_class_name, result)

            summary = str(result.summary if hasattr(result, "summary") else result)[:200]
            cf = ContextFrame(
                output=f"AutoGen execution for {group_class_name} completed. Summary: {summary}...",
            )
            state.active_context = cf
            state.context_history.append(cf)

            state.stage = label
        except Exception as e:
            logger.exception("AutoGen pipeline failed: %s", e)
            state.errors.append(f"autogen: {e}")
            state.failure_report = FailureReport(source="autogen", error_type="runtime_error", trace=str(e))

    return state


# Which pipeline agent a chat speaker stands for: the first row whose keywords appear in the speaker's name.
_SPEAKER_AGENTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("backend",), "backend"),
    (("frontend",), "frontend"),
    (("qa", "test"), "test_eng"),
    (("docs",), "docs"),
    (("devops",), "devops"),
)


def _agent_for_speaker(name: str) -> str | None:
    lowered = name.lower()
    return next((agent for keywords, agent in _SPEAKER_AGENTS if any(k in lowered for k in keywords)), None)


def _artifacts_from_chat(chat_history: list[dict[str, Any]]) -> list[Any]:
    """One Artifact per chat message whose speaker maps to an agent and whose text contains named code files."""
    from pydantic_agents.core.models.core import Artifact
    from pydantic_agents.core.utils import parse_code_blocks_to_files

    artifacts = []
    for msg in chat_history:
        content = msg.get("content", "")
        name = msg.get("name", "")
        agent_name = _agent_for_speaker(name) if content and name else None
        if not agent_name:
            continue
        files = parse_code_blocks_to_files(content)
        if not files or (len(files) == 1 and "output.txt" in files):  # no named files: not worth an artifact
            continue
        artifacts.append(
            Artifact(
                agent_name=agent_name,
                files=files,
                metadata={"agent_name": agent_name, "user_story_id": "global"},
            )
        )
    return artifacts


def _apply_group_result(state: PipelineState, group_class_name: str, result: Any) -> None:
    summary = str(result.summary if hasattr(result, "summary") else result)
    if group_class_name == "SdlcGroupChat":
        state.advisory_results["conversation"] = summary
        state.artifacts.extend(_artifacts_from_chat(getattr(result, "chat_history", [])))
    elif group_class_name == "DesignGroupChat":
        state.advisory_results["design_discussion"] = summary
        state.cto_approved = True
    elif group_class_name == "DebugGroupChat":
        state.feedback = summary
