import asyncio
import logging
from typing import Any

from pydantic_agents.core.models.core import FailureReport, PipelineState
from pydantic_agents.core.pipeline.autogen.groups import DebugGroupChat, DesignGroupChat, SdlcGroupChat
from pydantic_agents.core.pipeline.config import BLUEPRINTS, PipelineConfig
from pydantic_agents.core.session_recorder import recorder

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
            result = await asyncio.to_thread(user.initiate_chat, manager, message=task)
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


def _apply_group_result(state: PipelineState, group_class_name: str, result: Any) -> None:
    summary = str(result.summary if hasattr(result, "summary") else result)
    if group_class_name == "SdlcGroupChat":
        state.advisory_results["conversation"] = summary

        # Extract code block artifacts from chat history messages
        from pydantic_agents.core.models.core import Artifact
        from pydantic_agents.core.utils import parse_code_blocks_to_files

        chat_history = getattr(result, "chat_history", [])
        for msg in chat_history:
            content = msg.get("content", "")
            name = msg.get("name", "")
            if not content or not name:
                continue

            name_lower = name.lower()
            agent_name = None
            if "backend" in name_lower:
                agent_name = "backend"
            elif "frontend" in name_lower:
                agent_name = "frontend"
            elif "qa" in name_lower or "test" in name_lower:
                agent_name = "test_eng"
            elif "docs" in name_lower:
                agent_name = "docs"
            elif "devops" in name_lower:
                agent_name = "devops"

            if agent_name:
                files = parse_code_blocks_to_files(content)
                if files and not (len(files) == 1 and "output.txt" in files):
                    state.artifacts.append(
                        Artifact(
                            agent_name=agent_name,
                            files=files,
                            metadata={"agent_name": agent_name, "user_story_id": "global"},
                        )
                    )
    elif group_class_name == "DesignGroupChat":
        state.advisory_results["design_discussion"] = summary
        state.cto_approved = True
    elif group_class_name == "DebugGroupChat":
        state.feedback = summary
