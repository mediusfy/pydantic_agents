"""Pipeline factory — single source of truth for assembling any pipeline.

Dispatches to the correct framework based on PipelineConfig.framework.
Preserves the legacy ``build_default_pipeline()`` for backwards compatibility.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Any

from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import BLUEPRINTS, STAGES, PipelineConfig
from pydantic_agents.core.pipeline.orchestrator import OrgitrationPipeline
from pydantic_agents.core.pipeline.protocols import PipelineOrchestrator
from pydantic_agents.core.stages.advisory_stage import AdvisoryStage
from pydantic_agents.core.stages.ba_stage import BAStage
from pydantic_agents.core.stages.implementation_stage import ImplementationStage
from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

logger = logging.getLogger(__name__)


_FRAMEWORK_EXTRAS: dict[str, str] = {
    "crewai": "crewai",
    "autogen": "autogen",  # the pyautogen distribution is imported as `autogen`
}


def build_default_pipeline(config: PipelineConfig) -> OrgitrationPipeline:
    """Legacy: construct an ``OrgitrationPipeline`` with the four standard stages."""
    pipeline = OrgitrationPipeline(config)
    # Only the blueprint's own stages: quick_backend, docs_update and security_review used to run the whole SDLC
    # (BA and the CTO gate included) because every stage was registered regardless.
    wanted = {str(getattr(s, "value", s)) for s in BLUEPRINTS.get(config.blueprint, {}).get("stages") or []}
    stages = {
        STAGES.BA: BAStage,
        STAGES.ADVISORY: AdvisoryStage,
        STAGES.IMPLEMENTATION: ImplementationStage,
        STAGES.QUALITY_GATE: QualityGateStage,
    }
    for name, stage_cls in stages.items():
        if not wanted or name.value in wanted:
            pipeline.register_stage(name, stage_cls())
    return pipeline


def build_pipeline(config: PipelineConfig) -> PipelineOrchestrator | OrgitrationPipeline:
    """Dispatch to the correct framework orchestrator based on config.framework."""
    _ensure_dependency(config.framework)
    if config.framework == "langgraph":
        return _build_langgraph_pipeline(config)
    elif config.framework == "crewai":
        return _build_crewai_pipeline(config)
    elif config.framework == "autogen":
        return _build_autogen_pipeline(config)
    elif config.framework == "composite":
        return _build_composite_pipeline(config)
    else:
        return build_default_pipeline(config)


def _ensure_dependency(framework: str) -> None:
    extra = _FRAMEWORK_EXTRAS.get(framework)
    if extra is None:
        return
    if importlib.util.find_spec(extra) is None:
        raise ImportError(
            f"Framework '{framework}' requires the '{extra}' package. "
            f"Install it with: uv sync --extra {framework}  "
            f"or: pip install pydantic-agents[{framework}]"
        )


def _build_langgraph_pipeline(config: PipelineConfig) -> PipelineOrchestrator:
    from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
    from pydantic_agents.core.pipeline.langgraph.graphs import build_full_graph

    provider = PydanticAIAgentProvider()
    graph = build_full_graph(config, provider)
    return _GraphPipelineWrapper(graph, config)


def _build_crewai_pipeline(config: PipelineConfig) -> PipelineOrchestrator:
    from pydantic_agents.core.pipeline.crewai.orchestrator import CrewAIPipeline

    return CrewAIPipeline(config)


def _build_autogen_pipeline(config: PipelineConfig) -> PipelineOrchestrator:
    from pydantic_agents.core.pipeline.autogen.orchestrator import AutoGenPipeline

    return AutoGenPipeline(config)


def _build_composite_pipeline(config: PipelineConfig) -> PipelineOrchestrator:
    from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
    from pydantic_agents.core.pipeline.langgraph.graphs import build_full_graph

    provider = PydanticAIAgentProvider()
    graph = build_full_graph(config, provider)
    return _GraphPipelineWrapper(graph, config)


class _GraphPipelineWrapper:
    """Wraps a compiled LangGraph for the ``execute()`` interface.

    Provides the same ``async def execute()`` contract expected by callers
    (``main.py``, ``pipeline_jobs.py``) so LangGraph can be a drop-in replacement.
    """

    def __init__(self, graph: Any, config: PipelineConfig):
        self._graph = graph
        self._config = config

    async def execute(self) -> PipelineState:
        import uuid

        from pydantic_agents.core.limits import run_id_var

        run_id = run_id_var.get() or f"run-{uuid.uuid4().hex[:8]}"
        initial = PipelineState(run_id=run_id, stage="init", requirement=self._config.requirement or "")
        # A graph compiled with a checkpointer refuses to run without a thread id; the run id is the thread, so a
        # resume of the same run continues its saved state.
        final = await self._graph.ainvoke(initial, config={"configurable": {"thread_id": run_id}})
        return PipelineState.model_validate(final) if isinstance(final, dict) else final
