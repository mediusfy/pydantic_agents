"""Pipeline factory — single source of truth for assembling any pipeline.

Dispatches to the correct framework based on PipelineConfig.framework.
Preserves the legacy ``build_default_pipeline()`` for backwards compatibility.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import STAGES, PipelineConfig
from pydantic_agents.core.pipeline.orchestrator import OrgitrationPipeline
from pydantic_agents.core.pipeline.protocols import PipelineOrchestrator
from pydantic_agents.core.stages.advisory_stage import AdvisoryStage
from pydantic_agents.core.stages.ba_stage import BAStage
from pydantic_agents.core.stages.implementation_stage import ImplementationStage
from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

logger = logging.getLogger(__name__)


_FRAMEWORK_EXTRAS: dict[str, str] = {
    "crewai": "crewai",
    "autogen": "pyautogen",
}


def build_default_pipeline(config: PipelineConfig) -> OrgitrationPipeline:
    """Legacy: construct an ``OrgitrationPipeline`` with the four standard stages."""
    pipeline = OrgitrationPipeline(config)
    pipeline.register_stage(STAGES.BA, BAStage())
    pipeline.register_stage(STAGES.ADVISORY, AdvisoryStage())
    pipeline.register_stage(STAGES.IMPLEMENTATION, ImplementationStage())
    pipeline.register_stage(STAGES.QUALITY_GATE, QualityGateStage())
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

        initial = PipelineState(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            stage="init",
            requirement=self._config.requirement or "",
        )
        final = await self._graph.ainvoke(initial)
        return final
