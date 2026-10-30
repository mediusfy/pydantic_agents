"""Pipeline factory — single source of truth for assembling any pipeline.

Dispatches to the correct framework based on PipelineConfig.framework.
Preserves the legacy ``build_default_pipeline()`` for backwards compatibility.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Any

from pydantic_agents.core.models.core import FailureReport, PipelineState, UserStory
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
    # An unknown blueprint (or a stage-less one) fails closed: registering everything would run the full SDLC,
    # CTO gate included, for a typo.
    blueprint = BLUEPRINTS.get(config.blueprint)
    if blueprint is None or not blueprint.get("stages"):
        raise ValueError(
            f"Blueprint {config.blueprint!r} has no legacy stages to run. Available: {', '.join(BLUEPRINTS)}"
        )
    wanted = {str(getattr(s, "value", s)) for s in blueprint["stages"]}
    stages = {
        STAGES.BA: BAStage,
        STAGES.ADVISORY: AdvisoryStage,
        STAGES.IMPLEMENTATION: ImplementationStage,
        STAGES.QUALITY_GATE: QualityGateStage,
    }
    for name, stage_cls in stages.items():
        if name.value in wanted:
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

    def _resolve_run_id(self) -> str:
        import uuid

        from pydantic_agents.core.limits import run_id_var

        run_id = self._config.run_id or run_id_var.get()
        if self._config.resume and run_id is None:
            saver = getattr(self._graph, "checkpointer", None)
            finder = getattr(saver, "find_latest_run_id", None)
            if callable(finder):
                run_id = finder()
            if run_id is None:
                logger.warning(
                    "Resume requested for project %s but no checkpoint exists; starting a fresh run.",
                    self._config.project,
                )
        run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        if run_id_var.get() is None:
            run_id_var.set(run_id)
        return run_id

    async def _find_resume_config(self, run_id: str) -> dict[str, Any] | None:
        """Latest clean checkpoint config in ``run_id``'s history that still has work to run, or None."""
        get_history = getattr(self._graph, "aget_state_history", None)
        if not callable(get_history):
            return None
        base_cfg = {"configurable": {"thread_id": run_id}}
        async for snap in get_history(base_cfg):
            nxt = getattr(snap, "next", None)
            values = getattr(snap, "values", None) or {}
            if nxt and nxt != ("__start__",) and not values.get("errors") and not values.get("failure_report"):
                return dict(snap.config)
        return None

    def _initial_state(self, run_id: str) -> PipelineState:
        req = self._config.requirement or ""
        initial = PipelineState(run_id=run_id, stage="init", requirement=req)
        bp = BLUEPRINTS.get(self._config.blueprint, {})
        stages = {str(getattr(s, "value", s)) for s in (bp.get("stages") or [])}
        if self._config.step == "all" and "ba" not in stages and req:
            initial.user_stories = [UserStory(id="US-001", title=req[:80], gherkin=req)]
        return initial

    @staticmethod
    def _fill_blocked_failure(result: PipelineState, step: str) -> None:
        if step != "all" or result.approved or result.failure_report is not None:
            return
        if result.stage == "ba" and not result.user_stories:
            result.failure_report = FailureReport(
                source="ba",
                error_type="blocked",
                trace="The BA stage produced no user stories; nothing to design or implement.",
            )
        elif result.stage == "advisory" and not result.cto_approved:
            result.failure_report = FailureReport(
                source="advisory",
                error_type="blocked",
                trace="The CTO gate did not approve the run.",
            )

    async def execute(self) -> PipelineState:
        from pydantic_agents.core.adapters.workspace import ArtifactStorageAdapter

        run_id = self._resolve_run_id()
        resume_cfg = await self._find_resume_config(run_id) if self._config.resume else None
        if resume_cfg is not None:
            logger.info("Resuming LangGraph pipeline run %s from saved checkpoint.", run_id)
            final = await self._graph.ainvoke(None, config=resume_cfg)
        else:
            initial = self._initial_state(run_id)
            final = await self._graph.ainvoke(initial, config={"configurable": {"thread_id": run_id}})
        result = PipelineState.model_validate(final) if isinstance(final, dict) else final
        self._fill_blocked_failure(result, self._config.step)
        ArtifactStorageAdapter(str(self._config.output_dir())).write_artifacts(result.artifacts)
        return result
