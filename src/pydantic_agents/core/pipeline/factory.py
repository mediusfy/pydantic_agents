"""Pipeline factory — single source of truth for assembling any pipeline.

Dispatches between ``OrgitrationPipeline`` (``legacy``) and ``LangGraph`` (``langgraph``).
CrewAI, AutoGen, and ``composite`` bridges were retired so every pipeline execution honours ``UsageLimits``,
pipeline-wide token caps, OTel GenAI spans, and checkpoints.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage

from pydantic_agents.core.limits import pipeline_usage_var, run_id_var
from pydantic_agents.core.models.core import FailureReport, PipelineState, UserStory
from pydantic_agents.core.pipeline.config import BLUEPRINTS, STAGES, PipelineConfig
from pydantic_agents.core.pipeline.orchestrator import OrgitrationPipeline
from pydantic_agents.core.pipeline.protocols import PipelineOrchestrator
from pydantic_agents.core.stages.advisory_stage import AdvisoryStage
from pydantic_agents.core.stages.ba_stage import BAStage
from pydantic_agents.core.stages.implementation_stage import ImplementationStage
from pydantic_agents.core.stages.quality_gate_stage import QualityGateStage

logger = logging.getLogger(__name__)


def build_default_pipeline(config: PipelineConfig) -> OrgitrationPipeline:
    """Construct an ``OrgitrationPipeline`` registering only the blueprint's declared stages."""
    pipeline = OrgitrationPipeline(config)
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
    """Dispatch to the configured pipeline engine ('legacy' or 'langgraph')."""
    if config.framework == "langgraph":
        return _build_langgraph_pipeline(config)
    if config.framework == "legacy":
        return build_default_pipeline(config)
    raise ValueError(
        f"Unsupported pipeline framework {config.framework!r}; supported frameworks are 'legacy' and 'langgraph'."
    )


def _build_langgraph_pipeline(config: PipelineConfig) -> PipelineOrchestrator:
    from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
    from pydantic_agents.core.pipeline.langgraph.graphs import build_full_graph

    provider = PydanticAIAgentProvider()
    graph = build_full_graph(config, provider)
    return _GraphPipelineWrapper(graph, config)


class _GraphPipelineWrapper:
    """Wraps a compiled LangGraph for the ``execute()`` interface."""

    def __init__(self, graph: Any, config: PipelineConfig):
        self._graph = graph
        self._config = config

    def _resolve_run_id(self) -> str:
        import uuid

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
        return run_id or f"run-{uuid.uuid4().hex[:8]}"

    async def _find_resume_config(self, run_id: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Latest clean checkpoint config in ``run_id``'s history and its state values, or (None, {})."""
        get_history = getattr(self._graph, "aget_state_history", None)
        if not callable(get_history):
            return None, {}
        base_cfg = {"configurable": {"thread_id": run_id}}
        async for snap in get_history(base_cfg):
            nxt = getattr(snap, "next", None)
            values = getattr(snap, "values", None) or {}
            if nxt and nxt != ("__start__",) and not values.get("errors") and not values.get("failure_report"):
                return dict(snap.config), dict(values)
        return None, {}

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
        resume_cfg, resume_vals = await self._find_resume_config(run_id) if self._config.resume else (None, {})
        pipe_usage = RunUsage(
            requests=int(resume_vals.get("total_requests", 0) or 0),
            input_tokens=int(resume_vals.get("total_input_tokens", 0) or 0),
            output_tokens=int(resume_vals.get("total_output_tokens", 0) or 0),
        )
        run_tok = run_id_var.set(run_id) if run_id_var.get() is None else None
        usage_tok = pipeline_usage_var.set(pipe_usage)
        try:
            if resume_cfg is not None:
                logger.info("Resuming LangGraph pipeline run %s from saved checkpoint.", run_id)
                final = await self._graph.ainvoke(None, config=resume_cfg)
            else:
                initial = self._initial_state(run_id)
                final = await self._graph.ainvoke(initial, config={"configurable": {"thread_id": run_id}})
        except UsageLimitExceeded as exc:
            return PipelineState(
                run_id=run_id,
                stage="budget_exceeded",
                approved=False,
                total_input_tokens=pipe_usage.input_tokens,
                total_output_tokens=pipe_usage.output_tokens,
                total_requests=pipe_usage.requests,
                failure_report=FailureReport(source="pipeline", error_type="usage_limit", trace=str(exc)),
            )
        finally:
            pipeline_usage_var.reset(usage_tok)
            if run_tok is not None:
                run_id_var.reset(run_tok)

        result = PipelineState.model_validate(final) if isinstance(final, dict) else final
        result.total_input_tokens = pipe_usage.input_tokens
        result.total_output_tokens = pipe_usage.output_tokens
        result.total_requests = pipe_usage.requests
        self._fill_blocked_failure(result, self._config.step)
        ArtifactStorageAdapter(str(self._config.output_dir())).write_artifacts(result.artifacts)
        return result
