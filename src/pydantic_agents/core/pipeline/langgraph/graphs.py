from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from pydantic_agents.core.adapters.agents import AgentProviderPort
from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import BLUEPRINTS, PipelineConfig

from .checkpointer import DiskCheckpointSaver
from .nodes import (
    advisory_node,
    ba_node,
    crewai_implementation_node,
    implementation_node,
    merge_node,
    quality_gate_node,
)
from .routing import (
    route_after_advisory,
    route_after_ba,
    route_after_implementation,
    route_after_implementation_subgraph,
    route_after_quality_gate,
    route_retry,
)

_STAGE_ORDER = ("ba", "advisory", "implementation", "quality_gate")


def _noop(_state: PipelineState) -> dict[str, Any]:
    return {}


def _async_node(fn: Any, *args: Any) -> Any:
    """Wrap an async node function as a 1-arg coroutine so LangGraph awaits it without injecting RunnableConfig."""

    async def _node(state: PipelineState) -> dict[str, Any]:
        return await fn(state, *args)

    return _node


def build_implementation_subgraph(
    config: PipelineConfig,
    provider: AgentProviderPort,
    checkpointer: MemorySaver | None = None,
) -> Any:
    workflow = StateGraph(PipelineState)

    impl_fn = (
        _async_node(crewai_implementation_node, config)
        if config.framework == "composite"
        else _async_node(implementation_node, config, provider)
    )
    workflow.add_node("implementation", impl_fn)
    workflow.add_node("quality_gate", _async_node(quality_gate_node, config, provider))
    workflow.add_node("retry_or_exit", _noop)

    workflow.add_edge(START, "implementation")

    workflow.add_conditional_edges(
        "implementation",
        route_after_implementation,
        {"quality_gate": "quality_gate", "retry_implementation": "retry_or_exit", "end": END},
    )

    workflow.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {"retry_implementation": "retry_or_exit", "end": END},
    )

    workflow.add_conditional_edges(
        "retry_or_exit",
        route_retry,
        {"implementation": "implementation", "end": END},
    )

    return workflow.compile(checkpointer=checkpointer)


def _blueprint_stages(config: PipelineConfig) -> list[str]:
    """The ordered stage names declared on ``config.blueprint``; fails closed when missing."""
    bp = BLUEPRINTS.get(config.blueprint)
    if bp is None or not bp.get("stages"):
        raise ValueError(f"Blueprint {config.blueprint!r} has no stages to run. Available: {', '.join(BLUEPRINTS)}")
    return [str(getattr(s, "value", s)) for s in bp["stages"]]


def _add_stage_nodes(
    workflow: Any,
    config: PipelineConfig,
    provider: AgentProviderPort,
    wanted: set[str],
) -> None:
    """Register only the stage nodes present in ``wanted`` (plus ``merge``)."""
    if "ba" in wanted:
        workflow.add_node("ba", _async_node(ba_node, config, provider))
    if "advisory" in wanted:
        workflow.add_node("advisory", _async_node(advisory_node, config, provider))
    if "implementation" in wanted:
        # Subgraph inherits the parent graph's checkpointer at runtime when compiled without its own.
        workflow.add_node("implementation", build_implementation_subgraph(config, provider))
    if "quality_gate" in wanted:
        workflow.add_node("quality_gate", _async_node(quality_gate_node, config, provider))
    workflow.add_node("merge", _async_node(merge_node, config))


def _top_level_chain(wanted: set[str]) -> list[str]:
    """Ordered top-level nodes for a full run (when ``implementation`` is present, its subgraph runs the gate)."""
    if "implementation" in wanted:
        return [s for s in ("ba", "advisory", "implementation") if s in wanted]
    return [s for s in _STAGE_ORDER if s in wanted]


def _wire_full_graph_edges(workflow: Any, wanted: set[str]) -> None:
    chain = _top_level_chain(wanted)
    workflow.add_edge(START, chain[0])
    for idx, stage in enumerate(chain):
        nxt = chain[idx + 1] if idx + 1 < len(chain) else END
        if stage == "ba":
            workflow.add_conditional_edges("ba", route_after_ba, {"advisory": nxt, "end": END})
        elif stage == "advisory":
            workflow.add_conditional_edges("advisory", route_after_advisory, {"implementation": nxt, "end": END})
        elif stage == "implementation":
            workflow.add_conditional_edges(
                "implementation",
                route_after_implementation_subgraph,
                {"merge": "merge", "end": END},
            )
            workflow.add_edge("merge", END)
        else:
            workflow.add_edge(stage, nxt)


def build_full_graph(
    config: PipelineConfig,
    provider: AgentProviderPort,
    checkpointer: MemorySaver | None = None,
) -> Any:
    stages = _blueprint_stages(config)
    wanted = set(stages)
    is_single_step = config.step != "all"
    use_checkpoint = checkpointer if checkpointer is not None else DiskCheckpointSaver(config.project)

    workflow = StateGraph(PipelineState)
    _add_stage_nodes(workflow, config, provider, wanted)

    if is_single_step:
        target = next((s for s in _STAGE_ORDER if s in wanted), "ba")
        workflow.add_edge(START, target)
        workflow.add_edge(target, END)
        return workflow.compile(checkpointer=use_checkpoint)

    _wire_full_graph_edges(workflow, wanted)
    return workflow.compile(checkpointer=use_checkpoint)
