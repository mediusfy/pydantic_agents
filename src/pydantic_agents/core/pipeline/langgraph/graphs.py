from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from pydantic_agents.core.adapters.agents import AgentProviderPort
from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.pipeline.config import BLUEPRINTS, PipelineConfig

from .nodes import (
    advisory_node,
    ba_node,
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


def build_implementation_subgraph(
    config: PipelineConfig,
    provider: AgentProviderPort,
    checkpointer: MemorySaver | None = None,
) -> Any:
    workflow = StateGraph(PipelineState)

    workflow.add_node("implementation", lambda _s: implementation_node(_s, config, provider))
    workflow.add_node("quality_gate", lambda _s: quality_gate_node(_s, config, provider))
    workflow.add_node("retry_or_exit", lambda _s: {})

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


def build_full_graph(
    config: PipelineConfig,
    provider: AgentProviderPort,
    checkpointer: MemorySaver | None = None,
) -> Any:
    bp = BLUEPRINTS.get(config.blueprint, {})
    stages = bp.get("stages", [])
    is_single_step = config.step != "all"
    use_checkpoint = checkpointer or (MemorySaver() if config.resume else None)

    workflow = StateGraph(PipelineState)

    if not is_single_step or "ba" in stages:
        workflow.add_node("ba", lambda _s: ba_node(_s, config, provider))

    if not is_single_step or "advisory" in stages:
        workflow.add_node("advisory", lambda _s: advisory_node(_s, config, provider))

    if not is_single_step or "implementation" in stages:
        impl_subgraph = build_implementation_subgraph(config, provider, use_checkpoint)
        workflow.add_node("implementation", impl_subgraph)

    if not is_single_step or "quality_gate" in stages:
        workflow.add_node("quality_gate", lambda _s: quality_gate_node(_s, config, provider))

    workflow.add_node("merge", lambda _s: merge_node(_s, config))

    if is_single_step:
        target = "ba"
        for s in ("ba", "advisory", "implementation", "quality_gate"):
            if s in stages:
                target = s
                break
        workflow.add_edge(START, target)
        workflow.add_edge(target, END)
        return workflow.compile(checkpointer=use_checkpoint)

    workflow.add_edge(START, "ba")

    workflow.add_conditional_edges(
        "ba",
        route_after_ba,
        {"advisory": "advisory", "end": END},
    )

    workflow.add_conditional_edges(
        "advisory",
        route_after_advisory,
        {"implementation": "implementation", "end": END},
    )

    workflow.add_conditional_edges(
        "implementation",
        route_after_implementation_subgraph,
        {"merge": "merge", "end": END},
    )

    workflow.add_edge("merge", END)

    return workflow.compile(checkpointer=use_checkpoint)
