"""Bridge: pydantic_ai Agents / StageRunners → LangGraph node functions."""

from typing import Any

from pydantic_agents.core.adapters.agents import AgentProviderPort
from pydantic_agents.core.limits import llm_semaphore
from pydantic_agents.core.models.core import PipelineState
from pydantic_agents.core.registry import registry


async def as_langgraph_node(
    state: PipelineState,
    *,
    agent_name: str,
    provider: AgentProviderPort,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Run a registered pydantic_ai Agent as a LangGraph node.

    Returns a state delta dict that LangGraph merges into the shared state.
    """
    agent = registry.get(agent_name)
    resolved_prompt = prompt or _default_prompt(state, agent_name)
    try:
        result = await provider.run_agent(llm_semaphore(), agent_name, agent, resolved_prompt)
        return _state_delta(state, agent_name, result)
    except Exception as e:
        return _error_delta(state, agent_name, str(e))


def _default_prompt(state: PipelineState, agent_name: str) -> str:
    if agent_name == "ba":
        return f"Refine these requirements into user stories:\n{state.requirement}"
    if agent_name == "cto":
        stories = "\n".join(f"- {s.id}: {s.title}\n  {s.gherkin}" for s in (state.user_stories or []))
        return f"Review these stories:\n{stories}"
    if agent_name == "backend":
        stories = "\n".join(f"- {s.id}: {s.title}\n  {s.gherkin}" for s in (state.user_stories or []))
        return f"Generate Go backend for:\n{stories}"
    return f"Process requirements: {state.requirement}"


def _state_delta(state: PipelineState, agent_name: str, result: Any) -> dict[str, Any]:
    from pydantic_agents.core.pipeline.context import ContextFrame

    output = getattr(result, "data", getattr(result, "output", result))

    cf = ContextFrame(
        output=str(output)[:500],
    )

    delta: dict[str, Any] = {
        "active_context": cf,
        "context_history": [*state.context_history, cf] if state.context_history else [cf],
    }

    if agent_name == "ba":
        stories = output if isinstance(output, list) else []
        delta.update({"user_stories": stories, "stage": "ba"})
    elif agent_name in ("cto", "ux"):
        recs = getattr(output, "recommendations", [])
        delta.update({f"advisory_results.{agent_name}": "\n".join(recs) if recs else "(none)"})
    else:
        delta.update({"stage": agent_name})
    return delta


def _error_delta(state: PipelineState, agent_name: str, error: str) -> dict[str, Any]:
    from pydantic_agents.core.pipeline.context import ContextFrame

    cf = ContextFrame(
        output=f"Error in {agent_name}: {error}",
        command_traces=[error],
    )
    return {
        "errors": state.errors + [f"{agent_name}: {error}"],
        "stage": agent_name,
        "active_context": cf,
        "context_history": [*state.context_history, cf] if state.context_history else [cf],
    }
