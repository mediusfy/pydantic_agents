import json

from pydantic_ai import RunContext

from pydantic_agents.core.exceptions import (
    AgentExecutionError,
    AgentNotFoundError,
    AmbiguousCapabilityError,
)
from pydantic_agents.core.pipeline.context import ContextFrame
from pydantic_agents.core.registry import instrumented_query, registry
from pydantic_agents.core.session_recorder import current_run_context
from pydantic_agents.core.tools.registry import register_tool

MAX_DELEGATION_DEPTH = 3


def discover_agent_capabilities(ctx: RunContext) -> str:
    """Lists all available capabilities that other agents can perform.
    Use this to find out who to delegate to.

    When a capability is provided by more than one agent, all owners are listed.
    Callers must pass owner=<agent_name> to delegate_to_agent in that case.
    """
    lines: list[str] = []
    for cap, owners in registry.capabilities():
        if len(owners) == 1:
            lines.append(f"- Capability: '{cap}' (provided by {owners[0]} agent)")
        else:
            owners_str = ", ".join(owners)
            lines.append(f"- Capability: '{cap}' (AMBIGUOUS — provided by {owners_str}; pass owner=<agent_name>)")

    if not lines:
        return "No agents currently registered with capabilities."

    return "Available Agent Capabilities for Delegation:\n" + "\n".join(lines)


async def delegate_to_agent(
    ctx: RunContext,
    capability: str,
    prompt: str,
    context_summary: str | dict | ContextFrame,
    owner: str | None = None,
) -> str:
    """Delegate a sub-task to another agent based on their capability.

    To ensure proficient token usage and accurate responses:
    - capability: Which agent capability to use (see discover_agent_capabilities)
    - prompt: The specific task/question for the target agent
    - context_summary: Provide ONLY the absolutely necessary context (can be a string, dict, or ContextFrame). Do not dump full history.
    - owner: When a capability has multiple providers, pin the agent name here.
      Required for ambiguous capabilities; ignored when there is only one owner.
    """
    if isinstance(context_summary, dict):
        context_str = json.dumps(context_summary, indent=2)
    elif hasattr(context_summary, "model_dump"):
        context_str = json.dumps(context_summary.model_dump(), indent=2)
    else:
        context_str = str(context_summary)

    # `or {}` defends against the ContextVar being unset (default is now {},
    # but a caller could still explicitly set None outside of record_span).
    ctx_data = current_run_context.get() or {}
    current_depth = ctx_data.get("delegation_depth", 0)

    if current_depth >= MAX_DELEGATION_DEPTH:
        return "ERROR: Maximum delegation depth reached. You must complete the task with the information you have."

    full_prompt = (
        f"--- DELEGATED TASK ---\n"
        f"Context provided by caller:\n{context_str}\n\n"
        f"Task:\n{prompt}\n\n"
        f"Please provide an accurate and concise response."
    )

    attrs = {
        "capability": capability,
        "context_summary_length": len(context_str),
        "prompt_length": len(prompt),
        "context_summary_preview": context_str[:200],
        "caller_agent": ctx_data.get("agent_name", ""),
        "caller_pipeline_stage": ctx_data.get("pipeline_stage", ""),
        "caller_run_id": ctx_data.get("run_id", ""),
        "delegation_depth": current_depth + 1,
        "owner_pin": owner or "",
    }

    try:
        # The callee shares the caller's usage budget: a delegation tree spends one run's caps, not one per node.
        return await instrumented_query(
            capability=capability,
            prompt=full_prompt,
            span_name="a2a.delegate",
            attributes=attrs,
            delegation_depth=current_depth + 1,
            owner=owner,
            usage=getattr(ctx, "usage", None),
        )
    except AmbiguousCapabilityError as e:
        # Surface the candidate owners back to the LLM so it can retry with
        # a pinned owner instead of failing the whole run.
        return (
            f"ERROR: Capability '{capability}' is ambiguous. Candidates: {e.owners}. "
            f"Retry delegate_to_agent with owner=<agent_name>."
        )
    except (AgentNotFoundError, AgentExecutionError) as e:
        return f"ERROR: {e}"


register_tool(
    "discover_agent_capabilities",
    discover_agent_capabilities,
    "Find out what other agents can do.",
    ["a2a"],
)

register_tool(
    "delegate_to_agent",
    delegate_to_agent,
    "Delegate a task to another agent with concise context to save tokens.",
    ["a2a"],
)
