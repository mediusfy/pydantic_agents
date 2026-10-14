"""Bridge: pydantic_ai Agents → CrewAI Agent wrappers.

Each pydantic_ai Agent is adapted into a CrewAI Agent by extracting its
system prompt (as backstory), model config, and tool set. The resulting
CrewAI agent delegates LLM calls through the same underlying model.
"""

from typing import Any

from pydantic_ai import Agent as PydanticAgent


def _resolve_llm_config(agent: PydanticAgent) -> dict[str, Any]:
    """Extract LLM configuration from a pydantic_ai Agent for CrewAI."""
    model = getattr(agent, "_model", None)
    if model is None:
        return {"config_list": [{"model": "gpt-4o", "api_type": "openai"}]}
    model_name = getattr(model, "model_name", str(model))
    base_url = getattr(getattr(model, "provider", None), "base_url", None)
    config: dict[str, Any] = {"model": model_name}
    if base_url:
        config["base_url"] = str(base_url)
    return {"config_list": [config]}


def _extract_backstory(agent: PydanticAgent) -> str:
    """Extract the system prompt from a pydantic_ai Agent as a CrewAI backstory."""
    prompts = getattr(agent, "_system_prompts", None) or []
    if not prompts:
        return ""
    parts = [str(p) for p in prompts if p]
    return "\n\n".join(parts) if parts else ""


def _extract_tool_defs(agent: PydanticAgent) -> list[Any]:
    """Extract tool definitions from a pydantic_ai Agent for CrewAI tools."""
    tools: list[Any] = []
    toolset = getattr(agent, "_function_toolset", None)
    if toolset is None:
        return tools
    for tool in getattr(toolset, "tools", []):
        try:
            crew_tool = _pydantic_tool_to_crewai(tool)
            if crew_tool:
                tools.append(crew_tool)
        except Exception:
            pass
    return tools


def _pydantic_tool_to_crewai(pydantic_tool: Any) -> Any:
    """Convert a single pydantic_ai tool to a CrewAI-compatible tool.

    Returns a simple callable wrapped with name+description metadata.
    """
    try:
        from crewai.tools import BaseTool
    except ImportError:
        return None

    name = getattr(pydantic_tool, "name", "unknown_tool")
    description = getattr(pydantic_tool, "description", "") or ""
    fn = getattr(pydantic_tool, "run", None) or getattr(pydantic_tool, "fn", None)

    if fn is None:
        return None

    _callable = fn

    class _WrappedTool(BaseTool):
        name: str = name
        description: str = description

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            return _callable(*args, **kwargs)

    return _WrappedTool()


def as_crewai_agent(
    pydantic_agent: PydanticAgent,
    *,
    role: str,
    goal: str,
    allow_delegation: bool = False,
) -> Any:
    """Wrap a pydantic_ai Agent as a CrewAI Agent."""
    try:
        from crewai import Agent as CrewAgent
    except ImportError:
        raise RuntimeError("crewai is not installed. Run: uv add crewai") from None

    backstory = _extract_backstory(pydantic_agent)
    llm_config = _resolve_llm_config(pydantic_agent)
    tools = _extract_tool_defs(pydantic_agent)

    return CrewAgent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm_config["config_list"][0]["model"] if llm_config["config_list"] else "gpt-4o",
        tools=tools,
        allow_delegation=allow_delegation,
        verbose=True,
    )
