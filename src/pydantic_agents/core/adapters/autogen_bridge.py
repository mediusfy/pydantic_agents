"""Bridge: pydantic_ai Agents → AutoGen AssistantAgent wrappers.

Each pydantic_ai Agent is adapted into an AutoGen AssistantAgent,
preserving its system prompt, model configuration, and tool set
so it can participate in AutoGen group chats.
"""

import asyncio
from typing import Any

from pydantic_ai import Agent as PydanticAgent


def _build_llm_config(agent: PydanticAgent) -> list[dict[str, Any]]:
    """Extract LLM config list from a pydantic_ai Agent for AutoGen."""
    model = getattr(agent, "_model", None)
    if model is None:
        return [{"model": "gpt-4o", "api_type": "openai"}]

    model_name = getattr(model, "model_name", str(model))
    api_type = "openai"
    config: dict[str, Any] = {"model": model_name, "temperature": 0.1}

    provider = getattr(model, "provider", None)
    if provider:
        base_url = getattr(provider, "base_url", None)
        if base_url:
            config["base_url"] = str(base_url)
        api_key = getattr(provider, "api_key", None)
        if api_key:
            config["api_key"] = api_key
    config["api_type"] = api_type
    return [config]


def _extract_system_message(agent: PydanticAgent) -> str:
    """Extract the system prompt from a pydantic_ai Agent."""
    prompts = getattr(agent, "_system_prompts", None) or []
    if not prompts:
        return "You are a helpful AI assistant."
    parts = [str(p) for p in prompts if p]
    return "\n\n".join(parts) if parts else "You are a helpful AI assistant."


def _find_output_type(agent: PydanticAgent) -> Any:
    """Return the result type of a pydantic_ai Agent if set."""
    result_tool = getattr(agent, "_result_tool", None)
    if result_tool is None:
        return None
    return getattr(result_tool, "output_type", None)


def _register_tool_functions(agent: PydanticAgent) -> dict[str, Any]:
    """Extract pydantic_ai tool functions as AutoGen-compatible functions.

    Returns a dict mapping tool_name -> callable for use in
    function_registry or as tools in AssistantAgent.
    """
    functions: dict[str, Any] = {}
    toolset = getattr(agent, "_function_toolset", None)
    if toolset is None:
        return functions
    for tool in getattr(toolset, "tools", []):
        name = getattr(tool, "name", None)
        fn = getattr(tool, "run", None) or getattr(tool, "fn", None)
        if name and fn:
            functions[name] = fn
    return functions


def as_autogen_assistant(
    pydantic_agent: PydanticAgent,
    *,
    name: str,
    system_message: str | None = None,
    code_execution: bool = False,
) -> Any:
    """Wrap a pydantic_ai Agent as an AutoGen AssistantAgent."""
    try:
        from autogen import AssistantAgent
    except ImportError:
        raise RuntimeError("pyautogen is not installed. Run: uv add pyautogen") from None

    llm_config = {"config_list": _build_llm_config(pydantic_agent)}
    msg = system_message or _extract_system_message(pydantic_agent)
    tools = _register_tool_functions(pydantic_agent)

    agent = AssistantAgent(
        name=name,
        system_message=msg,
        llm_config=llm_config,
        code_execution_config={"use_docker": False} if code_execution else False,
    )

    for tool_name, tool_fn in tools.items():

        async def _make_async(fn):

            async def _wrapper(*args, **kwargs):
                if asyncio.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                return fn(*args, **kwargs)

            return _wrapper

        agent.register_function(function_map={tool_name: lambda *a, fn=tool_fn, **kw: fn(*a, **kw)})

    return agent
