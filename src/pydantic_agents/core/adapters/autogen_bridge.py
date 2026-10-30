"""Bridge: pydantic_ai Agents → AutoGen AssistantAgent wrappers.

Each pydantic_ai Agent is adapted into an AutoGen AssistantAgent,
preserving its system prompt, model configuration, and tool set
so it can participate in AutoGen group chats.
"""

import logging
from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent as PydanticAgent

from pydantic_agents.core.adapters.crewai_bridge import (
    _call_tool_function,
    _model_name_of,
    _model_spec_of,
    _needs_run_context,
)
from pydantic_agents.core.models.factory import _litellm_connection
from pydantic_agents.core.pydantic_ai_compat import agent_tools, tool_function
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)


def _build_llm_config(agent: PydanticAgent) -> list[dict[str, Any]]:
    """Extract LLM config list from a pydantic_ai Agent for AutoGen."""
    model = getattr(agent, "_model", None)
    if model is None:
        return [{"model": "gpt-4o", "api_type": "openai"}]

    model_name = _model_name_of(model)
    config: dict[str, Any] = {"model": model_name, "temperature": 0.1, "api_type": "openai"}

    if settings.model_provider == "litellm":
        base_url, api_key = _litellm_connection(_model_spec_of(model, model_name))
        config["base_url"] = base_url
        config["api_key"] = api_key
        return [config]

    provider = getattr(model, "provider", getattr(model, "_provider", None))
    if provider:
        provider_base_url = getattr(provider, "base_url", None)
        if provider_base_url:
            config["base_url"] = str(provider_base_url)
        provider_api_key = getattr(provider, "api_key", None)
        if provider_api_key:
            config["api_key"] = provider_api_key
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


def _wrap_autogen_tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    def _runner(*args: Any, **kwargs: Any) -> Any:
        return _call_tool_function(fn, *args, **kwargs)

    return _runner


def _register_tool_functions(agent: PydanticAgent) -> dict[str, Callable[..., Any]]:
    """Extract pydantic_ai tool functions as AutoGen-compatible synchronous callables.

    Returns a dict mapping tool_name -> callable for use in
    function_registry or as tools in AssistantAgent. Tools that require a
    pydantic_ai ``RunContext`` are skipped.
    """
    functions: dict[str, Callable[..., Any]] = {}
    for tool in agent_tools(agent):
        fn = tool_function(tool)
        name = str(getattr(tool, "name", "") or getattr(fn, "__name__", ""))
        if not name:
            continue
        if _needs_run_context(tool, fn):
            logger.debug("Skipping tool %s: it needs a pydantic_ai RunContext", name)
            continue
        functions[name] = _wrap_autogen_tool(fn)
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
        agent.register_function(function_map={tool_name: tool_fn})

    return agent
