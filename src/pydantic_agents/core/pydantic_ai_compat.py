"""The one place that reads pydantic_ai's private tool storage.

pydantic_ai has no public API for "the function tools attached to this Agent", so the AutoGen, CrewAI and agy
bridges, and ``tools.registry.apply_tools``, need ``Agent._function_toolset.tools`` and ``Tool.function``. The
previous private attribute (``_function_tools``) disappeared in a pydantic_ai upgrade, and every bridge silently
saw zero tools. The helpers here fail loudly instead:

- ``agent_tools`` raises ``PydanticAICompatibilityError`` for a real ``pydantic_ai.Agent`` that does not expose the
  expected shape (non-Agent duck-typed objects in tests may still use the legacy ``_function_tools`` list);
- ``tool_function`` raises when a tool has no callable ``function``;
- ``check_pydantic_ai_compatibility`` builds a throwaway agent with one tool and checks both, and runs once from
  ``register_all_agents``, so an incompatible upgrade breaks startup, ``make smoke`` and CI, not a pipeline run.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

from pydantic_agents.core.exceptions import PydanticAICompatibilityError


def _is_pydantic_ai_agent(agent: Any) -> bool:
    from pydantic_ai import Agent

    return isinstance(agent, Agent)


def agent_tools(agent: Any) -> list[Any]:
    """The function-tool objects attached to ``agent`` (pydantic_ai ``Tool`` instances), in registration order."""
    if agent is None:
        return []
    toolset = getattr(agent, "_function_toolset", None)
    tool_map = getattr(toolset, "tools", None)
    if isinstance(tool_map, dict):
        return list(tool_map.values())
    if _is_pydantic_ai_agent(agent):
        raise PydanticAICompatibilityError(
            "pydantic_ai.Agent no longer exposes _function_toolset.tools as a dict; the installed pydantic_ai is "
            "not compatible with core/pydantic_ai_compat.py (tools would silently disappear from every bridge)."
        )
    legacy = getattr(agent, "_function_tools", None)  # duck-typed stand-ins used by tests
    return list(legacy) if isinstance(legacy, (list, tuple)) else []


def agent_tool_names(agent: Any) -> set[str]:
    """Names of the function tools attached to ``agent``."""
    return {str(getattr(t, "name", "")) for t in agent_tools(agent)} - {""}


def tool_function(tool: Any) -> Callable[..., Any]:
    """The Python callable behind a pydantic_ai ``Tool``."""
    fn = getattr(tool, "function", None)
    if callable(fn):
        return fn
    raise PydanticAICompatibilityError(
        f"pydantic_ai tool {getattr(tool, 'name', tool)!r} has no callable 'function' attribute; the installed "
        "pydantic_ai is not compatible with core/pydantic_ai_compat.py."
    )


@cache
def check_pydantic_ai_compatibility() -> None:
    """Raise ``PydanticAICompatibilityError`` unless a real Agent exposes its tools the way the bridges read them."""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    probe = Agent(TestModel(), defer_model_check=True)

    @probe.tool_plain
    def _compat_probe() -> str:
        return "ok"

    tools = agent_tools(probe)
    if [getattr(t, "name", None) for t in tools] != ["_compat_probe"]:
        raise PydanticAICompatibilityError(
            f"pydantic_ai compatibility probe expected one tool '_compat_probe', found {tools!r}."
        )
    if tool_function(tools[0])() != "ok":
        raise PydanticAICompatibilityError(
            "pydantic_ai compatibility probe: Tool.function is not the registered function."
        )
