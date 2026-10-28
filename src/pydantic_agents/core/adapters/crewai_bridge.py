"""Bridge: pydantic_ai Agents → CrewAI Agent wrappers.

Each pydantic_ai Agent is adapted into a CrewAI Agent by extracting its
system prompt (as backstory), model config, and tool set. The resulting
CrewAI agent talks to the same LiteLLM proxy as the pydantic_ai agent.

``crewai`` is an optional extra: it is imported only inside the functions
that need it, so this module can be imported without it installed.
"""

import asyncio
import atexit
import inspect
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic_ai import Agent as PydanticAgent

from pydantic_agents.core.models.factory import _litellm_connection
from pydantic_agents.core.settings import settings

logger = logging.getLogger(__name__)


def _model_name_of(model: Any) -> str:
    """Model name without a provider prefix, without building the client."""
    if isinstance(model, str):
        return model.partition(":")[2] or model
    name = getattr(model, "model_name", None)
    if isinstance(name, str) and name:
        return name
    return str(model)


def _model_spec_of(model: Any, model_name: str) -> str:
    if isinstance(model, str):
        return model
    spec = getattr(model, "spec", None)
    return spec if isinstance(spec, str) and spec else f"openai:{model_name}"


def build_crewai_llm(pydantic_agent: PydanticAgent) -> Any:
    """Build the LLM handle CrewAI should use for a pydantic_ai Agent.

    With ``MODEL_PROVIDER=litellm`` CrewAI gets a ``crewai.LLM`` pointed at the
    LiteLLM proxy. CrewAI's litellm SDK needs a provider prefix, and the proxy
    speaks the OpenAI protocol, so the model is ``openai/<model group>``.
    For any other provider the bare model name is returned and CrewAI
    resolves the provider itself.
    """
    model = getattr(pydantic_agent, "_model", None)
    if model is None:
        raise ValueError("pydantic_ai agent has no model; cannot build a CrewAI LLM")
    model_name = _model_name_of(model)
    if settings.model_provider != "litellm":
        return model_name

    base_url, api_key = _litellm_connection(_model_spec_of(model, model_name))
    try:
        from crewai import LLM
    except ImportError:
        raise RuntimeError("crewai is not installed. Run: uv sync --extra crewai") from None
    return LLM(model=f"openai/{model_name}", base_url=base_url, api_key=api_key)


def _extract_backstory(agent: PydanticAgent) -> str:
    """Extract the system prompt from a pydantic_ai Agent as a CrewAI backstory."""
    prompts = getattr(agent, "_system_prompts", None) or []
    parts = [str(p) for p in prompts if p]
    return "\n\n".join(parts)


def _extract_tool_defs(agent: PydanticAgent) -> list[Any]:
    """Bridge every function tool of a pydantic_ai Agent that CrewAI can call."""
    tools: list[Any] = []
    toolset = getattr(agent, "_function_toolset", None)
    tool_map = getattr(toolset, "tools", None) or {}
    for tool in tool_map.values():
        try:
            crew_tool = _pydantic_tool_to_crewai(tool)
        except Exception as exc:
            logger.debug("Skipping tool %s: %s", getattr(tool, "name", tool), exc)
            continue
        if crew_tool is not None:
            tools.append(crew_tool)
    return tools


def _needs_run_context(pydantic_tool: Any, fn: Callable[..., Any]) -> bool:
    """Whether the tool function expects a pydantic_ai ``RunContext`` as its first argument."""
    takes_ctx = getattr(pydantic_tool, "takes_ctx", None)
    if takes_ctx is not None:
        return bool(takes_ctx)
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if not params:
        return False
    first = params[0]
    return first.name == "ctx" or "RunContext" in str(first.annotation)


async def _await(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


# One bounded pool for every bridged tool call, so a crew making many parallel calls does not start a thread per
# call. Each call still runs its coroutine on its own short-lived event loop inside the worker. More than
# max_workers concurrent calls queue (they never deadlock: a bridged call does not wait on another bridged call).
# The pool is shut down at interpreter exit without waiting, so a hung tool cannot block shutdown.
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="crewai-tool")
atexit.register(_TOOL_EXECUTOR.shutdown, wait=False)


def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
    """Resolve an awaitable from synchronous code.

    CrewAI calls tools synchronously. Without a running loop in this thread
    ``asyncio.run`` is enough; inside a running loop (CrewAI driven from an
    async caller) the coroutine runs on its own loop in a worker thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(awaitable))
    return _TOOL_EXECUTOR.submit(asyncio.run, _await(awaitable)).result()


def _call_tool_function(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return _run_awaitable(result)
    return result


def _pydantic_tool_to_crewai(pydantic_tool: Any) -> Any:
    """Convert a single pydantic_ai Tool to a CrewAI tool.

    Returns ``None`` for tools CrewAI cannot call: those whose function needs a
    pydantic_ai ``RunContext`` (there is none inside a crew).
    """
    try:
        from crewai.tools import BaseTool
    except ImportError:
        return None

    candidate = getattr(pydantic_tool, "function", None)
    if candidate is None or not callable(candidate):
        return None
    fn: Callable[..., Any] = candidate

    tool_name = str(getattr(pydantic_tool, "name", "") or getattr(fn, "__name__", "unknown_tool"))
    if _needs_run_context(pydantic_tool, fn):
        logger.debug("Skipping tool %s: it needs a pydantic_ai RunContext", tool_name)
        return None
    tool_description = str(getattr(pydantic_tool, "description", "") or f"Run the {tool_name} tool")

    class _WrappedTool(BaseTool):
        name: str = tool_name
        description: str = tool_description

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            return _call_tool_function(fn, *args, **kwargs)

    return _WrappedTool()


def as_crewai_agent(
    pydantic_agent: PydanticAgent,
    *,
    role: str,
    goal: str,
    allow_delegation: bool = False,
    verbose: bool = True,
) -> Any:
    """Wrap a pydantic_ai Agent as a CrewAI Agent."""
    try:
        from crewai import Agent as CrewAgent
    except ImportError:
        raise RuntimeError("crewai is not installed. Run: uv sync --extra crewai") from None

    return CrewAgent(
        role=role,
        goal=goal,
        backstory=_extract_backstory(pydantic_agent),
        llm=build_crewai_llm(pydantic_agent),
        tools=_extract_tool_defs(pydantic_agent),
        allow_delegation=allow_delegation,
        verbose=verbose,
    )
