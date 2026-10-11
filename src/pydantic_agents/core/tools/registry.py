import inspect
import logging
import time
from collections.abc import Callable
from functools import wraps

import logfire

from pydantic_agents.core.session_recorder import current_run_context

logger = logging.getLogger(__name__)


class ToolDef:
    def __init__(self, name: str, fn: Callable, description: str, categories: list[str]):
        self.name = name
        self.fn = fn
        self.description = description
        self.categories = categories


tool_registry: dict[str, ToolDef] = {}


def _build_span_attrs(tool_name: str, kwargs: dict) -> dict:
    run_ctx = current_run_context.get() or {}
    attrs = {
        "tool_name": tool_name,
        "arg_count": len(kwargs),
        **{f"run_{k}": v for k, v in run_ctx.items() if v},
    }
    for key in ("prompt", "requirements", "context_summary"):
        if key in kwargs:
            attrs[f"arg_{key}_len"] = len(kwargs[key])
    for key in ("component_name", "test_id", "pattern", "capability"):
        if key in kwargs:
            attrs[f"arg_{key}"] = kwargs[key]
    return attrs


def _wrap_tool_with_tracing(fn: Callable, tool_name: str) -> Callable:
    is_async = inspect.iscoroutinefunction(fn)

    if is_async:

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            attrs = _build_span_attrs(tool_name, kwargs)
            start = time.time()
            try:
                with logfire.span(f"tool.{tool_name}", **attrs, _level="info"):
                    result = await fn(*args, **kwargs)
                duration = time.time() - start
                if isinstance(result, (str, list)):
                    logger.debug(f"tool.{tool_name} completed in {duration:.1f}s, size={len(result)}")
                return result
            except Exception as e:
                duration = time.time() - start
                logger.warning(f"tool.{tool_name} failed in {duration:.1f}s: {e}")
                raise

        return async_wrapper

    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        attrs = _build_span_attrs(tool_name, kwargs)
        start = time.time()
        try:
            with logfire.span(f"tool.{tool_name}", **attrs, _level="info"):
                result = fn(*args, **kwargs)
            duration = time.time() - start
            if isinstance(result, (str, list)):
                logger.debug(f"tool.{tool_name} completed in {duration:.1f}s, size={len(result)}")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.warning(f"tool.{tool_name} failed in {duration:.1f}s: {e}")
            raise

    return sync_wrapper


def register_tool(name: str, fn: Callable, description: str, categories: list[str] | None = None):
    categories = categories or []
    tool_registry[name] = ToolDef(name, fn, description, categories)


def get_tools_for_agent(agent_name: str, categories: list[str]) -> list[tuple[str, Callable]]:
    tools = []
    for tool_def in tool_registry.values():
        if any(cat in tool_def.categories for cat in categories) or agent_name in tool_def.categories:
            tools.append((tool_def.name, tool_def.fn))
    return tools


def _tool_takes_context(fn: Callable) -> bool:
    """Check if the first param of fn is annotated with RunContext."""
    import typing

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        return False
    first = params[0]
    if first.annotation is sig.empty:
        return False
    hints = typing.get_type_hints(fn)
    ann = hints.get(first.name, first.annotation)
    from pydantic_ai import RunContext

    return ann is RunContext or getattr(ann, "__origin__", None) is RunContext


def apply_tools(agent, agent_name: str, categories: list[str]):
    tools = get_tools_for_agent(agent_name, categories)
    for name, fn in tools:
        if not hasattr(agent, "_function_toolset") or name not in agent._function_toolset.tools:
            wrapped = _wrap_tool_with_tracing(fn, name)
            if _tool_takes_context(fn):
                agent.tool(wrapped)
            else:
                agent.tool_plain(wrapped)
