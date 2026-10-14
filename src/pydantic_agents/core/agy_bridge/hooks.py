"""hooks — Python-side hook decorators that bridge agy's lifecycle events.

This module provides decorators matching the Antigravity SDK hook patterns:
  - on_session_start / on_session_end
  - pre_turn / post_turn
  - pre_tool_call / post_tool_call
  - on_tool_error
  - on_interaction
  - on_compaction

These are extension-layer hooks — they delegate to BridgeAgent's Hooks container
and do NOT modify any pydantic_ai agent code.

Usage:
    from pydantic_agents.core.agy_bridge.hooks import pre_turn, post_turn

    @pre_turn("ux")
    async def log_prompt(prompt: str) -> HookResult | None:
        print(f"UX prompt: {prompt}")
        return None  # allow

    @post_turn("backend")
    async def log_output(output: str):
        await save_to_log(output)
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_agents.core.agy_bridge.bridge_agent import HookResult, Hooks

logger = logging.getLogger(__name__)

# Global hook registry keyed by agent name.
_hook_registry: dict[str, Hooks] = {}


# ── Registry helpers ─────────────────────────────────────────────────


def _ensure_hooks(agent_name: str) -> Hooks:
    """Get or create a Hooks container for an agent."""
    if agent_name not in _hook_registry:
        from pydantic_agents.core.agy_bridge.bridge_agent import Hooks

        _hook_registry[agent_name] = Hooks()
    return _hook_registry[agent_name]


def get_hooks(agent_name: str) -> Hooks | None:
    """Retrieve the Hooks container for a given agent, if any hooks registered."""
    return _hook_registry.get(agent_name)


def clear_hooks(agent_name: str | None = None) -> None:
    """Clear hooks for one agent or all agents."""
    if agent_name:
        _hook_registry.pop(agent_name, None)
    else:
        _hook_registry.clear()


# ── Session Hooks ────────────────────────────────────────────────────


def on_session_start(agent_name: str) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]:
    """Decorator: register a session start hook for an agent."""

    def decorator(fn: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        hooks = _ensure_hooks(agent_name)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> None:
            await fn(*args, **kwargs)

        # Fix: Must be an async function that returns HookResult or None
        async def session_start_shim(_prompt: str):
            await fn()
            return None

        hooks.pre_turn.insert(0, session_start_shim)
        logger.debug("Registered on_session_start hook for agent '%s'", agent_name)
        return wrapper

    return decorator


def on_session_end(agent_name: str) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]:
    """Decorator: register a session end hook for an agent."""

    def decorator(fn: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        hooks = _ensure_hooks(agent_name)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> None:
            await fn(*args, **kwargs)

        # Fix: Must be an async function
        async def session_end_shim(_output: str):
            await fn()

        hooks.post_turn.append(session_end_shim)
        logger.debug("Registered on_session_end hook for agent '%s'", agent_name)
        return wrapper

    return decorator


# ── Turn Hooks ───────────────────────────────────────────────────────


def pre_turn(
    agent_name: str,
) -> Callable[
    [Callable[[str], Awaitable[HookResult | None]]],
    Callable[[str], Awaitable[HookResult | None]],
]:
    """Decorator: register a pre-turn hook for an agent."""

    def decorator(
        fn: Callable[[str], Awaitable[HookResult | None]],
    ) -> Callable[[str], Awaitable[HookResult | None]]:
        hooks = _ensure_hooks(agent_name)

        @functools.wraps(fn)
        async def wrapper(prompt: str) -> HookResult | None:
            return await fn(prompt)

        hooks.pre_turn.append(wrapper)
        logger.debug("Registered pre_turn hook for agent '%s'", agent_name)
        return wrapper

    return decorator


def post_turn(
    agent_name: str,
) -> Callable[[Callable[[str], Awaitable[None]]], Callable[[str], Awaitable[None]]]:
    """Decorator: register a post-turn hook for an agent."""

    def decorator(fn: Callable[[str], Awaitable[None]]) -> Callable[[str], Awaitable[None]]:
        hooks = _ensure_hooks(agent_name)

        @functools.wraps(fn)
        async def wrapper(output: str) -> None:
            await fn(output)

        hooks.post_turn.append(wrapper)
        logger.debug("Registered post_turn hook for agent '%s'", agent_name)
        return wrapper

    return decorator


# ── Tool Hooks ───────────────────────────────────────────────────────


def pre_tool_call(
    agent_name: str,
) -> Callable[
    [Callable[[dict], Awaitable[HookResult | None]]],
    Callable[[dict], Awaitable[HookResult | None]],
]:
    """Decorator: register a pre-tool-call hook for an agent."""

    def decorator(
        fn: Callable[[dict], Awaitable[HookResult | None]],
    ) -> Callable[[dict], Awaitable[HookResult | None]]:
        hooks = _ensure_hooks(agent_name)

        @functools.wraps(fn)
        async def wrapper(tool_call: dict) -> HookResult | None:
            return await fn(tool_call)

        hooks.pre_tool_call.append(wrapper)
        logger.debug("Registered pre_tool_call hook for agent '%s'", agent_name)
        return wrapper

    return decorator


def post_tool_call(
    agent_name: str,
) -> Callable[[Callable[[dict], Awaitable[None]]], Callable[[dict], Awaitable[None]]]:
    """Decorator: register a post-tool-call hook."""

    def decorator(fn: Callable[[dict], Awaitable[None]]) -> Callable[[dict], Awaitable[None]]:
        hooks = _ensure_hooks(agent_name)

        @functools.wraps(fn)
        async def wrapper(tool_result: dict) -> None:
            await fn(tool_result)

        hooks.post_tool_call.append(wrapper)
        logger.debug("Registered post_tool_call hook for agent '%s'", agent_name)
        return wrapper

    return decorator


# ── Error / Interaction / Compaction ─────────────────────────────────


def on_tool_error(
    agent_name: str,
) -> Callable[[Callable[[Exception], Awaitable[Exception | None]]], Callable[[Exception], Awaitable[Exception | None]]]:
    """Decorator: register a tool error hook."""

    def decorator(
        fn: Callable[[Exception], Awaitable[Exception | None]],
    ) -> Callable[[Exception], Awaitable[Exception | None]]:
        @functools.wraps(fn)
        async def wrapper(exc: Exception) -> Exception | None:
            return await fn(exc)

        logger.debug("Registered on_tool_error hook for agent '%s'", agent_name)
        return wrapper

    return decorator


def on_interaction(
    agent_name: str,
) -> Callable[
    [Callable[[dict], Awaitable[None]]],
    Callable[[dict], Awaitable[None]],
]:
    """Decorator: register an interaction hook."""

    def decorator(fn: Callable[[dict], Awaitable[None]]) -> Callable[[dict], Awaitable[None]]:
        @functools.wraps(fn)
        async def wrapper(spec: dict) -> None:
            await fn(spec)

        logger.debug("Registered on_interaction hook for agent '%s'", agent_name)
        return wrapper

    return decorator


def on_compaction(
    agent_name: str,
) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]:
    """Decorator: register a compaction hook."""

    def decorator(fn: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> None:
            await fn(*args, **kwargs)

        logger.debug("Registered on_compaction hook for agent '%s'", agent_name)
        return wrapper

    return decorator
