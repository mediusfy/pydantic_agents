import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

import logfire
from pydantic_ai import Agent

from orchestration.exceptions import (
    AgentExecutionError,
    AgentNotFoundError,
    AmbiguousCapabilityError,
)
from orchestration.session_recorder import current_run_context
from orchestration.telemetry import telemetry

logger = logging.getLogger(__name__)


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, tuple[Agent[Any, Any], list[str], Callable | None, Callable | None]] = {}
        # capability -> ordered list of agent names that claim it.
        # Multiple owners are *allowed* but query() will refuse to pick for you;
        # callers must pass owner=<agent_name> to disambiguate.
        self._cap_index: dict[str, list[str]] = {}
        self._initialized: set[str] = set()

        from orchestration.adapters.agents import PydanticAIAgentProvider
        from orchestration.settings import settings

        self._provider = PydanticAIAgentProvider()
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)

    def register(
        self,
        name: str,
        agent: Agent[Any, Any],
        capabilities: list[str],
        deps_factory: Callable | None = None,
        post_process_hook: Callable | None = None,
    ):
        if name in self._agents:
            raise ValueError(f"Agent '{name}' is already registered.")
        self._agents[name] = (agent, capabilities, deps_factory, post_process_hook)
        for cap in capabilities:
            owners = self._cap_index.setdefault(cap, [])
            if owners:
                # Escalated from WARNING to ERROR: overlapping capabilities are now
                # a hard config issue because query() will refuse to guess.
                logger.error(
                    "Capability '%s' is now claimed by multiple agents %s; "
                    "callers must pass owner=<agent_name> to query().",
                    cap,
                    [*owners, name],
                )
            owners.append(name)
        # Lazy tool wiring is keyed on agent name; clear any stale entry from a prior register/unregister cycle.
        self._initialized.discard(name)

    def unregister(self, name: str):
        agent_tuple = self._agents.pop(name, None)
        if agent_tuple:
            for cap in agent_tuple[1]:
                owners = self._cap_index.get(cap)
                if not owners:
                    continue
                if name in owners:
                    owners.remove(name)
                if not owners:
                    self._cap_index.pop(cap, None)
        self._initialized.discard(name)

    def owners_of(self, capability: str) -> list[str]:
        """Return the ordered list of agent names that claim a capability.

        Empty list means no agent provides it.
        """
        return list(self._cap_index.get(capability, []))

    def is_registered(self, name: str) -> bool:
        """Whether an agent with this name is currently in the registry."""
        return name in self._agents

    def capabilities(self) -> list[tuple[str, list[str]]]:
        """Return (capability, owners) pairs in stable (sorted) order.

        Useful for building human-facing lists (e.g. MCP capability tools)
        without reaching into the private index.
        """
        return [(cap, list(owners)) for cap, owners in sorted(self._cap_index.items())]

    def get(self, name: str) -> Any:
        agent_tuple = self._agents.get(name)
        if agent_tuple is None:
            raise ValueError(f"Agent '{name}' is not registered.")

        agent = agent_tuple[0]
        if name not in self._initialized:
            self._lazy_init(name, agent)
            self._initialized.add(name)
        return agent

    def _lazy_init(self, name: str, agent: Agent[Any, Any]):
        from orchestration.tool_assignments import get_tool_categories
        from orchestration.tools import apply_tools

        categories = get_tool_categories(name)
        if categories:
            apply_tools(agent, name, categories)

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    def items(self):
        return self._agents.items()

    def _resolve_owner(self, capability: str, owner: str | None) -> str:
        """Pick the agent that should serve a capability call.

        - No owners registered → AgentNotFoundError.
        - ``owner`` pinned and registered for the capability → use it.
        - ``owner`` pinned but not registered for the capability → AgentNotFoundError.
        - ``owner`` omitted, single owner → use it.
        - ``owner`` omitted, multiple owners → AmbiguousCapabilityError.
        """
        owners = self._cap_index.get(capability)
        if not owners:
            raise AgentNotFoundError(f"No agent found for capability '{capability}'")
        if owner is not None:
            if owner not in owners:
                raise AgentNotFoundError(
                    f"Agent '{owner}' does not provide capability '{capability}'. Available owners: {owners}"
                )
            return owner
        if len(owners) > 1:
            raise AmbiguousCapabilityError(capability, owners)
        return owners[0]

    async def query(
        self,
        capability: str,
        prompt: str,
        timeout: float | None = None,
        deps: dict | None = None,
        owner: str | None = None,
    ) -> str:
        """Query an agent by capability, returning its response as a string.

        Pass ``owner=<agent_name>`` to pin a specific agent when a capability has
        multiple providers; otherwise an ambiguous capability raises
        :class:`AmbiguousCapabilityError`.

        Raises :class:`AgentNotFoundError` if no agent (or no matching agent) claims
        the capability, :class:`AmbiguousCapabilityError` on unpinned overlap, and
        :class:`AgentExecutionError` if the agent run itself fails.
        """
        agent_name = self._resolve_owner(capability, owner)
        agent, _caps, deps_factory, _post_hook = self._agents[agent_name]
        try:
            resolved_deps = deps or {}
            if deps_factory and not deps:
                resolved_deps = deps_factory() if callable(deps_factory) else {}

            result = await self._provider.run_agent(
                semaphore=self._semaphore,
                agent_name=agent_name,
                agent=agent,
                prompt=prompt,
                deps=resolved_deps,
                timeout=timeout or 3600.0,
            )
            if hasattr(result, "data"):
                return str(result.data)
            return str(result.output) if hasattr(result, "output") else str(result)
        except TimeoutError as e:
            raise AgentExecutionError(
                f"Agent '{agent_name}' timed out for capability '{capability}'", is_retryable=True
            ) from e
        except Exception as e:
            logger.warning("Agent '%s' failed for capability '%s': %s", agent_name, capability, e)
            raise AgentExecutionError(f"Agent '{agent_name}' failed: {e}", is_retryable=False) from e


async def instrumented_query(
    capability: str,
    prompt: str,
    span_name: str = "registry.query",
    attributes: dict[str, Any] | None = None,
    timeout: float | None = None,
    delegation_depth: int = 0,
    owner: str | None = None,
) -> str:
    """Query an agent by capability with full Logfire + JSONL telemetry instrumentation.

    This is the recommended entry point for MCP tools and pipeline stages.
    It wraps registry.query() with:
    - A Logfire span recording the capability, agent name, and timing
    - telemetry.track() for JSONL telemetry
    - Error attribution to the right agent
    - Delegation depth propagation to prevent infinite delegation loops
    - Optional ``owner`` pin so pipelines can target a specific agent when a
      capability has multiple providers.
    """
    # Resolve up-front so the span attribute is the *actual* agent that will run,
    # not the speculative owners[0]. If resolution fails, the exception surfaces
    # before we open a span, which keeps traces honest.
    agent_name = registry._resolve_owner(capability, owner)
    span_attrs: dict[str, Any] = {
        "capability": capability,
        "agent_name": agent_name,
        "prompt_length": len(prompt),
        **(attributes or {}),
    }

    start = time.time()
    success = False

    # Propagate delegation depth into context so the callee inherits the counter.
    # Defensive `or {}` belt-and-suspenders the ContextVar default — once
    # session_recorder.current_run_context defaults to {}, this is a no-op, but
    # it also protects against a caller deliberately setting None.
    ctx_data = current_run_context.get() or {}
    token = None
    if delegation_depth > 0:
        token = current_run_context.set(
            {
                **ctx_data,
                "delegation_depth": delegation_depth,
            }
        )

    try:
        with logfire.span(span_name, **span_attrs, _level="info"):
            result = await registry.query(capability, prompt, timeout=timeout, owner=owner)
            success = True
            return result
    finally:
        if token is not None:
            current_run_context.reset(token)
        duration = time.time() - start
        with contextlib.suppress(Exception):
            await telemetry.log_call(agent_name, duration, success, metadata=span_attrs)


registry = AgentRegistry()
