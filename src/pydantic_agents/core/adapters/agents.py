import asyncio
from abc import ABC, abstractmethod
from typing import Any

import logfire
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from pydantic_agents.core.limits import is_transient_error, permit_held, run_id_var, run_model_settings, usage_limits
from pydantic_agents.core.settings import settings
from pydantic_agents.core.telemetry import telemetry


class AgentProviderPort(ABC):
    """Abstracts the interaction with the underlying LLM framework."""

    @abstractmethod
    async def run_agent(
        self,
        semaphore: asyncio.Semaphore,
        agent_name: str,
        agent: Any,
        prompt: str,
        deps: Any = None,
        timeout_seconds: float | None = None,
        usage: Any = None,
    ) -> Any:
        pass


class PydanticAIAgentProvider(AgentProviderPort):
    """Concrete implementation using pydantic_ai for agent execution.

    This is the only place that calls ``Agent.run``: it applies the concurrency permit, the per-run usage caps, the
    timeout, the transport retry, the per-call model settings (output cap, Langfuse session tags), the JSONL
    telemetry and an ``agent.run`` span tagged for Langfuse. ``usage`` lets a delegated call share its caller's
    budget instead of getting a fresh one.
    """

    async def run_agent(
        self,
        semaphore: asyncio.Semaphore,
        agent_name: str,
        agent: Any,
        prompt: str,
        deps: Any = None,
        timeout_seconds: float | None = None,
        usage: Any = None,
    ) -> Any:
        timeout_seconds = timeout_seconds or settings.agent_timeout_seconds

        @retry(
            stop=stop_after_attempt(settings.agent_transport_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception(is_transient_error),
            reraise=True,
        )
        async def _run_with_retry():
            kwargs: dict[str, Any] = {"usage_limits": usage_limits(agent_name)}
            if (ms := run_model_settings(agent_name)) is not None:
                kwargs["model_settings"] = ms
            if deps is not None:
                kwargs["deps"] = deps
            if usage is not None:
                kwargs["usage"] = usage
            return await asyncio.wait_for(agent.run(prompt, **kwargs), timeout=timeout_seconds)

        async def _traced():
            tags = {
                "langfuse.session.id": run_id_var.get() or "",
                "langfuse.trace.name": f"pydantic-agents/{agent_name}",
            }
            with logfire.span("agent.run", agent_name=agent_name, prompt_length=len(prompt), **tags, _level="info"):
                return await telemetry.track(agent_name, _run_with_retry)

        if permit_held.get():
            # Delegated call: the parent's permit already covers it. The flag is inherited by tasks created while
            # the permit is held (contextvars are copied into new tasks), so a fire-and-forget task spawned from
            # inside an agent run would bypass the limiter too: do not spawn unrelated background work from tools.
            return await _traced()

        async with semaphore:
            token = permit_held.set(True)
            try:
                return await _traced()
            finally:
                permit_held.reset(token)
