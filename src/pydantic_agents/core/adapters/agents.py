import asyncio
from abc import ABC, abstractmethod
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from pydantic_agents.core.limits import is_transient_error, permit_held, trace_model_settings, usage_limits
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
    ) -> Any:
        pass


class PydanticAIAgentProvider(AgentProviderPort):
    """Concrete implementation using pydantic_ai for agent execution."""

    async def run_agent(
        self,
        semaphore: asyncio.Semaphore,
        agent_name: str,
        agent: Any,
        prompt: str,
        deps: Any = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        timeout_seconds = timeout_seconds or settings.agent_timeout_seconds

        @retry(
            stop=stop_after_attempt(settings.agent_transport_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception(is_transient_error),
            reraise=True,
        )
        async def _run_with_retry():
            limits = usage_limits(agent_name)
            extra = {"model_settings": ms} if (ms := trace_model_settings(agent_name)) else {}
            run = (
                agent.run(prompt, usage_limits=limits, **extra)
                if deps is None
                else agent.run(prompt, deps=deps, usage_limits=limits, **extra)
            )
            return await asyncio.wait_for(run, timeout=timeout_seconds)

        if permit_held.get():
            # Delegated call: the parent's permit already covers it. The flag is inherited by tasks created while
            # the permit is held (contextvars are copied into new tasks), so a fire-and-forget task spawned from
            # inside an agent run would bypass the limiter too: do not spawn unrelated background work from tools.
            return await telemetry.track(agent_name, _run_with_retry)

        async with semaphore:
            token = permit_held.set(True)
            try:
                return await telemetry.track(agent_name, _run_with_retry)
            finally:
                permit_held.reset(token)
