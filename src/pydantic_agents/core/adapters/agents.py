import asyncio
from abc import ABC, abstractmethod
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from pydantic_agents.core.limits import is_transient_error, permit_held, usage_limits
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
        timeout: float | None = None,
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
        timeout: float | None = None,
    ) -> Any:
        timeout = timeout or settings.agent_timeout_seconds

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception(is_transient_error),
            reraise=True,
        )
        async def _run_with_retry():
            kwargs = {"usage_limits": usage_limits(agent_name)}
            if deps is not None:
                kwargs["deps"] = deps
            return await asyncio.wait_for(agent.run(prompt, **kwargs), timeout=timeout)

        if permit_held.get():
            # Delegated call: the parent's permit already covers it.
            return await telemetry.track(agent_name, _run_with_retry)

        async with semaphore:
            token = permit_held.set(True)
            try:
                return await telemetry.track(agent_name, _run_with_retry)
            finally:
                permit_held.reset(token)
