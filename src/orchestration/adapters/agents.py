import asyncio
from abc import ABC, abstractmethod
from typing import Any

from pydantic_ai import UsageLimits
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from orchestration.telemetry import telemetry


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
        timeout: float = 3600.0,
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
        timeout: float = 3600.0,
    ) -> Any:
        async with semaphore:

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=30),
                retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
                reraise=True,
            )
            async def _run_with_retry():
                coro = (
                    agent.run(prompt, deps=deps, usage_limits=UsageLimits(request_limit=500))
                    if deps is not None
                    else agent.run(prompt, usage_limits=UsageLimits(request_limit=500))
                )
                return await asyncio.wait_for(coro, timeout=timeout)

            return await telemetry.track(agent_name, _run_with_retry)
