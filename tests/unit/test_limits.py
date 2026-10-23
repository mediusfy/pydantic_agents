"""Retry policy, usage caps and the shared LLM concurrency limiter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
from pydantic_agents.core.limits import is_transient_error, llm_semaphore, usage_limits
from pydantic_agents.core.settings import settings


@pytest.mark.parametrize(
    "exc, transient",
    [
        (ModelHTTPError(429, "m"), True),
        (ModelHTTPError(503, "m"), True),
        (ModelHTTPError(400, "m"), False),
        (ModelHTTPError(401, "m"), False),
        (ModelAPIError("m", "connection reset"), True),
        (ConnectionError("boom"), True),
        (TimeoutError(), False),
        (TimeoutError(), False),
        (ValueError("x"), False),
    ],
)
def test_is_transient_error(exc, transient):
    assert is_transient_error(exc) is transient


@pytest.mark.parametrize(
    "agent, expected",
    [("ba", settings.agent_request_limit), ("backend", 60), (None, settings.agent_request_limit)],
)
def test_usage_limits_are_per_role_and_capped(agent, expected):
    limits = usage_limits(agent)
    assert limits.request_limit == expected
    assert limits.total_tokens_limit == settings.agent_total_tokens_limit
    assert expected < 500


def test_defaults_are_tighter_than_before():
    assert settings.agent_retries == 2
    assert settings.agent_timeout_seconds == 600.0


async def test_llm_semaphore_is_shared():
    assert llm_semaphore() is llm_semaphore()
    assert llm_semaphore()._value == settings.max_concurrent_llm_calls  # noqa: SLF001


@pytest.mark.parametrize(
    "errors, expected_calls, succeeds",
    [
        ([ModelHTTPError(429, "m"), ModelHTTPError(503, "m")], 3, True),
        ([ModelHTTPError(400, "m")], 1, False),
        ([ValueError("bad output")], 1, False),
    ],
)
async def test_provider_retries_only_transient_errors(monkeypatch, errors, expected_calls, succeeds):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())  # skip tenacity's backoff
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=[*errors, "ok"])

    provider = PydanticAIAgentProvider()
    coro = provider.run_agent(asyncio.Semaphore(1), "ba", agent, "hi")
    if succeeds:
        assert await coro == "ok"
    else:
        with pytest.raises(type(errors[0])):
            await coro
    assert agent.run.await_count == expected_calls
    assert agent.run.await_args.kwargs["usage_limits"].request_limit == settings.agent_request_limit


async def test_provider_does_not_retry_timeouts():
    async def slow(*_a, **_k):
        await asyncio.sleep(5)

    agent = MagicMock()
    agent.run = AsyncMock(side_effect=slow)
    with pytest.raises(TimeoutError):
        await PydanticAIAgentProvider().run_agent(asyncio.Semaphore(1), "ba", agent, "hi", timeout=0.05)
    assert agent.run.await_count == 1


async def test_delegated_call_does_not_deadlock_on_a_single_permit():
    """A parent holding the only permit delegates to a child that goes through the same limiter."""
    provider = PydanticAIAgentProvider()
    sem = asyncio.Semaphore(1)

    child = MagicMock()
    child.run = AsyncMock(return_value="child")

    async def parent_run(prompt, **_kwargs):
        return await provider.run_agent(sem, "child", child, prompt)

    parent = MagicMock()
    parent.run = parent_run

    assert await asyncio.wait_for(provider.run_agent(sem, "parent", parent, "go"), timeout=3) == "child"
    assert sem._value == 1  # noqa: SLF001 -- the permit was released
