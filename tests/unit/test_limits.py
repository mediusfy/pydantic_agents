"""Retry policy, usage caps and the shared LLM concurrency limiter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
from pydantic_agents.core.limits import is_transient_error, llm_semaphore, usage_limits
from pydantic_agents.core.settings import settings


@pytest.mark.parametrize(
    "exc, transient",
    [
        (ModelHTTPError(429, "m"), True),
        (ModelHTTPError(500, "m"), True),
        (ModelHTTPError(502, "m"), True),
        (ModelHTTPError(503, "m"), True),
        (ModelHTTPError(504, "m"), True),
        (ModelHTTPError(400, "m"), False),
        (ModelHTTPError(401, "m"), False),
        (ModelAPIError("m", "connection reset"), True),
        (ConnectionError("boom"), True),
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


async def test_llm_semaphore_is_shared_and_sized_by_settings():
    sem = llm_semaphore()
    assert llm_semaphore() is sem
    for _ in range(settings.max_concurrent_llm_calls):
        await asyncio.wait_for(sem.acquire(), timeout=1)
    assert sem.locked()
    for _ in range(settings.max_concurrent_llm_calls):
        sem.release()


def test_each_event_loop_gets_its_own_semaphore():
    async def grab():
        return llm_semaphore()

    first, second = asyncio.run(grab()), asyncio.run(grab())
    assert first is not second


def test_role_request_limits_name_registered_agents():
    from pydantic_agents.agents.registration import register_all_agents
    from pydantic_agents.core.limits import _ROLE_REQUEST_LIMITS
    from pydantic_agents.core.registry import registry

    register_all_agents()
    assert set(_ROLE_REQUEST_LIMITS) <= set(registry.list_agents())


@pytest.mark.parametrize(
    "errors",
    [
        [ModelHTTPError(429, "m"), ModelHTTPError(503, "m")],
        [ModelHTTPError(500, "m")],
        [ModelHTTPError(502, "m"), ModelHTTPError(504, "m")],
        [ConnectionError("reset")],
    ],
)
async def test_provider_retries_transient_errors_until_success(monkeypatch, errors):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())  # skip tenacity's backoff
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=[*errors, "ok"])

    result = await PydanticAIAgentProvider().run_agent(asyncio.Semaphore(1), "ba", agent, "hi")

    assert result == "ok"
    assert agent.run.await_count == len(errors) + 1
    assert agent.run.await_args.kwargs["usage_limits"].request_limit == settings.agent_request_limit


@pytest.mark.parametrize("error", [ModelHTTPError(400, "m"), ModelHTTPError(401, "m"), ValueError("bad output")])
async def test_provider_does_not_retry_permanent_errors(monkeypatch, error):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=[error, "ok"])

    provider, sem = PydanticAIAgentProvider(), asyncio.Semaphore(1)
    with pytest.raises(type(error)):
        await provider.run_agent(sem, "ba", agent, "hi")

    assert agent.run.await_count == 1


async def test_provider_gives_up_after_the_configured_attempts(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=ModelHTTPError(429, "m"))

    provider, sem = PydanticAIAgentProvider(), asyncio.Semaphore(1)
    with pytest.raises(ModelHTTPError):
        await provider.run_agent(sem, "ba", agent, "hi")

    assert agent.run.await_count == settings.agent_transport_attempts


async def test_provider_does_not_retry_timeouts():
    async def slow(*_a, **_k):
        await asyncio.sleep(5)

    agent = MagicMock()
    agent.run = AsyncMock(side_effect=slow)
    provider, sem = PydanticAIAgentProvider(), asyncio.Semaphore(1)
    with pytest.raises(TimeoutError):
        await provider.run_agent(sem, "ba", agent, "hi", timeout_seconds=0.05)
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
    assert not sem.locked()  # the permit was released


async def test_langgraph_bridge_runs_agents_under_the_shared_limiter():
    from pydantic_agents.core.adapters.langgraph_bridge import as_langgraph_node
    from pydantic_agents.core.models.core import PipelineState

    seen = {}

    async def run_agent(semaphore, agent_name, agent, prompt, **_kwargs):
        seen["semaphore"] = semaphore
        raise RuntimeError("stop here")  # the delta shape is covered elsewhere; we only check the limiter

    provider = MagicMock(run_agent=run_agent)
    with patch("pydantic_agents.core.adapters.langgraph_bridge.registry.get", return_value=MagicMock()):
        await as_langgraph_node(
            PipelineState(run_id="r", stage="init", requirement="x"), agent_name="ba", provider=provider
        )

    assert seen["semaphore"] is llm_semaphore()
