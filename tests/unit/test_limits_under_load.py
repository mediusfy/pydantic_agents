"""The concurrency cap, retries and timeouts when many callers arrive at once.

``test_limits.py`` covers each behaviour with one call. These drive the real provider with many concurrent callers
and a fake model that records how many runs are in flight, then check the invariants that matter in production: the
cap is never exceeded, every permit is released however a run ends, and one bad run does not hold up the rest.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
import tenacity
from pydantic_ai.exceptions import ModelHTTPError

from pydantic_agents.core.adapters import agents as agents_module
from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider
from pydantic_agents.core.settings import settings

RUN_SECONDS = 0.01


class Load:
    """Counts the runs in flight so a test can read the peak."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.runs = 0

    @asynccontextmanager
    async def run(self):
        self.runs += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            self.active -= 1


def fake_agent(load: Load, behaviour=None):
    """An agent whose ``run`` takes RUN_SECONDS under ``load``; ``behaviour(prompt, attempt)`` may raise or hang."""
    attempts: dict[str, int] = {}

    async def run(prompt, **_kwargs):
        attempts[prompt] = attempts.get(prompt, 0) + 1
        async with load.run():
            if behaviour is not None:
                await behaviour(prompt, attempts[prompt])
            await asyncio.sleep(RUN_SECONDS)
            return f"done {prompt}"

    agent = MagicMock()
    agent.run = run
    agent.attempts = attempts
    return agent


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Skip tenacity's multi-second wait between attempts without touching asyncio.sleep, which the fakes use."""
    monkeypatch.setattr(agents_module, "wait_exponential", lambda **_kwargs: tenacity.wait_none())


async def assert_all_permits_free(sem: asyncio.Semaphore, cap: int) -> None:
    """Every permit can be taken again: none leaked by a failed, timed-out or cancelled run."""
    for _ in range(cap):
        await asyncio.wait_for(sem.acquire(), timeout=1)
    assert sem.locked()


@pytest.mark.parametrize("cap", [1, 2, 5])
async def test_peak_concurrency_never_exceeds_the_cap(cap):
    load, sem = Load(), asyncio.Semaphore(cap)
    agent, provider = fake_agent(load), PydanticAIAgentProvider()

    results = await asyncio.gather(*(provider.run_agent(sem, "ba", agent, f"p{i}") for i in range(20)))

    assert results == [f"done p{i}" for i in range(20)]
    assert load.peak == cap  # saturated, and never above
    await assert_all_permits_free(sem, cap)


async def test_failures_release_permits_and_do_not_block_the_rest(monkeypatch):
    monkeypatch.setattr(settings, "agent_transport_attempts", 3)  # so a wrongly retried permanent error shows up

    async def behaviour(prompt, _attempt):
        index = int(prompt[1:])
        if index % 3 == 0:
            raise ValueError("bad output")
        if index % 3 == 1:
            raise ModelHTTPError(400, "m")

    load, sem = Load(), asyncio.Semaphore(2)
    agent, provider = fake_agent(load, behaviour), PydanticAIAgentProvider()

    results = await asyncio.gather(
        *(provider.run_agent(sem, "ba", agent, f"p{i}") for i in range(12)), return_exceptions=True
    )

    assert [type(r) for r in results[0::3]] == [ValueError] * 4
    assert [type(r) for r in results[1::3]] == [ModelHTTPError] * 4
    assert results[2::3] == [f"done p{i}" for i in (2, 5, 8, 11)]
    assert load.runs == 12  # permanent errors are not retried
    assert load.peak <= 2
    await assert_all_permits_free(sem, 2)


async def test_a_timed_out_run_releases_its_permit_and_the_queue_gets_its_full_timeout():
    """The timeout starts when a run gets its permit, not when it joins the queue."""

    async def behaviour(prompt, _attempt):
        if prompt == "hang":
            await asyncio.sleep(10)

    load, sem = Load(), asyncio.Semaphore(1)
    agent, provider = fake_agent(load, behaviour), PydanticAIAgentProvider()

    hung = asyncio.create_task(provider.run_agent(sem, "ba", agent, "hang", timeout_seconds=0.1))
    await asyncio.sleep(0.01)  # let it take the only permit
    queued = [asyncio.create_task(provider.run_agent(sem, "ba", agent, f"q{i}", timeout_seconds=0.1)) for i in range(3)]

    with pytest.raises(TimeoutError):
        await hung
    assert await asyncio.gather(*queued) == ["done q0", "done q1", "done q2"]
    assert load.peak == 1
    await assert_all_permits_free(sem, 1)


async def test_cancelling_running_and_queued_calls_releases_the_permit():
    gate = asyncio.Event()

    async def behaviour(prompt, _attempt):
        if prompt == "holder":
            await gate.wait()

    load, sem = Load(), asyncio.Semaphore(1)
    agent, provider = fake_agent(load, behaviour), PydanticAIAgentProvider()

    holder = asyncio.create_task(provider.run_agent(sem, "ba", agent, "holder"))
    await asyncio.sleep(0.01)
    waiting = asyncio.create_task(provider.run_agent(sem, "ba", agent, "waiting"))
    await asyncio.sleep(0.01)

    waiting.cancel()  # cancelled while queued: must not take or leak a permit
    holder.cancel()  # cancelled while running: must hand its permit back
    for task in (waiting, holder):
        with pytest.raises(asyncio.CancelledError):
            await task

    assert await asyncio.wait_for(provider.run_agent(sem, "ba", agent, "after"), timeout=1) == "done after"
    assert load.active == 0
    await assert_all_permits_free(sem, 1)


async def test_retries_keep_the_permit_so_the_cap_holds_while_calls_retry(monkeypatch):
    monkeypatch.setattr(settings, "agent_transport_attempts", 3)

    async def behaviour(_prompt, attempt):
        if attempt == 1:
            raise ModelHTTPError(429, "m")

    load, sem = Load(), asyncio.Semaphore(2)
    agent, provider = fake_agent(load, behaviour), PydanticAIAgentProvider()

    results = await asyncio.gather(*(provider.run_agent(sem, "ba", agent, f"p{i}") for i in range(8)))

    assert results == [f"done p{i}" for i in range(8)]
    assert load.runs == 16  # each call made exactly one failed and one successful attempt
    assert set(agent.attempts.values()) == {2}
    assert load.peak <= 2
    await assert_all_permits_free(sem, 2)


@pytest.mark.parametrize("attempts", [1, 3])
async def test_the_retry_budget_is_per_call_when_everything_fails(monkeypatch, attempts):
    monkeypatch.setattr(settings, "agent_transport_attempts", attempts)

    async def behaviour(_prompt, _attempt):
        raise ModelHTTPError(503, "m")

    load, sem = Load(), asyncio.Semaphore(3)
    agent, provider = fake_agent(load, behaviour), PydanticAIAgentProvider()

    results = await asyncio.gather(
        *(provider.run_agent(sem, "ba", agent, f"p{i}") for i in range(10)), return_exceptions=True
    )

    assert all(isinstance(r, ModelHTTPError) for r in results)
    assert load.runs == 10 * attempts  # no storm: a failing gateway costs at most N x attempts model calls
    assert set(agent.attempts.values()) == {attempts}
    assert load.peak <= 3
    await assert_all_permits_free(sem, 3)


async def test_delegation_fan_out_neither_deadlocks_nor_lets_root_runs_exceed_the_cap():
    """Children run under their parent's permit, so the cap bounds root runs, not the calls a parent fans out to."""
    cap, children_per_parent = 2, 3
    parents, children = Load(), Load()
    sem, provider = asyncio.Semaphore(cap), PydanticAIAgentProvider()
    child = fake_agent(children)

    async def parent_run(prompt, **_kwargs):
        async with parents.run():
            outputs = await asyncio.gather(
                *(provider.run_agent(sem, "child", child, f"{prompt}-c{i}") for i in range(children_per_parent))
            )
            return outputs

    parent = MagicMock()
    parent.run = parent_run

    results = await asyncio.wait_for(
        asyncio.gather(*(provider.run_agent(sem, "parent", parent, f"p{i}") for i in range(6))), timeout=5
    )

    assert len(results) == 6
    assert parents.peak == cap
    assert children.peak <= cap * children_per_parent  # the multiplication the cap does not bound
    assert children.runs == 6 * children_per_parent
    await assert_all_permits_free(sem, cap)


async def test_a_mixed_load_completes_and_leaks_nothing(monkeypatch):
    monkeypatch.setattr(settings, "agent_transport_attempts", 2)

    async def behaviour(prompt, attempt):
        kind = int(prompt[1:]) % 5
        if kind == 0:
            raise ValueError("permanent")
        if kind == 1 and attempt == 1:
            raise ModelHTTPError(429, "m")  # recovers on the retry
        if kind == 2:
            raise ModelHTTPError(503, "m")  # never recovers
        if kind == 3:
            await asyncio.sleep(10)  # hangs until the timeout

    load, sem = Load(), asyncio.Semaphore(3)
    agent, provider = fake_agent(load, behaviour), PydanticAIAgentProvider()

    results = await asyncio.gather(
        *(provider.run_agent(sem, "ba", agent, f"p{i}", timeout_seconds=0.05) for i in range(40)),
        return_exceptions=True,
    )

    kinds = [results[k::5] for k in range(5)]
    assert all(isinstance(r, ValueError) for r in kinds[0])
    assert all(isinstance(r, str) for r in kinds[1])
    assert all(isinstance(r, ModelHTTPError) for r in kinds[2])
    assert all(isinstance(r, TimeoutError) for r in kinds[3])
    assert all(isinstance(r, str) for r in kinds[4])
    # Per group of five calls: permanent 1 run, rate-limited 2, unavailable 2 (the attempts), hung 1 (timeouts are
    # not retried), healthy 1.
    assert load.runs == 8 * (1 + 2 + 2 + 1 + 1)
    assert load.peak <= 3
    assert load.active == 0
    await assert_all_permits_free(sem, 3)
