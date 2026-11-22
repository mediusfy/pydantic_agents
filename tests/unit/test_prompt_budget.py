"""The prompt character budget enforced in core/adapters/agents.py before Agent.run.

Origin: a `cto` run billed 555k input tokens because context assembly was unbounded (PR #84, "Not addressed here").
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_agents.core.adapters.agents import PydanticAIAgentProvider, cap_prompt, prompt_char_budget
from pydantic_agents.core.settings import Settings, settings


@pytest.mark.parametrize(
    ("name", "token_budget", "chars_per_token", "want"),
    [
        ("default settings", 60_000, 4.0, 240_000),
        ("stricter ratio", 60_000, 2.5, 150_000),
        ("disabled by zero budget", 0, 4.0, 0),
        ("disabled by zero ratio", 60_000, 0.0, 0),
        ("disabled by negative budget", -1, 4.0, 0),
    ],
)
def test_prompt_char_budget(monkeypatch, name, token_budget, chars_per_token, want):
    monkeypatch.setattr(settings, "agent_prompt_token_budget", token_budget)
    monkeypatch.setattr(settings, "agent_prompt_chars_per_token", chars_per_token)
    assert prompt_char_budget() == want, name


def test_default_budget_is_below_the_token_alert_threshold():
    """A single prompt at the budget must not by itself trip the per-run token tripwire."""
    assert Settings.model_fields["agent_prompt_token_budget"].default < (
        Settings.model_fields["agent_token_alert_threshold"].default
    )


@pytest.mark.parametrize(
    ("name", "token_budget", "prompt_len", "want_capped"),
    [
        ("normal-sized run passes through raw", 1_000, 400, False),
        ("exactly at the budget passes through raw", 1_000, 4_000, False),
        ("oversized run is capped", 1_000, 40_000, True),
        ("555k-token-class context is capped", 1_000, 2_220_000, True),
        ("cap switched off passes oversized through raw", 0, 40_000, False),
    ],
)
def test_cap_prompt(monkeypatch, name, token_budget, prompt_len, want_capped):
    monkeypatch.setattr(settings, "agent_prompt_token_budget", token_budget)
    monkeypatch.setattr(settings, "agent_prompt_chars_per_token", 4.0)
    head, tail = "TASK-HEAD ", " TASK-TAIL"
    prompt = head + "x" * (prompt_len - len(head) - len(tail)) + tail

    capped = cap_prompt("cto", prompt)

    if not want_capped:
        assert capped == prompt, name
        return
    assert capped != prompt, name
    assert len(capped) <= prompt_char_budget(), name
    assert "TRUNCATED" in capped, name
    # The task framing at both ends survives: the head is the system framing, the tail is the request itself.
    assert capped.startswith(head), name
    assert capped.endswith(tail), name


async def test_run_agent_sends_the_capped_prompt_to_the_model(monkeypatch):
    """The guardrail sits in run_agent, so no caller can reach the model with an unbounded prompt."""
    monkeypatch.setattr(settings, "agent_prompt_token_budget", 1_000)
    monkeypatch.setattr(settings, "agent_prompt_chars_per_token", 4.0)
    agent = MagicMock()
    agent.run = AsyncMock(return_value="ok")
    oversized = "y" * 500_000

    await PydanticAIAgentProvider().run_agent(asyncio.Semaphore(1), "cto", agent, oversized)

    sent = agent.run.await_args.args[0]
    assert sent != oversized
    assert len(sent) <= 4_000
    assert "TRUNCATED" in sent


async def test_run_agent_leaves_a_normal_prompt_untouched(monkeypatch):
    monkeypatch.setattr(settings, "agent_prompt_token_budget", 1_000)
    monkeypatch.setattr(settings, "agent_prompt_chars_per_token", 4.0)
    agent = MagicMock()
    agent.run = AsyncMock(return_value="ok")

    await PydanticAIAgentProvider().run_agent(asyncio.Semaphore(1), "cto", agent, "review this design")

    assert agent.run.await_args.args[0] == "review this design"
