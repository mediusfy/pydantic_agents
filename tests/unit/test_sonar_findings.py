"""Regression tests for the bugs a local Sonar scan reported."""

import types
from unittest.mock import MagicMock

import pytest

from pydantic_agents.core.exceptions import AgentExecutionError
from pydantic_agents.core.session_recorder import current_run_context
from pydantic_agents.core.tools.docs import update_documentation


def test_run_context_default_is_immutable():
    default = current_run_context.get()
    assert isinstance(default, types.MappingProxyType)
    with pytest.raises(TypeError):
        default["leak"] = "state"  # type: ignore[index]


def test_run_context_default_is_still_usable_as_a_mapping():
    assert current_run_context.get().get("agent_name") is None
    assert {**current_run_context.get(), "agent_name": "x"} == {"agent_name": "x"}


def _bridge_without_agent():
    from pydantic_agents.core.agy_bridge.bridge_agent import BridgeAgent

    agent = BridgeAgent.__new__(BridgeAgent)
    agent._name = "ghost"
    agent._agent = None
    agent._resolve = MagicMock()
    agent._hooks = MagicMock(pre_turn=[], post_turn=[], pre_tool_call=[], post_tool_call=[])
    return agent


async def test_bridge_agent_without_a_resolved_agent_raises_a_typed_error():
    agent = _bridge_without_agent()
    with pytest.raises(AgentExecutionError, match="ghost"):
        await agent.run("hi")


async def test_unresolved_agent_failure_is_still_recorded():
    """The failure must mark the conversation failed and log a failed call, like any other run failure."""
    from unittest.mock import AsyncMock, patch

    from pydantic_agents.core.agy_bridge.bridge_agent import Conversation

    agent = _bridge_without_agent()
    conversation = Conversation(
        conversation_id="c1", agent_name="ghost", model="m", history=[], created_at=0.0, updated_at=0.0, status="active"
    )
    with (
        patch("pydantic_agents.core.telemetry.telemetry.log_call", AsyncMock()) as log_call,
        pytest.raises(AgentExecutionError),
    ):
        await agent.run("hi", conversation=conversation)

    assert conversation.status == "failed"
    assert log_call.await_args.args[2] is False


@pytest.mark.parametrize(
    "decision, expected",
    [("deny", "Hook denied: nope"), ("modify", "ok")],
)
async def test_pre_turn_hooks_can_deny_or_modify(decision, expected):
    from unittest.mock import AsyncMock

    agent = _bridge_without_agent()
    if decision == "deny":
        agent._hooks.pre_turn = [AsyncMock(return_value=MagicMock(allow=False, message="nope"))]
        assert await agent.run("hi") == expected
    else:
        hook = AsyncMock(return_value=MagicMock(allow=True, modified_input="rewritten"))
        agent._hooks.pre_turn = [hook, AsyncMock(return_value=None)]
        modified, denied = await agent._run_pre_turn_hooks("hi")
        assert (modified, denied) == ("rewritten", None)


@pytest.mark.parametrize(
    "mode, first, second, expected",
    [
        ("append", "one", "two", "\n\none\n\ntwo"),
        ("overwrite", "one", "two", "two"),
    ],
)
async def test_update_documentation_modes(tmp_path, monkeypatch, mode, first, second, expected):
    monkeypatch.chdir(tmp_path)
    await update_documentation(MagicMock(), "adr/001.md", first, mode="append")
    result = await update_documentation(MagicMock(), "adr/001.md", second, mode=mode)

    assert result == "Documentation updated: adr/001.md"
    assert (tmp_path / "docs" / "adr" / "001.md").read_text() == expected


async def test_update_documentation_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = await update_documentation(MagicMock(), "../../etc/evil.md", "x")
    assert result.startswith("Error")
    assert not (tmp_path / "etc").exists()


@pytest.mark.parametrize("mode", ["prepend", "", "OVERWRITE"])
async def test_update_documentation_rejects_unknown_modes(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    result = await update_documentation(MagicMock(), "a.md", "x", mode=mode)
    assert result.startswith("Error: mode must be one of")
    assert not (tmp_path / "docs").exists()
